
"""FlowBP-Sparse trainer.

Generalises the FlowBP-Lagrange connector idea to the full sampling chain:

1. Roll out the full N-step trajectory once with no_grad and cache
   (x_history, v_history).
2. Sample K distinct timesteps with a late-biased categorical
   distribution (closer to clean image gets higher prob, but every
   index is reachable).
3. Re-forward the transformer at those K cached x_k's, replacing the
   cached velocities with the new differentiable ones.
4. Compose the trajectory in fp32 by full Euler integration:
       x_0_hat = x_init + sum_i (sigma_{i+1} - sigma_i) * v_i
   where v_i is the active grad-bearing velocity if i is selected,
   otherwise v_history[i].detach().
5. Straight-through residual replaces the forward value with the cached
   true x_0; backward flows through x_0_hat into the K active
   velocities only.
6. Reward gradient on x_0_hat -> backward into the K transformer calls
   for those indices, each with effective coefficient
   `(sigma_{i+1} - sigma_i) * grad_rescale_factor`.

Default semantics:
- `flowbp_sparse_num_active_steps = 3`
- `flowbp_sparse_late_bias = 2.0`   (w_i ∝ (i+1)^bias, idx 0..T-1)
- `flowbp_sparse_grad_rescale = 0.0` (raw per-step Δσ_k gradient)
  Same shape as `flowbp_lagrange_grad_rescale`:
      factor = 1 + grad_rescale * (Σ|Δσ_all| / Σ|Δσ_active| - 1)
  so `gr ∈ {0.2, 0.4, 0.6}` boost active gradient toward
  ReFL/full-Euler magnitude without going past it.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from flowbp.trainers.common.rollout_window import resolve_train_step_window
from flowbp.trainers.flux1.leapalign import (
    LeapAlignFluxTrainer,
    run_training,
    unpack_latents,
)
from flowbp.trainers.flux1.flowbp_lagrange import (
    _decode_packed_latents_for_wandb,
    _model_forward,
    _rollout_with_cache,
)


def _sample_late_biased_indices(
    num_steps: int,
    num_active: int,
    bias: float,
    generator: torch.Generator | None,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> list[int]:
    """Sample ``num_active`` distinct indices from ``[start_idx, end_idx)``.

    Probability is proportional to ``(idx + 1) ** bias``, so larger idx
    (closer to clean image, smaller sigma) is favoured. ``bias = 0``
    falls back to uniform; large bias concentrates on the tail.
    """
    end_idx = int(num_steps if end_idx is None else end_idx)
    start_idx = int(start_idx)
    if not 0 <= start_idx < end_idx <= num_steps:
        raise ValueError(
            f"Invalid sampling window [{start_idx}, {end_idx}) for "
            f"num_steps={num_steps}"
        )

    window_len = end_idx - start_idx
    num_active = max(1, min(int(num_active), window_len))
    if num_active >= window_len:
        return list(range(start_idx, end_idx))
    weights = torch.arange(
        start_idx + 1,
        end_idx + 1,
        dtype=torch.float32,
    ) ** float(bias)
    weights = weights / weights.sum()
    selected = torch.multinomial(
        weights,
        num_samples=int(num_active),
        replacement=False,
        generator=generator,
    )
    return sorted(start_idx + int(idx) for idx in selected.tolist())


def _scale_active_v_gradient(v: torch.Tensor, factor: float) -> torch.Tensor:
    """Forward unchanged; backward gradient through ``v`` scaled by ``factor``."""
    if factor == 1.0:
        return v
    if factor == 0.0:
        return v.detach()
    detached = v.detach()
    return detached + float(factor) * (v - detached)


def _maybe_save_connector_dump(args, prefix: str, payload: dict) -> None:
    interval = int(getattr(args, "connector_dump_interval", 0) or 0)
    step = getattr(args, "current_train_step", None)
    if interval <= 0 or step is None or int(step) % interval != 0:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    dump_dir = getattr(args, "connector_dump_dir", None)
    if not dump_dir:
        dump_dir = os.path.join(args.output_dir, "connector_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    saved = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            saved[key] = value.detach().cpu()
        else:
            saved[key] = value
    torch.save(saved, os.path.join(dump_dir, f"{prefix}_step_{int(step):06d}.pt"))


def _maybe_log_connector_wandb(args, vae, prefix: str, payload: dict | None) -> None:
    interval = int(getattr(args, "connector_wandb_interval", 0) or 0)
    step = getattr(args, "current_train_step", None)
    if interval <= 0 or step is None or int(step) % interval != 0:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    if payload is None or wandb.run is None:
        return

    max_samples = int(getattr(args, "connector_wandb_num_samples", 2) or 2)
    log_payload = {}
    for name in ("x0_pred", "x0_true"):
        images = _decode_packed_latents_for_wandb(
            args,
            vae,
            payload.get(name),
            max_samples,
        )
        if images:
            log_payload[f"connector/{prefix}_{name}"] = [
                wandb.Image(image, caption=f"{prefix} {name} sample {idx}")
                for idx, image in enumerate(images)
            ]
    if log_payload:
        wandb.log(log_payload, step=int(step))


def _compose_x0_from_velocities(
    x_init: torch.Tensor,
    sigmas: torch.Tensor,
    v_history: list[torch.Tensor],
    active_velocities: dict[int, torch.Tensor],
    grad_rescale_factor: float,
) -> torch.Tensor:
    """Full Euler integration of the cached trajectory.

    For active indices the freshly-forwarded velocities (with rescaled
    gradient) are used; for the rest the cached detached velocities are
    used. Returns fp32 — the connector stays in fp32 to match
    LeapAlign's precision path; autocast at the transformer/VAE
    boundary handles down-casting.
    """
    device = x_init.device
    delta = torch.zeros_like(x_init, dtype=torch.float32)
    for idx, cached_v in enumerate(v_history):
        sigma_i = sigmas[idx].to(device=device, dtype=torch.float32)
        sigma_next = sigmas[idx + 1].to(device=device, dtype=torch.float32)
        weight = sigma_next - sigma_i
        if idx in active_velocities:
            v = _scale_active_v_gradient(
                active_velocities[idx], grad_rescale_factor,
            )
        else:
            v = cached_v.detach()
        delta = delta + weight * v.float()
    return x_init.float() + delta


def sample_flowbp_sparse_trajectory(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    generator,
):
    """Roll out + compose; return the connector x_0_conn carrying gradients."""
    cache = _rollout_with_cache(
        args,
        device,
        transformer,
        fm_scheduler,
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
    )
    timesteps = cache["timesteps"]
    total_steps = timesteps.size(0)
    assert len(cache["v_history"]) == total_steps
    assert len(cache["sigmas"]) >= total_steps + 1

    train_start_idx, train_end_idx = resolve_train_step_window(args, total_steps)
    train_window_len = train_end_idx - train_start_idx
    num_active = int(getattr(args, "flowbp_sparse_num_active_steps", 3))
    if num_active < 1:
        raise ValueError(
            f"flowbp_sparse_num_active_steps must be >=1, got {num_active}"
        )
    num_active = min(num_active, train_window_len)
    late_bias = float(getattr(args, "flowbp_sparse_late_bias", 2.0))
    grad_rescale = float(getattr(args, "flowbp_sparse_grad_rescale", 0.0))
    # gr=0  → no rescale (raw per-step Δσ_k gradient).
    # gr=1  → full Euler-equivalent (active velocity coefficient = Σ|Δσ_all|).
    # gr>1  → push beyond full Euler magnitude (each active gets MORE than its
    #         "fair share" of the trajectory weight). Useful when even gr=1 is
    #         insufficient (e.g. with strong late-bias sampling where Δσ_k itself
    #         is tiny).
    if grad_rescale < 0.0:
        raise ValueError(
            f"flowbp_sparse_grad_rescale must be >= 0, got {grad_rescale}"
        )

    selected_indices = _sample_late_biased_indices(
        num_steps=total_steps,
        num_active=num_active,
        bias=late_bias,
        generator=generator,
        start_idx=train_start_idx,
        end_idx=train_end_idx,
    )

    sigmas_seg = cache["sigmas"][: total_steps + 1].to(
        device=device, dtype=torch.float32,
    )
    deltas_abs = (sigmas_seg[1:] - sigmas_seg[:-1]).abs()
    total_abs = deltas_abs[train_start_idx:train_end_idx].sum()
    active_abs = sum(deltas_abs[k] for k in selected_indices)

    grad_rescale_factor = 1.0
    if grad_rescale > 0.0 and float(active_abs) > 1e-8:
        full_factor = float((total_abs / active_abs).item())
        grad_rescale_factor = 1.0 + grad_rescale * (full_factor - 1.0)

    # Re-forward each selected step with grad on transformer params.
    active_velocities: dict[int, torch.Tensor] = {}
    for k in selected_indices:
        x_k = cache["x_history"][k].detach()
        v_k = _model_forward(
            args,
            transformer,
            x_k,
            timesteps[k],
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            cache["image_ids"],
            cache["cfg_guidance"],
        )
        active_velocities[int(k)] = v_k

    x_0_hat = _compose_x0_from_velocities(
        x_init=cache["x_history"][0],
        sigmas=cache["sigmas"],
        v_history=cache["v_history"],
        active_velocities=active_velocities,
        grad_rescale_factor=grad_rescale_factor,
    )

    x_0_true = cache["x0"].detach()
    # Straight-through: forward value = true cached x_0; backward flows
    # through x_0_hat into the active velocities only.
    x_0_conn = x_0_hat + (x_0_true - x_0_hat).detach()

    err_compose_per_sample = torch.abs(x_0_hat.detach() - x_0_true).mean(
        dim=tuple(range(1, x_0_hat.ndim)),
        keepdim=False,
    )
    err_compose = err_compose_per_sample.mean()

    sample_weights = torch.ones(
        x_0_conn.shape[0], device=device, dtype=torch.float32,
    )

    selected_idx_tensor = torch.tensor(
        selected_indices, device=device, dtype=torch.float32,
    )
    traj_metrics = {
        "compose_err": err_compose.detach().float(),
        "active_count": torch.tensor(
            float(num_active), device=device, dtype=torch.float32,
        ),
        "active_min_idx": selected_idx_tensor.min(),
        "active_max_idx": selected_idx_tensor.max(),
        "active_mean_idx": selected_idx_tensor.mean(),
        "grad_rescale_factor": torch.tensor(
            float(grad_rescale_factor), device=device, dtype=torch.float32,
        ),
        "active_abs_weight_sum": active_abs.detach().float(),
        "total_abs_weight_sum": total_abs.detach().float(),
        "flowbp_sparse_grad_rescale": torch.tensor(
            float(grad_rescale), device=device, dtype=torch.float32,
        ),
        "flowbp_sparse_late_bias": torch.tensor(
            float(late_bias), device=device, dtype=torch.float32,
        ),
        "train_step_start_idx": torch.tensor(
            float(train_start_idx), device=device, dtype=torch.float32,
        ),
        "train_step_end_idx": torch.tensor(
            float(train_end_idx), device=device, dtype=torch.float32,
        ),
        # Lagrange-style aliases so existing dashboards still work.
        "d0": err_compose.detach().float(),
        "dj": err_compose.detach().float(),
        "w_sim": torch.tensor(1.0, device=device, dtype=torch.float32),
    }

    connector_payload = {
        "step": getattr(args, "current_train_step", None),
        "selected_indices": torch.tensor(selected_indices),
        "x0_pred": x_0_hat.detach(),
        "x0_true": x_0_true.detach(),
        "compose_err_per_sample": err_compose_per_sample.detach(),
        "grad_rescale_factor": float(grad_rescale_factor),
        "flowbp_sparse_grad_rescale": float(grad_rescale),
        "flowbp_sparse_late_bias": float(late_bias),
        "train_step_start_idx": int(train_start_idx),
        "train_step_end_idx": int(train_end_idx),
        "active_count": int(num_active),
        "connector_scheme": "flowbp_sparse_euler",
    }
    _maybe_save_connector_dump(args, "flowbp_sparse", connector_payload)
    args._last_connector_payload = connector_payload

    return x_0_conn, sample_weights, traj_metrics


def get_flowbp_sparse_reward_loss(
    args,
    latents_pred_x0,
    sample_weights,
    vae,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latents_pred_x0 = unpack_latents(latents_pred_x0, args.h, args.w, 8)
        latents_pred_x0 = (latents_pred_x0 / 0.3611) + 0.1159
        image_pred_x0 = vae.decode(latents_pred_x0, return_dict=False)[0]
        image_pred_x0 = (image_pred_x0 * 0.5 + 0.5).clamp(0, 1)

    if args.use_hpsv2:
        image_pred_x0 = preprocess_val(image_pred_x0)
        text = tokenizer(caption).to(
            device=image_pred_x0.device, non_blocking=True,
        )
        with torch.amp.autocast("cuda"):
            outputs = reward_model(image_pred_x0, text)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            reward_pred_x0 = torch.einsum(
                "bc,bc->b", image_features, text_features,
            )
    else:
        raise NotImplementedError(
            "Only HPSv2 is currently supported for FlowBP-Sparse."
        )

    raw_loss = F.relu(-reward_pred_x0 + args.loss_relu_clip) * args.loss_grad_scale
    loss = torch.mean(
        sample_weights.to(raw_loss.device, raw_loss.dtype) * raw_loss
    )
    return loss, reward_pred_x0.detach().mean()


def train_flowbp_sparse_one_step(
    args,
    inner_step,
    device,
    transformer,
    vae,
    fm_scheduler,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    preprocess_val,
    select_idx_generator,
):
    (
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        caption,
    ) = next(loader)
    text_ids = torch.zeros(
        encoder_hidden_states.shape[1], 3,
    ).to(device=device, dtype=text_ids.dtype)

    latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_sparse_trajectory(
        args,
        device,
        transformer,
        fm_scheduler,
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        select_idx_generator,
    )
    _maybe_log_connector_wandb(
        args,
        vae,
        "flowbp_sparse",
        getattr(args, "_last_connector_payload", None),
    )
    loss, avg_reward_pred_x0 = get_flowbp_sparse_reward_loss(
        args,
        latents_pred_x0,
        sample_weights,
        vae,
        reward_model,
        tokenizer,
        caption,
        preprocess_val,
    )

    (loss / args.gradient_accumulation_steps).backward()
    avg_loss = loss.detach()
    dist.all_reduce(
        avg_loss.div_(args.gradient_accumulation_steps),
        op=dist.ReduceOp.AVG,
    )
    dist.all_reduce(
        avg_reward_pred_x0.div_(args.gradient_accumulation_steps),
        op=dist.ReduceOp.AVG,
    )
    reduced_traj_metrics = {}
    for metric_name, metric_value in traj_metrics.items():
        if metric_value is None:
            reduced_traj_metrics[metric_name] = None
            continue
        metric_tensor = metric_value.detach().float().to(device)
        dist.all_reduce(
            metric_tensor.div_(args.gradient_accumulation_steps),
            op=dist.ReduceOp.AVG,
        )
        reduced_traj_metrics[metric_name] = metric_tensor.item()

    if (inner_step + 1) % args.gradient_accumulation_steps == 0:
        grad_norm = transformer.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        grad_norm = grad_norm.item()
    else:
        grad_norm = None
    return (
        avg_loss.item(),
        avg_reward_pred_x0.item(),
        grad_norm,
        reduced_traj_metrics,
    )


class FlowBPSparseFluxTrainer(LeapAlignFluxTrainer):
    """FlowBP-Sparse trainer wrapper.

    Single rollout per training step; K active steps re-forwarded with
    grad; full Euler composition gives x_0_hat ≈ true x_0 in forward,
    while backward flows only through the K active velocities, each
    rescaled by ``flowbp_sparse_grad_rescale``.
    """

    def train(self):
        self.args.trainer = "flowbp_sparse"
        return run_training(self.args)

    def train_one_step(
        self,
        inner_step,
        device,
        transformer,
        vae,
        fm_scheduler,
        reward_model,
        tokenizer,
        optimizer,
        lr_scheduler,
        loader,
        preprocess_val,
        select_idx_generator,
    ):
        return train_flowbp_sparse_one_step(
            self.args,
            inner_step,
            device,
            transformer,
            vae,
            fm_scheduler,
            reward_model,
            tokenizer,
            optimizer,
            lr_scheduler,
            loader,
            preprocess_val,
            select_idx_generator,
        )
