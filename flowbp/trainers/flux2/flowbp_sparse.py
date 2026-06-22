
"""FlowBP-Sparse trainer for FLUX.2.

Mirrors :mod:`flowbp.trainers.flux1.flowbp_sparse` but uses the FLUX.2
forward / latent packing pipeline (Qwen3 text embeds + 4D position ids,
patchify+BN VAE normalization, ``_cfg_aware_forward`` for undistilled
checkpoints). The connector math — late-biased step sampling, full Euler
composition, straight-through residual onto the cached ``x_0``, and the
``flowbp_sparse_grad_rescale`` gradient compensation are reused from the
FLUX.1 implementation so behavior stays in lock-step across the two backbones.

Algorithm recap:

1. Roll out the full N-step FLUX.2 trajectory once with ``no_grad`` and cache
   ``(x_history, v_history)``.
2. Sample ``flowbp_sparse_num_active_steps`` distinct timesteps from a power-law
   distribution ``w_i ∝ (i+1)^flowbp_sparse_late_bias`` (bigger idx → smaller
   sigma → closer to clean image gets more weight).
3. Re-forward the transformer at those K cached ``x_k``'s WITH gradient
   enabled, replacing the cached detached velocity.
4. Compose ``x_0_hat`` in fp32 via full Euler integration; cached steps use
   the detached cached velocity, active steps use the new differentiable one
   (scaled by ``flowbp_sparse_grad_rescale``).
5. Straight-through: forward value = true cached ``x_0``; backward flows
   only through the K active velocities.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from flowbp.trainers.common.rollout_window import resolve_train_step_window
from flowbp.trainers.flux1.flowbp_sparse import (
    _compose_x0_from_velocities,
    _sample_late_biased_indices,
)
from flowbp.trainers.flux2.flowbp_lagrange import (
    _model_forward,
    _rollout_with_cache,
)
from flowbp.trainers.flux2.leapalign import (
    LeapAlignFlux2Trainer,
    _decode_packed_latents_for_wandb,
    _maybe_save_connector_dump,
    decode_packed_latents,
    encode_or_load_prompts,
    run_training,
)


def _maybe_log_connector_wandb(args, vae, prefix: str, payload: dict | None) -> None:
    """FLUX.2 image-logging hook tailored to FlowBP-Sparse payload keys.

    FlowBP-Sparse only produces ``x0_pred`` / ``x0_true`` (there is
    no intermediate ``x_j`` like LeapAlign / FlowBP-Lagrange), so we only decode
    those two tensors.
    but routed through the FLUX.2 VAE de-BN + unpatchify path in
    ``_decode_packed_latents_for_wandb``.
    """
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
            args, vae, payload.get(name), max_samples,
        )
        if images:
            log_payload[f"connector/{prefix}_{name}"] = [
                wandb.Image(image, caption=f"{prefix} {name} sample {idx}")
                for idx, image in enumerate(images)
            ]
    if log_payload:
        wandb.log(log_payload, step=int(step))


def sample_flowbp_sparse_trajectory(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    text_ids,
    generator,
):
    """Roll out + compose; return the connector ``x_0_conn`` carrying gradients."""
    cache = _rollout_with_cache(
        args,
        device,
        transformer,
        fm_scheduler,
        encoder_hidden_states,
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
    # gr=0  -> no rescale (raw per-step delta-sigma_k gradient).
    # gr=1  -> full Euler-equivalent (active velocity coefficient = sum |delta-sigma_all|).
    # gr>1  -> push beyond full Euler magnitude.
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
    # ``_model_forward`` here is the FLUX.2 version (no pooled_projections,
    # routed through ``_cfg_aware_forward``).
    active_velocities: dict[int, torch.Tensor] = {}
    for k in selected_indices:
        x_k = cache["x_history"][k].detach()
        v_k = _model_forward(
            args,
            transformer,
            x_k,
            timesteps[k],
            encoder_hidden_states,
            text_ids,
            cache["image_ids"],
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
        # Keep Lagrange-style metric aliases so existing dashboards work.
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
        "connector_scheme": "flowbp_sparse_euler_flux2",
    }
    _maybe_save_connector_dump(args, "flowbp_sparse_flux2", connector_payload)
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
    """FLUX.2 reward loss: packed -> unpack -> de-BN -> unpatchify -> VAE -> HPSv2."""
    image_pred_x0 = decode_packed_latents(args, vae, latents_pred_x0)

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
            "Only HPSv2 is currently supported for FlowBP-Sparse FLUX.2."
        )

    raw_loss = F.relu(-reward_pred_x0 + args.loss_relu_clip) * args.loss_grad_scale
    loss = torch.mean(
        sample_weights.to(raw_loss.device, raw_loss.dtype) * raw_loss
    )
    return loss, reward_pred_x0.detach().mean()


def train_flowbp_sparse_one_step_flux2(
    args,
    inner_step,
    device,
    transformer,
    vae,
    fm_scheduler,
    text_encoder,
    text_tokenizer,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    preprocess_val,
    select_idx_generator,
):
    batch = next(loader)
    prompt_embeds, text_ids, caption = encode_or_load_prompts(
        args, batch, device, text_encoder, text_tokenizer,
    )

    latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_sparse_trajectory(
        args,
        device,
        transformer,
        fm_scheduler,
        prompt_embeds,
        text_ids,
        select_idx_generator,
    )
    _maybe_log_connector_wandb(
        args, vae, "flowbp_sparse_flux2",
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

    # CFG backward compensation: see the matching block in
    # ``flowbp_lagrange.train_flowbp_lagrange_one_step_flux2`` for the
    # rationale. ``_cfg_aware_forward`` detaches the negative branch, so the
    # effective gradient through theta gets multiplied by ``cfg_guidance``;
    # dividing the backward loss by ``cfg_guidance`` recovers the non-CFG
    # gradient magnitude and keeps ``grad_norm`` / ``max_grad_norm`` thresholds
    # comparable across CFG settings. Toggle off via
    # ``--no-cfg_grad_norm_compensate``.
    avg_loss = loss.detach()
    backward_factor = 1.0 / args.gradient_accumulation_steps
    cfg_scale = float(getattr(args, "cfg_guidance", 1.0))
    cfg_compensate = bool(getattr(args, "cfg_grad_norm_compensate", True))
    if cfg_scale > 1.0 and cfg_compensate:
        backward_factor = backward_factor / cfg_scale
    (loss * backward_factor).backward()

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


class FlowBPSparseFlux2Trainer(LeapAlignFlux2Trainer):
    """FlowBP-Sparse trainer wrapper for FLUX.2.

    Single rollout per training step; K active steps re-forwarded with
    grad; full Euler composition gives ``x_0_hat`` matching the true cached
    ``x_0`` in forward, while backward flows only through the K active
    velocities, each rescaled by ``flowbp_sparse_grad_rescale``.
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
        text_encoder,
        text_tokenizer,
        reward_model,
        tokenizer,
        optimizer,
        lr_scheduler,
        loader,
        preprocess_val,
        select_idx_generator,
    ):
        return train_flowbp_sparse_one_step_flux2(
            self.args,
            inner_step,
            device,
            transformer,
            vae,
            fm_scheduler,
            text_encoder,
            text_tokenizer,
            reward_model,
            tokenizer,
            optimizer,
            lr_scheduler,
            loader,
            preprocess_val,
            select_idx_generator,
        )
