
"""FlowBP-Bridge trainer for FLUX.1.

The connector keeps the same unbiased forward path as FlowBP-Sparse:
roll out once, cache the discrete sigma schedule and every velocity, then
reconstruct intermediate/end latents by exact Euler composition over the cached
trajectory. A randomly sampled ``j`` splits the rollout into ``0 -> j`` and
``j -> x_0``. Active velocities are sampled from both halves whenever possible,
so the final reward gradient sees one nested layer through ``x_j`` while still
allowing arbitrary cached steps to be fine-tuned like FlowBP-Sparse.

The post-segment sampler always activates the anchor ``j`` itself (and draws
the remaining post-active indices from ``[j+1, total_steps)`` with the same
late-biased weighting), so the nested gradient path through ``xj_model_input``
fires on every training step instead of being a coin flip.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from flowbp.trainers.common.rollout_window import resolve_train_step_window
from flowbp.trainers.flux1.flowbp_sparse import (
    _sample_late_biased_indices,
    _scale_active_v_gradient,
    get_flowbp_sparse_reward_loss,
)
from flowbp.trainers.flux1.flowbp_lagrange import (
    _maybe_log_connector_wandb,
    _maybe_save_connector_dump,
    _model_forward,
    _rollout_with_cache,
)
from flowbp.trainers.common.jk_sampling import sample_jk_indices
from flowbp.trainers.flux1.leapalign import (
    LeapAlignFluxTrainer,
    run_training,
)


def _compose_interval_from_velocities(
    x_start: torch.Tensor,
    sigmas: torch.Tensor,
    v_history: list[torch.Tensor],
    start_idx: int,
    target_idx: int,
    active_velocities: dict[int, torch.Tensor],
    grad_rescale_factor: float,
) -> torch.Tensor:
    """Exact Euler composition over ``[start_idx, target_idx)``."""
    if target_idx <= start_idx:
        raise ValueError(
            f"target_idx must be > start_idx, got {start_idx=} {target_idx=}"
        )
    if len(sigmas) < len(v_history) + 1:
        raise ValueError(
            f"sigmas length must be at least len(v_history)+1, got "
            f"{len(sigmas)=}, {len(v_history)=}"
        )

    device = x_start.device
    delta = torch.zeros_like(x_start, dtype=torch.float32)
    for idx in range(start_idx, target_idx):
        sigma_i = sigmas[idx].to(device=device, dtype=torch.float32)
        sigma_next = sigmas[idx + 1].to(device=device, dtype=torch.float32)
        weight = sigma_next - sigma_i
        if idx in active_velocities:
            velocity = _scale_active_v_gradient(
                active_velocities[idx],
                grad_rescale_factor,
            )
        else:
            velocity = v_history[idx].detach()
        delta = delta + weight * velocity.float()
    return x_start.float() + delta


def _allocate_active_counts(
    total_active: int,
    pre_len: int,
    post_len: int,
) -> tuple[int, int]:
    """Split active budget across both halves, keeping both non-empty."""
    if pre_len <= 0 or post_len <= 0:
        raise ValueError(f"Both segments must be non-empty, got {pre_len=} {post_len=}")

    total_active = min(int(total_active), pre_len + post_len)
    if total_active < 2:
        raise ValueError(
            "FlowBP-Bridge needs at least two active velocities so both sides of "
            f"j can receive gradients, got {total_active}"
        )

    pre_count = int(round(total_active * pre_len / float(pre_len + post_len)))
    pre_count = min(max(pre_count, 1), pre_len)
    post_count = total_active - pre_count

    if post_count < 1:
        post_count = 1
        pre_count = total_active - post_count
    if post_count > post_len:
        post_count = post_len
        pre_count = total_active - post_count

    pre_count = min(max(pre_count, 1), pre_len)
    post_count = min(max(post_count, 1), post_len)

    missing = total_active - pre_count - post_count
    if missing > 0:
        add_pre = min(missing, pre_len - pre_count)
        pre_count += add_pre
        missing -= add_pre
    if missing > 0:
        post_count += min(missing, post_len - post_count)

    return pre_count, post_count


def _sample_segment_indices(
    start_idx: int,
    length: int,
    num_active: int,
    late_bias: float,
    generator: torch.Generator | None,
) -> list[int]:
    offsets = _sample_late_biased_indices(
        num_steps=length,
        num_active=num_active,
        bias=late_bias,
        generator=generator,
    )
    return [int(start_idx + offset) for offset in offsets]


def _sample_post_indices_force_j(
    j_idx: int,
    post_len: int,
    num_active: int,
    late_bias: float,
    generator: torch.Generator | None,
) -> list[int]:
    """Sample post-segment active steps while always activating the anchor j.

    Mirrors the FlowBP-Bridge post-index sampler in the SD3.5 trainer: the first
    slot is reserved for ``j_idx`` so the nested gradient path through
    ``xj_model_input`` is taken on every step; the remaining ``num_active - 1``
    slots are drawn from ``[j_idx + 1, j_idx + post_len)`` with the same
    late-biased weighting used elsewhere.
    """
    num_active = max(1, min(int(num_active), int(post_len)))
    post_active_indices = [int(j_idx)]
    remaining = num_active - 1
    if remaining > 0 and post_len > 1:
        post_active_indices.extend(
            _sample_segment_indices(
                start_idx=int(j_idx) + 1,
                length=int(post_len) - 1,
                num_active=remaining,
                late_bias=late_bias,
                generator=generator,
            )
        )
    return sorted(post_active_indices)


def _build_active_velocities(
    args,
    transformer,
    cache,
    timesteps: torch.Tensor,
    active_indices: list[int],
    j_idx: int,
    xj_model_input: torch.Tensor | None,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
) -> dict[int, torch.Tensor]:
    active_velocities: dict[int, torch.Tensor] = {}
    for idx in active_indices:
        if idx == j_idx and xj_model_input is not None:
            x_i = xj_model_input
        else:
            x_i = cache["x_history"][idx].detach()
        active_velocities[idx] = _model_forward(
            args,
            transformer,
            x_i,
            timesteps[idx],
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            cache["image_ids"],
            cache["cfg_guidance"],
        )
    return active_velocities


def sample_flowbp_bridge_trajectory(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    generator,
):
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

    select_indices, _k_idx, j_idx = sample_jk_indices(args, total_steps, generator)
    train_start_idx, train_end_idx = resolve_train_step_window(
        args,
        total_steps,
        min_window=3,
    )
    pre_len = int(j_idx) - train_start_idx
    post_len = int(total_steps - j_idx)

    num_active_total = int(getattr(args, "flowbp_sparse_num_active_steps", 4))
    pre_active_count, post_active_count = _allocate_active_counts(
        num_active_total,
        pre_len,
        post_len,
    )

    late_bias = float(getattr(args, "flowbp_sparse_late_bias", 2.0))
    grad_rescale = float(getattr(args, "flowbp_sparse_grad_rescale", 0.0))
    if grad_rescale < 0.0:
        raise ValueError(
            f"flowbp_sparse_grad_rescale must be >= 0, got {grad_rescale}"
        )

    pre_active_indices = _sample_segment_indices(
        start_idx=train_start_idx,
        length=pre_len,
        num_active=pre_active_count,
        late_bias=late_bias,
        generator=generator,
    )
    post_active_indices = _sample_post_indices_force_j(
        j_idx=j_idx,
        post_len=post_len,
        num_active=post_active_count,
        late_bias=late_bias,
        generator=generator,
    )
    active_indices = pre_active_indices + post_active_indices

    sigmas_full = cache["sigmas"][: total_steps + 1].to(
        device=device,
        dtype=torch.float32,
    )
    deltas_abs = (sigmas_full[1:] - sigmas_full[:-1]).abs()
    total_abs = deltas_abs[train_start_idx:train_end_idx].sum()
    active_abs = sum(deltas_abs[idx] for idx in active_indices)

    grad_rescale_factor = 1.0
    if grad_rescale > 0.0 and float(active_abs) > 1e-8:
        full_factor = float((total_abs / active_abs).item())
        grad_rescale_factor = 1.0 + grad_rescale * (full_factor - 1.0)

    pre_active_velocities = _build_active_velocities(
        args=args,
        transformer=transformer,
        cache=cache,
        timesteps=timesteps,
        active_indices=pre_active_indices,
        j_idx=j_idx,
        xj_model_input=None,
        encoder_hidden_states=encoder_hidden_states,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
    )

    x_j = cache["x_history"][j_idx]
    xj_hat = _compose_interval_from_velocities(
        x_start=cache["x_history"][0],
        sigmas=cache["sigmas"],
        v_history=cache["v_history"],
        start_idx=0,
        target_idx=j_idx,
        active_velocities=pre_active_velocities,
        grad_rescale_factor=grad_rescale_factor,
    )
    xj_conn = xj_hat + (x_j - xj_hat).detach()

    alpha = float(args.alpha)
    xj_model_input = float(alpha) * xj_conn + (1.0 - float(alpha)) * xj_conn.detach()

    post_active_velocities = _build_active_velocities(
        args=args,
        transformer=transformer,
        cache=cache,
        timesteps=timesteps,
        active_indices=post_active_indices,
        j_idx=j_idx,
        xj_model_input=xj_model_input,
        encoder_hidden_states=encoder_hidden_states,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
    )

    x0_hat = _compose_interval_from_velocities(
        x_start=xj_conn,
        sigmas=cache["sigmas"],
        v_history=cache["v_history"],
        start_idx=j_idx,
        target_idx=total_steps,
        active_velocities=post_active_velocities,
        grad_rescale_factor=grad_rescale_factor,
    )
    x0_true = cache["x0"].detach()
    x0_conn = x0_hat + (x0_true - x0_hat).detach()

    d_j = torch.abs(x_j.double() - xj_hat.double()).mean(
        dim=tuple(range(1, x_j.ndim)),
        keepdim=False,
    )
    d_0 = torch.abs(x0_true.double() - x0_hat.double()).mean(
        dim=tuple(range(1, x0_true.ndim)),
        keepdim=False,
    )

    sample_weights = torch.ones(
        x0_conn.shape[0],
        device=device,
        dtype=torch.float32,
    )
    pre_idx_tensor = torch.tensor(pre_active_indices, device=device, dtype=torch.float32)
    post_idx_tensor = torch.tensor(
        post_active_indices,
        device=device,
        dtype=torch.float32,
    )
    active_idx_tensor = torch.tensor(active_indices, device=device, dtype=torch.float32)

    traj_metrics = {
        "d0": d_0.detach().float().mean(),
        "dj": d_j.detach().float().mean(),
        "w_sim": torch.tensor(1.0, device=device, dtype=torch.float32),
        "active_count": torch.tensor(
            float(len(active_indices)), device=device, dtype=torch.float32,
        ),
        "pre_j_active_count": torch.tensor(
            float(len(pre_active_indices)), device=device, dtype=torch.float32,
        ),
        "post_j_active_count": torch.tensor(
            float(len(post_active_indices)), device=device, dtype=torch.float32,
        ),
        "active_min_idx": active_idx_tensor.min(),
        "active_max_idx": active_idx_tensor.max(),
        "active_mean_idx": active_idx_tensor.mean(),
        "pre_active_mean_idx": pre_idx_tensor.mean(),
        "post_active_mean_idx": post_idx_tensor.mean(),
        "j_active": torch.tensor(
            1.0 if int(j_idx) in post_active_indices else 0.0,
            device=device,
            dtype=torch.float32,
        ),
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
        "alpha": torch.tensor(
            float(alpha), device=device, dtype=torch.float32,
        ),
        "train_step_start_idx": torch.tensor(
            float(train_start_idx), device=device, dtype=torch.float32,
        ),
        "train_step_end_idx": torch.tensor(
            float(train_end_idx), device=device, dtype=torch.float32,
        ),
        "j_idx": torch.tensor(float(j_idx), device=device, dtype=torch.float32),
    }

    connector_payload = {
        "step": getattr(args, "current_train_step", None),
        "select_indices": select_indices.detach(),
        "j_idx": int(j_idx),
        "pre_active_indices": torch.tensor(pre_active_indices),
        "post_active_indices": torch.tensor(post_active_indices),
        "active_indices": torch.tensor(active_indices),
        "xj_pred": xj_hat.detach(),
        "xj": x_j.detach(),
        "x0_pred": x0_hat.detach(),
        "x0": x0_true.detach(),
        "dj_per_sample": d_j.detach(),
        "d0_per_sample": d_0.detach(),
        "grad_rescale_factor": float(grad_rescale_factor),
        "flowbp_sparse_grad_rescale": float(grad_rescale),
        "flowbp_sparse_late_bias": float(late_bias),
        "alpha": float(alpha),
        "train_step_start_idx": int(train_start_idx),
        "train_step_end_idx": int(train_end_idx),
        "active_count": int(len(active_indices)),
        "pre_j_active_count": int(len(pre_active_indices)),
        "post_j_active_count": int(len(post_active_indices)),
        "connector_scheme": "flowbp_bridge_euler",
    }
    _maybe_save_connector_dump(args, "flowbp_bridge", connector_payload)
    args._last_connector_payload = connector_payload

    return x0_conn, sample_weights, traj_metrics


def train_flowbp_bridge_one_step(
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
        encoder_hidden_states.shape[1],
        3,
    ).to(device=device, dtype=text_ids.dtype)

    latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_bridge_trajectory(
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
        "flowbp_bridge",
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


class FlowBPBridgeFluxTrainer(LeapAlignFluxTrainer):
    """FLUX.1 trainer wrapper for FlowBP-Bridge."""

    def train(self):
        self.args.trainer = "flowbp_bridge"
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
        return train_flowbp_bridge_one_step(
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
