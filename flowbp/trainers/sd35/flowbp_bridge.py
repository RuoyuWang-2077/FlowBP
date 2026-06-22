"""FlowBP-Bridge trainer for SD3.5.

SD3.5 port of FlowBP-Bridge: pick an intermediate
anchor ``j`` via the shared jk sampler, allocate a late-biased active-step
budget to the ``[0, j)`` and ``[j, T)`` segments, and compose each interval
with full Euler integration so reward gradients reach both sides of ``j``.
"""

from __future__ import annotations

import torch

from flowbp.trainers.common.rollout_window import resolve_train_step_window
from flowbp.trainers.sd35.flowbp_sparse import _sample_late_biased_indices
from flowbp.trainers.sd35.flowbp_lagrange import _rollout_with_cache_sd35
from flowbp.trainers.common.jk_sampling import sample_jk_indices
from flowbp.trainers.sd35.leapalign import (
    LeapAlignSD35Trainer,
    _build_traj_filter_mask,
    _maybe_save_connector_dump,
    _model_forward_sd35,
    run_training_sd35,
)


def _compose_interval_from_velocities_sd35(
    x_start: torch.Tensor,
    sigmas: torch.Tensor,
    v_history: list[torch.Tensor],
    start_idx: int,
    target_idx: int,
    active_velocities: dict[int, torch.Tensor],
    grad_rescale_factor: float,
) -> torch.Tensor:
    """Exact Euler composition over one trajectory interval."""
    if target_idx <= start_idx:
        raise ValueError(
            f"target_idx must be > start_idx, got start_idx={start_idx}, target_idx={target_idx}"
        )
    if len(sigmas) < len(v_history) + 1:
        raise ValueError(
            f"sigmas length must be at least len(v_history)+1, got "
            f"len(sigmas)={len(sigmas)}, len(v_history)={len(v_history)}"
        )

    x = x_start.float()
    for idx in range(start_idx, target_idx):
        sigma_i = sigmas[idx].to(device=x.device, dtype=torch.float32)
        sigma_next = sigmas[idx + 1].to(device=x.device, dtype=torch.float32)
        delta = sigma_next - sigma_i
        if idx in active_velocities:
            v = active_velocities[idx].float()
            v = v.detach() + grad_rescale_factor * (v - v.detach())
        else:
            v = v_history[idx].detach().float()
        x = x + delta * v
    return x


def _allocate_flowbp_bridge_active_counts(
    total_active: int,
    pre_len: int,
    post_len: int,
) -> tuple[int, int]:
    """Split active-step budget across both sides of the intermediate anchor."""
    if pre_len <= 0 or post_len <= 0:
        raise ValueError(f"Both FlowBP-Bridge segments must be non-empty, got {pre_len=} {post_len=}")

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


def _sample_flowbp_bridge_segment_indices(
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


def _sample_flowbp_bridge_post_indices(
    j_idx: int,
    post_len: int,
    num_active: int,
    late_bias: float,
    generator: torch.Generator | None,
) -> list[int]:
    """Sample post-segment active steps while always activating the anchor j."""
    num_active = max(1, min(int(num_active), int(post_len)))
    post_active_indices = [int(j_idx)]
    remaining = num_active - 1
    if remaining > 0:
        post_active_indices.extend(
            _sample_flowbp_bridge_segment_indices(
                start_idx=int(j_idx) + 1,
                length=int(post_len) - 1,
                num_active=remaining,
                late_bias=late_bias,
                generator=generator,
            )
        )
    return sorted(post_active_indices)


def _build_flowbp_bridge_active_velocities_sd35(
    args,
    transformer,
    cache,
    timesteps: torch.Tensor,
    active_indices: list[int],
    j_idx: int,
    xj_model_input: torch.Tensor | None,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
) -> dict[int, torch.Tensor]:
    active_velocities: dict[int, torch.Tensor] = {}
    for idx in active_indices:
        if idx == j_idx and xj_model_input is not None:
            x_i = xj_model_input
        else:
            x_i = cache["x_history"][idx].detach()
        active_velocities[idx] = _model_forward_sd35(
            args,
            transformer,
            x_i,
            timesteps[idx],
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
    return active_velocities


def sample_flowbp_bridge_trajectory_sd35(
    args,
    device,
    transformer,
    fm_scheduler,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    generator,
):
    """SD3.5 port of FlowBP-Bridge."""
    args._last_sample_keep_mask = None
    cache = _rollout_with_cache_sd35(
        args,
        device,
        transformer,
        fm_scheduler,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )
    timesteps = cache["timesteps"]
    total_steps = timesteps.size(0)
    assert len(cache["v_history"]) == total_steps
    assert len(cache["sigmas"]) >= total_steps + 1

    select_indices, k_idx, j_idx = sample_jk_indices(args, total_steps, generator)
    jk_truncated = bool(getattr(args, "_last_jk_truncated", False))
    train_start_idx, train_end_idx = resolve_train_step_window(
        args,
        total_steps,
        min_window=3,
    )
    pre_len = int(j_idx) - train_start_idx
    post_len = int(total_steps - j_idx)

    num_active_total = int(getattr(args, "flowbp_sparse_num_active_steps", 4))
    pre_active_count, post_active_count = _allocate_flowbp_bridge_active_counts(
        num_active_total,
        pre_len,
        post_len,
    )

    late_bias = float(getattr(args, "flowbp_sparse_late_bias", 2.0))
    grad_rescale = float(getattr(args, "flowbp_sparse_grad_rescale", 0.0))
    if grad_rescale < 0.0:
        raise ValueError(f"flowbp_sparse_grad_rescale must be >= 0, got {grad_rescale}")

    pre_active_indices = _sample_flowbp_bridge_segment_indices(
        start_idx=train_start_idx,
        length=pre_len,
        num_active=pre_active_count,
        late_bias=late_bias,
        generator=generator,
    )
    post_active_indices = _sample_flowbp_bridge_post_indices(
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

    pre_active_velocities = _build_flowbp_bridge_active_velocities_sd35(
        args=args,
        transformer=transformer,
        cache=cache,
        timesteps=timesteps,
        active_indices=pre_active_indices,
        j_idx=j_idx,
        xj_model_input=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )

    x_j = cache["x_history"][j_idx]
    xj_hat = _compose_interval_from_velocities_sd35(
        x_start=cache["x_history"][0],
        sigmas=cache["sigmas"],
        v_history=cache["v_history"],
        start_idx=0,
        target_idx=j_idx,
        active_velocities=pre_active_velocities,
        grad_rescale_factor=grad_rescale_factor,
    )
    xj_conn = xj_hat + (x_j - xj_hat).detach()
    xj_conn = xj_conn.to(torch.bfloat16)

    alpha = float(args.alpha)
    xj_model_input = float(alpha) * xj_conn + (1.0 - float(alpha)) * xj_conn.detach()

    post_active_velocities = _build_flowbp_bridge_active_velocities_sd35(
        args=args,
        transformer=transformer,
        cache=cache,
        timesteps=timesteps,
        active_indices=post_active_indices,
        j_idx=j_idx,
        xj_model_input=xj_model_input,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )

    x0_hat = _compose_interval_from_velocities_sd35(
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
    x0_conn = x0_conn.to(torch.bfloat16)

    d_j = torch.abs(x_j.double() - xj_hat.double()).mean(
        dim=tuple(range(1, x_j.ndim)),
        keepdim=False,
    )
    d_0 = torch.abs(x0_true.double() - x0_hat.double()).mean(
        dim=tuple(range(1, x0_true.ndim)),
        keepdim=False,
    )
    _, filter_metrics = _build_traj_filter_mask(args, d_j, d_0)

    sample_weights = torch.ones(
        x0_conn.shape[0],
        device=device,
        dtype=torch.float32,
    )
    pre_idx_tensor = torch.tensor(pre_active_indices, device=device, dtype=torch.float32)
    post_idx_tensor = torch.tensor(post_active_indices, device=device, dtype=torch.float32)
    active_idx_tensor = torch.tensor(active_indices, device=device, dtype=torch.float32)

    traj_metrics = {
        "d0": d_0.detach().float().mean(),
        "dj": d_j.detach().float().mean(),
        "w_sim": torch.tensor(1.0, device=device, dtype=torch.float32),
        "active_count": torch.tensor(float(len(active_indices)), device=device, dtype=torch.float32),
        "pre_j_active_count": torch.tensor(float(len(pre_active_indices)), device=device, dtype=torch.float32),
        "post_j_active_count": torch.tensor(float(len(post_active_indices)), device=device, dtype=torch.float32),
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
        "grad_rescale_factor": torch.tensor(float(grad_rescale_factor), device=device, dtype=torch.float32),
        "active_abs_weight_sum": active_abs.detach().float(),
        "total_abs_weight_sum": total_abs.detach().float(),
        "flowbp_sparse_grad_rescale": torch.tensor(float(grad_rescale), device=device, dtype=torch.float32),
        "flowbp_sparse_late_bias": torch.tensor(float(late_bias), device=device, dtype=torch.float32),
        "alpha": torch.tensor(float(alpha), device=device, dtype=torch.float32),
        "train_step_start_idx": torch.tensor(float(train_start_idx), device=device, dtype=torch.float32),
        "train_step_end_idx": torch.tensor(float(train_end_idx), device=device, dtype=torch.float32),
        "j_idx": torch.tensor(float(j_idx), device=device, dtype=torch.float32),
        "k_idx": torch.tensor(float(k_idx), device=device, dtype=torch.float32),
        "jk_gap": torch.tensor(float(j_idx - k_idx), device=device, dtype=torch.float32),
        "j_rev": torch.tensor(float(total_steps - j_idx), device=device, dtype=torch.float32),
        "k_rev": torch.tensor(float(total_steps - k_idx), device=device, dtype=torch.float32),
        "jk_truncated": torch.tensor(1.0 if jk_truncated else 0.0, device=device, dtype=torch.float32),
    }
    traj_metrics.update(filter_metrics)

    connector_payload = {
        "step": getattr(args, "current_train_step", None),
        "select_indices": select_indices.detach(),
        "k_idx": int(k_idx),
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
        "connector_scheme": "sd35_flowbp_bridge",
    }
    _maybe_save_connector_dump(args, "sd35_flowbp_bridge", connector_payload)
    args._last_connector_payload = connector_payload

    return x0_conn, sample_weights, traj_metrics


class FlowBPBridgeSD35Trainer(LeapAlignSD35Trainer):
    def train(self):
        self.args.trainer = "flowbp_bridge"
        return run_training_sd35(self.args)
