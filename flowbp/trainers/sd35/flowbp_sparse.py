"""FlowBP-Sparse trainer for SD3.5.

FlowBP-Sparse trainer for SD3.5: single no-grad rollout per
training step; K active steps re-forwarded with grad; full Euler composition
produces ``x_0_hat``; straight-through residual onto the cached ``x_0``;
backward only flows through the K active velocities, each rescaled by
``flowbp_sparse_grad_rescale``.

Owns ``_sample_late_biased_indices`` (reused by the FlowBP-Bridge trainer,
shared by FlowBP-Bridge).
"""

from __future__ import annotations

import torch

from flowbp.trainers.common.rollout_window import resolve_train_step_window
from flowbp.trainers.sd35.flowbp_lagrange import _rollout_with_cache_sd35
from flowbp.trainers.sd35.leapalign import (
    LeapAlignSD35Trainer,
    _maybe_save_connector_dump,
    _model_forward_sd35,
    run_training_sd35,
)


def _sample_late_biased_indices(
    num_steps,
    num_active,
    bias,
    generator,
    start_idx=0,
    end_idx=None,
):
    """Power-law late-biased multinomial sampling of K distinct timesteps.

    Mirrors FlowBP-Sparse ``_sample_late_biased_indices``:
    weights ``w_i = (i + 1) ** bias`` (in rollout cache, larger idx = closer
    to clean image side = larger sigma delta consumed in that step). Larger
    ``bias`` skews active steps toward the image end, which is where reward
    gradients are most informative for HPSv2-style rewards.
    """
    end_idx = int(num_steps if end_idx is None else end_idx)
    start_idx = int(start_idx)
    if not 0 <= start_idx < end_idx <= num_steps:
        raise ValueError(
            f"Invalid sampling window [{start_idx}, {end_idx}) for "
            f"num_steps={num_steps}"
        )
    window_len = end_idx - start_idx
    num_active = max(1, min(int(num_active), int(window_len)))
    if num_active >= window_len:
        return list(range(start_idx, end_idx))
    idx = torch.arange(start_idx, end_idx, dtype=torch.float32)
    weights = (idx + 1.0).pow(float(bias))
    weights = weights / weights.sum()
    chosen = torch.multinomial(
        weights,
        num_samples=num_active,
        replacement=False,
        generator=generator,
    )
    return sorted(start_idx + int(k) for k in chosen.tolist())


def _compose_x0_from_velocities(
    x_init,
    sigmas,
    v_history,
    active_velocities,
    grad_rescale_factor,
):
    """Full Euler integration giving x_0_hat.

    Cached steps use the detached cached velocity; active steps use the new
    differentiable velocity, with its grad-flowing component scaled by
    ``grad_rescale_factor``. Forward value is unchanged by the rescale
    (``v.detach() + g * (v - v.detach())`` evaluates to ``v``); only the
    backward gradient scales by ``g``. Mirrors the shared FLUX implementation.
    """
    total_steps = len(v_history)
    x = x_init.float()
    for idx in range(total_steps):
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


def sample_flowbp_sparse_trajectory_sd35(
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
    """FlowBP-Sparse sampler for SD3.5.

    FlowBP-Sparse SD3.5 trajectory sampler:
    one no-grad rollout caches (x_history, v_history); K timesteps drawn from
    power-law late-biased distribution; those K active steps are re-forwarded
    WITH gradient; x_0_hat is composed via full Euler integration in fp32;
    a straight-through residual matches x_0_conn forward to the true cached
    x_0 while routing backward through the K active velocities (each rescaled
    by grad_rescale_factor).
    """
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

    active_velocities: dict[int, torch.Tensor] = {}
    for k in selected_indices:
        x_k = cache["x_history"][k].detach()
        v_k = _model_forward_sd35(
            args,
            transformer,
            x_k,
            timesteps[k],
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        active_velocities[int(k)] = v_k

    x_0_hat = _compose_x0_from_velocities(
        x_init=cache["x_history"][0],
        sigmas=cache["sigmas"],
        v_history=cache["v_history"],
        active_velocities=active_velocities,
        grad_rescale_factor=grad_rescale_factor,
    )

    x_0_true = cache["x0"].detach().float()
    x_0_conn = x_0_hat + (x_0_true - x_0_hat).detach()
    x_0_conn = x_0_conn.to(torch.bfloat16)

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
        # Lagrange-style aliases so existing wandb dashboards still work.
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
        "connector_scheme": "sd35_flowbp_sparse",
    }
    _maybe_save_connector_dump(args, "sd35_flowbp_sparse", connector_payload)
    args._last_connector_payload = connector_payload

    return x_0_conn, sample_weights, traj_metrics


class FlowBPSparseSD35Trainer(LeapAlignSD35Trainer):
    """FlowBP-Sparse trainer wrapper for SD3.5."""

    def train(self):
        self.args.trainer = "flowbp_sparse"
        return run_training_sd35(self.args)
