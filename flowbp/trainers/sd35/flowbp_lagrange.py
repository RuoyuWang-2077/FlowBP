"""FlowBP-Lagrange trainer for SD3.5.

Owns the shared no-grad rollout cache (``_rollout_with_cache_sd35``) that the
FlowBP-Sparse and FlowBP-Bridge SD3.5 trainers reuse.
"""

from __future__ import annotations

import torch

from flowbp.trainers.common.flowbp_lagrange_utils import (
    _lagrange_quadrature_predict,
    _scale_nonstart_gradient_forward_value,
    _select_active_support_indices,
    _select_interval_support_indices,
)
from flowbp.trainers.common.jk_sampling import sample_jk_indices
from flowbp.trainers.sd35.leapalign import (
    LeapAlignSD35Trainer,
    _build_traj_filter_mask,
    _maybe_save_connector_dump,
    _model_forward_sd35,
    _prepare_timesteps,
    _sample_initial_latents,
    run_training_sd35,
)


@torch.no_grad()
def _rollout_with_cache_sd35(
    args,
    device,
    transformer,
    fm_scheduler,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
):
    latents_xt, _, _ = _sample_initial_latents(args, transformer, device, prompt_embeds.shape[0])
    timesteps = _prepare_timesteps(args, fm_scheduler, transformer, latents_xt, device)
    x_history = []
    v_history = []
    for idx, step_t in enumerate(timesteps):
        x_history.append(latents_xt.detach())
        v_t = _model_forward_sd35(
            args,
            transformer,
            latents_xt,
            step_t,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        v_history.append(v_t.detach())
        sigma = fm_scheduler.sigmas[idx].to(device=latents_xt.device, dtype=torch.float32)
        sigma_next = fm_scheduler.sigmas[idx + 1].to(device=latents_xt.device, dtype=torch.float32)
        latents_xt = (latents_xt.float() + (sigma_next - sigma) * v_t.float()).to(torch.bfloat16)

    return {
        "x_history": x_history,
        "v_history": v_history,
        "x0": latents_xt.detach(),
        "timesteps": timesteps,
        "sigmas": fm_scheduler.sigmas,
    }


def _build_active_support_velocities_sd35(
    args,
    transformer,
    cache,
    support_indices,
    start_idx,
    current_v,
    timesteps,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
):
    mode = getattr(args, "flowbp_lagrange_grad_support_mode", "start")
    max_active = getattr(args, "flowbp_lagrange_max_active_supports", None)
    grad_scale = float(getattr(args, "flowbp_lagrange_grad_support_scale", 1.0))
    active_indices = _select_active_support_indices(
        support_indices=support_indices,
        start_idx=start_idx,
        mode=mode,
        max_active=max_active,
    )
    active_velocities = {}
    for idx in active_indices:
        if idx == start_idx:
            active_velocities[idx] = current_v
            continue
        x_i = cache["x_history"][idx].detach()
        v_i = _model_forward_sd35(
            args,
            transformer,
            x_i,
            timesteps[idx],
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        active_velocities[idx] = _scale_nonstart_gradient_forward_value(v_i, grad_scale)
    return active_velocities, active_indices


def sample_flowbp_lagrange_trajectory_sd35(
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
    select_indices, k_idx, j_idx = sample_jk_indices(args, total_steps, generator)
    jk_truncated = bool(getattr(args, "_last_jk_truncated", False))
    x_k = cache["x_history"][k_idx]
    x_j = cache["x_history"][j_idx]
    x_0 = cache["x0"]
    connector_order = int(getattr(args, "flowbp_lagrange_connector_order", 4))
    detach_history = bool(getattr(args, "flowbp_lagrange_detach_history", True))
    grad_rescale = float(getattr(args, "flowbp_lagrange_grad_rescale", 0.0))
    weight_scheme = getattr(args, "flowbp_lagrange_weight_scheme", "lagrange")

    support_kj = _select_interval_support_indices(
        start_idx=k_idx,
        target_idx=j_idx,
        num_velocities=len(cache["v_history"]),
        order=connector_order,
    )
    v_k = _model_forward_sd35(
        args,
        transformer,
        x_k,
        timesteps[k_idx],
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )
    active_velocities_kj, active_indices_kj = _build_active_support_velocities_sd35(
        args,
        transformer,
        cache,
        support_kj,
        k_idx,
        v_k,
        timesteps,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )
    xj_lagrange, info_kj = _lagrange_quadrature_predict(
        x_start=x_k,
        current_v=v_k,
        cached_velocities=cache["v_history"],
        sigmas=cache["sigmas"],
        start_idx=k_idx,
        target_idx=j_idx,
        order=connector_order,
        detach_history=detach_history,
        support_indices=support_kj,
        active_velocities=active_velocities_kj,
        grad_rescale=grad_rescale,
        weight_scheme=weight_scheme,
    )
    sigma_k = cache["sigmas"][k_idx].to(device=x_k.device, dtype=torch.float32)
    sigma_j = cache["sigmas"][j_idx].to(device=x_k.device, dtype=torch.float32)
    xj_euler = x_k.float() + (sigma_j - sigma_k) * v_k.float()

    anchor_lambda = float(getattr(args, "flowbp_lagrange_anchor_lambda", 1.0))
    if anchor_lambda < 0.0 or anchor_lambda > 1.0:
        raise ValueError(
            f"flowbp_lagrange_anchor_lambda must be in [0, 1], got {anchor_lambda}"
        )
    if anchor_lambda == 1.0:
        xhat_j_pred = xj_lagrange
    elif anchor_lambda == 0.0:
        xhat_j_pred = xj_euler
    else:
        xhat_j_pred = xj_euler + anchor_lambda * (xj_lagrange - xj_euler)
    xhat_j_pred = xhat_j_pred.to(torch.bfloat16)
    xj_conn = xhat_j_pred + (x_j - xhat_j_pred).detach()

    alpha = float(args.alpha)
    xj_model_input = float(alpha) * xj_conn + (1.0 - float(alpha)) * xj_conn.detach()

    support_j0 = _select_interval_support_indices(
        start_idx=j_idx,
        target_idx=total_steps,
        num_velocities=len(cache["v_history"]),
        order=connector_order,
    )
    v_j = _model_forward_sd35(
        args,
        transformer,
        xj_model_input,
        timesteps[j_idx],
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )
    active_velocities_j0, active_indices_j0 = _build_active_support_velocities_sd35(
        args,
        transformer,
        cache,
        support_j0,
        j_idx,
        v_j,
        timesteps,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )
    xhat_0_pred, info_j0 = _lagrange_quadrature_predict(
        x_start=xj_conn,
        current_v=v_j,
        cached_velocities=cache["v_history"],
        sigmas=cache["sigmas"],
        start_idx=j_idx,
        target_idx=total_steps,
        order=connector_order,
        detach_history=detach_history,
        support_indices=support_j0,
        active_velocities=active_velocities_j0,
        grad_rescale=grad_rescale,
        weight_scheme=weight_scheme,
    )
    xhat_0_pred = xhat_0_pred.to(torch.bfloat16)
    x0_conn = xhat_0_pred + (x_0 - xhat_0_pred).detach()

    d_j = torch.abs(x_j.double() - xhat_j_pred.double()).mean(dim=tuple(range(1, x_j.ndim)), keepdim=False)
    d_0 = torch.abs(x_0.double() - xhat_0_pred.double()).mean(dim=tuple(range(1, x_0.ndim)), keepdim=False)
    d_j_euler_per_sample = torch.abs(x_j.double() - xj_euler.double()).mean(
        dim=tuple(range(1, x_j.ndim)),
        keepdim=False,
    )
    d_j_lagrange_per_sample = torch.abs(
        x_j.double() - xj_lagrange.double()
    ).mean(
        dim=tuple(range(1, x_j.ndim)),
        keepdim=False,
    )
    payload = {
        "step": getattr(args, "current_train_step", None),
        "k_idx": k_idx,
        "j_idx": j_idx,
        "select_indices": select_indices.detach(),
        "xj_pred": xhat_j_pred.detach(),
        "xj": x_j.detach(),
        "x0_pred": xhat_0_pred.detach(),
        "x0": x_0.detach(),
        "dj_per_sample": d_j.detach(),
        "d0_per_sample": d_0.detach(),
        "support_kj": support_kj,
        "support_j0": support_j0,
        "weights_kj": info_kj["weights"].detach(),
        "weights_j0": info_j0["weights"].detach(),
        "weight_scheme": str(weight_scheme),
        "xj_euler": xj_euler.detach(),
        "xj_lagrange": xj_lagrange.detach(),
        "xj_anchor": xhat_j_pred.detach(),
        "flowbp_lagrange_anchor_lambda": float(anchor_lambda),
        "dj_euler_per_sample": d_j_euler_per_sample.detach(),
        "dj_lagrange_per_sample": d_j_lagrange_per_sample.detach(),
        "connector_scheme": "sd35_flowbp_lagrange",
    }
    _maybe_save_connector_dump(args, "sd35_flowbp_lagrange", payload)
    args._last_connector_payload = payload

    tau = float(args.tau)
    d_j_clipped = d_j.clip(min=float(tau))
    d_0_clipped = d_0.clip(min=float(tau))
    w_sim = 1.0 / (d_j_clipped + d_0_clipped)
    _, filter_metrics = _build_traj_filter_mask(args, d_j, d_0)

    def stat(indices, name):
        if name == "count":
            value = len(indices)
        elif not indices:
            value = -1
        elif name == "min":
            value = min(indices)
        elif name == "max":
            value = max(indices)
        elif name == "span":
            value = max(indices) - min(indices)
        else:
            raise ValueError(name)
        return torch.tensor(value, device=x_0.device, dtype=torch.float32)

    traj_metrics = {
        "d0": d_0.detach().mean(),
        "dj": d_j.detach().mean(),
        "d0_clipped": d_0_clipped.detach().mean(),
        "dj_clipped": d_j_clipped.detach().mean(),
        "w_sim": w_sim.detach().mean(),
        "d0_max": d_0.detach().max(),
        "dj_max": d_j.detach().max(),
        "w_sim_max": w_sim.detach().max(),
        "w_sim_min": w_sim.detach().min(),
        "w_sim_std": w_sim.detach().std(),
        "dj_euler": d_j_euler_per_sample.detach().mean(),
        "dj_lagrange": d_j_lagrange_per_sample.detach().mean(),
        "dj_euler_max": d_j_euler_per_sample.detach().max(),
        "dj_lagrange_max": d_j_lagrange_per_sample.detach().max(),
        "j_idx": torch.tensor(float(j_idx), device=x_0.device, dtype=torch.float32),
        "k_idx": torch.tensor(float(k_idx), device=x_0.device, dtype=torch.float32),
        "jk_gap": torch.tensor(float(j_idx - k_idx), device=x_0.device, dtype=torch.float32),
        "j_rev": torch.tensor(float(total_steps - j_idx), device=x_0.device, dtype=torch.float32),
        "k_rev": torch.tensor(float(total_steps - k_idx), device=x_0.device, dtype=torch.float32),
        "jk_truncated": torch.tensor(1.0 if jk_truncated else 0.0, device=x_0.device, dtype=torch.float32),
        "flowbp_lagrange_anchor_lambda": torch.tensor(
            float(anchor_lambda), device=x_0.device, dtype=torch.float32
        ),
        "kj_start_weight": info_kj["start_weight"].float(),
        "j0_start_weight": info_j0["start_weight"].float(),
        "kj_weight_abs_sum": info_kj["weight_abs_sum"].float(),
        "j0_weight_abs_sum": info_j0["weight_abs_sum"].float(),
        "kj_support_count": torch.tensor(len(info_kj["support_indices"]), device=x_0.device, dtype=torch.float32),
        "j0_support_count": torch.tensor(len(info_j0["support_indices"]), device=x_0.device, dtype=torch.float32),
        "kj_active_support_count": stat(active_indices_kj, "count"),
        "j0_active_support_count": stat(active_indices_j0, "count"),
        "kj_active_support_min": stat(active_indices_kj, "min"),
        "kj_active_support_max": stat(active_indices_kj, "max"),
        "j0_active_support_min": stat(active_indices_j0, "min"),
        "j0_active_support_max": stat(active_indices_j0, "max"),
        "kj_active_support_span": stat(active_indices_kj, "span"),
        "j0_active_support_span": stat(active_indices_j0, "span"),
        "kj_grad_rescale": info_kj["grad_rescale_factor"],
        "j0_grad_rescale": info_j0["grad_rescale_factor"],
        "flowbp_lagrange_weight_scheme_is_uniform": torch.tensor(
            1.0 if str(weight_scheme).lower() == "uniform" else 0.0,
            device=x_0.device,
            dtype=torch.float32,
        ),
        "flowbp_lagrange_weight_scheme_is_ab": torch.tensor(
            1.0 if str(weight_scheme).lower() == "adams_bashforth" else 0.0,
            device=x_0.device,
            dtype=torch.float32,
        ),
    }
    traj_metrics.update(filter_metrics)
    return x0_conn, w_sim.detach(), traj_metrics


class FlowBPLagrangeSD35Trainer(LeapAlignSD35Trainer):
    def train(self):
        self.args.trainer = "flowbp_lagrange"
        return run_training_sd35(self.args)
