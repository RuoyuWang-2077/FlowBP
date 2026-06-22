
"""FlowBP-Lagrange trainer for FLUX.2.

Mirrors :mod:`flowbp.trainers.flux1.flowbp_lagrange` but with the FLUX.2
forward + latent packing pipeline. All connector math (support selection,
Lagrange weights, gradient rescaling, anchor-lambda blending) is identical.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from flowbp.trainers.common.jk_sampling import sample_jk_indices
from flowbp.trainers.flux2.leapalign import (
    FLUX2_PATCH_CHANNELS,
    FLUX2_TOTAL_DOWNSAMPLE,
    LeapAlignFlux2Trainer,
    _cfg_aware_forward,
    _decode_packed_latents_for_wandb,
    _maybe_save_connector_dump,
    compute_empirical_mu,
    decode_packed_latents,
    encode_or_load_prompts,
    pack_latents,
    prepare_latent_ids,
    run_training,
)


def _model_forward(
    args,
    transformer,
    latents,
    step_t,
    encoder_hidden_states,
    text_ids,
    image_ids,
):
    """CFG-aware variant of the base helper routed through ``_cfg_aware_forward``
    so undistilled klein-base checkpoints get true classifier-free guidance.
    """
    timestep = step_t.expand(latents.shape[0]).to(latents.dtype)
    return _cfg_aware_forward(
        args,
        transformer,
        latents,
        timestep,
        encoder_hidden_states,
        text_ids,
        image_ids,
    )


def _select_interval_support_indices(
    start_idx: int,
    target_idx: int,
    num_velocities: int,
    order: int = 4,
) -> list[int]:
    assert start_idx < target_idx, (start_idx, target_idx)
    assert 0 <= start_idx < num_velocities

    order = max(1, int(order))
    last_idx = min(target_idx - 1, num_velocities - 1)
    interval = list(range(start_idx, last_idx + 1))
    if len(interval) <= order:
        return interval

    positions = torch.linspace(0, len(interval) - 1, steps=order).round().long().tolist()
    support = [interval[p] for p in positions]
    support[0] = start_idx

    dedup: list[int] = []
    for idx in support:
        if idx not in dedup:
            dedup.append(idx)
    return dedup


def _select_active_support_indices(
    support_indices: list[int],
    start_idx: int,
    mode: str = "start",
    max_active: int | None = None,
) -> list[int]:
    mode = str(mode).lower()
    if start_idx not in support_indices and mode != "none":
        raise ValueError(
            f"start_idx={start_idx} must be present in support_indices={support_indices}"
        )

    if mode == "none":
        selected: list[int] = []
    elif mode == "start":
        selected = [start_idx]
    elif mode == "midpoint":
        if len(support_indices) <= 1:
            selected = [start_idx]
        else:
            selected = support_indices[:2]
    elif mode == "all":
        selected = list(support_indices)
    else:
        raise ValueError(
            "Invalid flowbp_lagrange_grad_support_mode="
            f"{mode!r}; expected one of 'none', 'start', 'midpoint', or 'all'."
        )

    selected_set = set(selected)
    ordered = [idx for idx in support_indices if idx in selected_set]
    if max_active is not None:
        max_active = int(max_active)
    if max_active is not None and max_active > 0:
        ordered = ordered[:max_active]
    return ordered


def _scale_nonstart_gradient_forward_value(
    tensor: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if scale == 1:
        return tensor
    if scale == 0:
        return tensor.detach()
    detached = tensor.detach()
    return detached + float(scale) * (tensor - detached)


def _build_active_support_velocities(
    args,
    transformer,
    cache,
    support_indices: list[int],
    start_idx: int,
    current_v: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states,
    text_ids,
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
    support_set = set(support_indices)
    if any(idx not in support_set for idx in active_indices):
        raise RuntimeError(
            f"active_indices={active_indices} must be a subset of "
            f"support_indices={support_indices}"
        )
    if str(mode).lower() != "none" and start_idx not in active_indices:
        raise RuntimeError(
            f"start_idx={start_idx} must be active for mode={mode!r}; "
            f"active_indices={active_indices}"
        )

    active_velocities: dict[int, torch.Tensor] = {}
    for idx in active_indices:
        if idx == start_idx:
            active_velocities[idx] = current_v
            continue
        if not (0 <= idx < len(cache["x_history"])):
            raise RuntimeError(
                f"Non-start active support idx={idx} is out of range for "
                f"x_history length={len(cache['x_history'])}"
            )
        x_i = cache["x_history"][idx].detach()
        v_i = _model_forward(
            args,
            transformer,
            x_i,
            timesteps[idx],
            encoder_hidden_states,
            text_ids,
            cache["image_ids"],
        )
        active_velocities[idx] = _scale_nonstart_gradient_forward_value(
            v_i, grad_scale,
        )
    return active_velocities, active_indices


def _lagrange_integral_weights(
    sigma_points: torch.Tensor,
    sigma_start: torch.Tensor,
    sigma_target: torch.Tensor,
    device,
    dtype,
) -> torch.Tensor:
    points64 = sigma_points.detach().to(device=device, dtype=torch.float64)
    start64 = torch.as_tensor(sigma_start, device=device, dtype=torch.float64)
    target64 = torch.as_tensor(sigma_target, device=device, dtype=torch.float64)

    weights = []
    for i in range(points64.numel()):
        coeff = torch.ones(1, device=device, dtype=torch.float64)
        denom = torch.ones((), device=device, dtype=torch.float64)
        for j in range(points64.numel()):
            if j == i:
                continue
            new_coeff = torch.zeros(
                coeff.numel() + 1, device=device, dtype=torch.float64,
            )
            new_coeff[:-1] += -points64[j] * coeff
            new_coeff[1:] += coeff
            coeff = new_coeff
            denom = denom * (points64[i] - points64[j])

        coeff = coeff / denom
        integral = torch.zeros((), device=device, dtype=torch.float64)
        for power, c in enumerate(coeff):
            integral = integral + c / float(power + 1) * (
                target64 ** (power + 1) - start64 ** (power + 1)
            )
        weights.append(integral)

    return torch.stack(weights).to(device=device, dtype=dtype)


def _uniform_integral_weights(
    num_points: int,
    sigma_start: torch.Tensor,
    sigma_target: torch.Tensor,
    device,
    dtype,
) -> torch.Tensor:
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    start = torch.as_tensor(sigma_start, device=device, dtype=torch.float64)
    target = torch.as_tensor(sigma_target, device=device, dtype=torch.float64)
    weight = (target - start) / float(num_points)
    return torch.full((num_points,), weight.item(), device=device, dtype=dtype)


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
    for name in ("xj_pred", "xj", "x0_pred", "x0"):
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


def _lagrange_quadrature_predict(
    x_start,
    current_v,
    cached_velocities,
    sigmas,
    start_idx,
    target_idx,
    order=4,
    detach_history=True,
    support_indices=None,
    active_velocities=None,
    grad_rescale=0.0,
    weight_scheme="lagrange",
):
    assert target_idx > start_idx, (start_idx, target_idx)
    assert len(sigmas) == len(cached_velocities) + 1
    assert 0 <= start_idx < len(cached_velocities)
    assert 0 < target_idx < len(sigmas)

    device = x_start.device
    dtype = torch.float32
    if support_indices is None:
        support_indices = _select_interval_support_indices(
            start_idx=start_idx,
            target_idx=target_idx,
            num_velocities=len(cached_velocities),
            order=order,
        )
    else:
        support_indices = list(support_indices)
    assert target_idx not in support_indices, (target_idx, support_indices)
    assert all(0 <= idx < len(cached_velocities) for idx in support_indices)

    use_legacy_start_gradient = active_velocities is None
    if active_velocities is None:
        active_velocities = {}
    active_support_indices = [
        idx for idx in support_indices if idx in active_velocities
    ]
    sigma_points = sigmas[support_indices].to(device=device, dtype=torch.float64)
    sigma_start = sigmas[start_idx].to(device=device, dtype=torch.float64)
    sigma_target = sigmas[target_idx].to(device=device, dtype=torch.float64)
    weight_scheme = str(weight_scheme).lower()
    if weight_scheme == "lagrange":
        weights = _lagrange_integral_weights(
            sigma_points=sigma_points,
            sigma_start=sigma_start,
            sigma_target=sigma_target,
            device=device,
            dtype=dtype,
        )
    elif weight_scheme == "uniform":
        weights = _uniform_integral_weights(
            num_points=len(support_indices),
            sigma_start=sigma_start,
            sigma_target=sigma_target,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            f"Invalid flowbp_lagrange_weight_scheme={weight_scheme!r}; "
            "expected 'lagrange' or 'uniform'."
        )

    grad_rescale_factor = 1.0
    if grad_rescale > 0 and active_support_indices:
        total_abs_weight = weights.abs().sum()
        idx_to_pos = {idx: pos for pos, idx in enumerate(support_indices)}
        active_abs_weight = sum(
            weights[idx_to_pos[idx]].abs() for idx in active_support_indices
        )
        if active_abs_weight > 1e-8:
            full_factor = (total_abs_weight / active_abs_weight).item()
            grad_rescale_factor = 1.0 + grad_rescale * (full_factor - 1.0)

    delta = torch.zeros_like(x_start, dtype=torch.float32)
    for weight, idx in zip(weights, support_indices):
        if idx in active_velocities:
            velocity = active_velocities[idx]
            if grad_rescale_factor != 1.0:
                v_detached = velocity.detach()
                velocity = v_detached + grad_rescale_factor * (velocity - v_detached)
        elif idx == start_idx and use_legacy_start_gradient:
            velocity = current_v
            if grad_rescale_factor != 1.0:
                v_detached = velocity.detach()
                velocity = v_detached + grad_rescale_factor * (velocity - v_detached)
        else:
            velocity = cached_velocities[idx]
            if detach_history:
                velocity = velocity.detach()
        delta = delta + weight * velocity.float()

    info = {
        "support_indices": support_indices,
        "active_support_indices": active_support_indices,
        "weights": weights.detach(),
        "start_weight": weights[0].detach(),
        "weight_abs_sum": weights.abs().sum().detach(),
        "grad_rescale_factor": torch.tensor(
            grad_rescale_factor, device=device, dtype=torch.float32
        ),
    }
    return x_start.float() + delta, info


def _sample_initial_latents_flux2(args, device, batch_size):
    h_tokens = args.h // FLUX2_TOTAL_DOWNSAMPLE
    w_tokens = args.w // FLUX2_TOTAL_DOWNSAMPLE
    latents = torch.randn(
        (batch_size, FLUX2_PATCH_CHANNELS, h_tokens, w_tokens),
        device=device,
        dtype=torch.bfloat16,
    )
    return pack_latents(latents), h_tokens, w_tokens


@torch.no_grad()
def _rollout_with_cache(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    text_ids,
):
    sample_steps = int(getattr(args, "rollout_steps", None) or args.sampling_steps or 25)
    batch_size = encoder_hidden_states.shape[0]
    latents_xt, h_tokens, w_tokens = _sample_initial_latents_flux2(
        args, device, batch_size,
    )
    # int64 to match the inference pipeline; bf16 would silently quantize
    # any position >= 256 inside `Flux2PosEmbed`. See the matching comment in
    # `leapalign_flux2_trainer.sample_trajectory`.
    image_ids = prepare_latent_ids(
        batch_size, h_tokens, w_tokens, device, torch.int64,
    )

    sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
    image_seq_len = latents_xt.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=sample_steps)
    fm_scheduler.set_timesteps(sample_steps, device=device, sigmas=sigmas, mu=mu)
    timesteps = fm_scheduler.timesteps
    assert timesteps.shape[0] == sample_steps

    x_history = []
    v_history = []
    for idx, step_t in enumerate(timesteps):
        x_history.append(latents_xt.detach())
        v_t = _model_forward(
            args,
            transformer,
            latents_xt,
            step_t,
            encoder_hidden_states,
            text_ids,
            image_ids,
        )
        v_history.append(v_t.detach())
        sigma = fm_scheduler.sigmas[idx].to(
            device=latents_xt.device, dtype=torch.float32,
        )
        sigma_next = fm_scheduler.sigmas[idx + 1].to(
            device=latents_xt.device, dtype=torch.float32,
        )
        latents_xt = (
            latents_xt.float() + (sigma_next - sigma) * v_t.float()
        ).to(torch.bfloat16)

    return {
        "x_history": x_history,
        "v_history": v_history,
        "x0": latents_xt.detach(),
        "timesteps": timesteps,
        "sigmas": fm_scheduler.sigmas,
        "image_ids": image_ids,
    }


def sample_flowbp_lagrange_trajectory(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    text_ids,
    generator,
):
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
    assert len(cache["sigmas"]) == total_steps + 1

    # j-k index sampling: dispatch on ``args.jk_sampling_mode`` (uniform /
    # dirichlet / midpoint / midpoint_j). The helper validates the sampled
    # ordering.
    select_indices, k_idx, j_idx = sample_jk_indices(
        args, total_steps, generator,
    )

    x_k = cache["x_history"][k_idx]
    x_j = cache["x_history"][j_idx]
    x_0 = cache["x0"]
    connector_order = int(getattr(args, "flowbp_lagrange_connector_order", 4))
    detach_history = bool(getattr(args, "flowbp_lagrange_detach_history", True))
    grad_rescale = float(getattr(args, "flowbp_lagrange_grad_rescale", 0.0))
    weight_scheme = getattr(args, "flowbp_lagrange_weight_scheme", "lagrange")
    debug_flowbp_lagrange_connector = bool(
        getattr(args, "debug_flowbp_lagrange_connector", False)
    )

    support_kj = _select_interval_support_indices(
        start_idx=k_idx,
        target_idx=j_idx,
        num_velocities=len(cache["v_history"]),
        order=connector_order,
    )
    v_k = _model_forward(
        args,
        transformer,
        x_k,
        timesteps[k_idx],
        encoder_hidden_states,
        text_ids,
        cache["image_ids"],
    )
    active_velocities_kj, active_indices_kj = _build_active_support_velocities(
        args=args,
        transformer=transformer,
        cache=cache,
        support_indices=support_kj,
        start_idx=k_idx,
        current_v=v_k,
        timesteps=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        text_ids=text_ids,
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

    sigma_k_kj = cache["sigmas"][k_idx].to(device=x_k.device, dtype=torch.float32)
    sigma_j_kj = cache["sigmas"][j_idx].to(device=x_k.device, dtype=torch.float32)
    xj_euler = x_k.float() + (sigma_j_kj - sigma_k_kj) * v_k.float()

    lambda_anchor = float(getattr(args, "flowbp_lagrange_anchor_lambda", 1.0))
    if lambda_anchor < 0.0 or lambda_anchor > 1.0:
        raise ValueError(
            f"flowbp_lagrange_anchor_lambda must be in [0, 1], got {lambda_anchor}"
        )

    if lambda_anchor == 1.0:
        xhat_j_pred = xj_lagrange
    elif lambda_anchor == 0.0:
        xhat_j_pred = xj_euler
    else:
        xhat_j_pred = xj_euler + lambda_anchor * (xj_lagrange - xj_euler)

    xj_conn = xhat_j_pred + (x_j - xhat_j_pred).detach()

    if debug_flowbp_lagrange_connector:
        xhat_euler = x_k.float() + (
            cache["sigmas"][j_idx].to(x_k.device, torch.float32)
            - cache["sigmas"][k_idx].to(x_k.device, torch.float32)
        ) * v_k.float()
        xhat_order1, _ = _lagrange_quadrature_predict(
            x_start=x_k,
            current_v=v_k,
            cached_velocities=cache["v_history"],
            sigmas=cache["sigmas"],
            start_idx=k_idx,
            target_idx=j_idx,
            order=1,
            detach_history=True,
            weight_scheme="lagrange",
        )
        err = (xhat_euler - xhat_order1).abs().max()
        if err > 1e-4:
            raise RuntimeError(
                f"Order-1 connector does not match Euler: err={err.item()}"
            )

    alpha = float(args.alpha)
    xj_model_input = float(alpha) * xj_conn + (1.0 - float(alpha)) * xj_conn.detach()

    support_j0 = _select_interval_support_indices(
        start_idx=j_idx,
        target_idx=total_steps,
        num_velocities=len(cache["v_history"]),
        order=connector_order,
    )
    v_j = _model_forward(
        args,
        transformer,
        xj_model_input,
        timesteps[j_idx],
        encoder_hidden_states,
        text_ids,
        cache["image_ids"],
    )
    active_velocities_j0, active_indices_j0 = _build_active_support_velocities(
        args=args,
        transformer=transformer,
        cache=cache,
        support_indices=support_j0,
        start_idx=j_idx,
        current_v=v_j,
        timesteps=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        text_ids=text_ids,
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
    x0_conn = xhat_0_pred + (x_0 - xhat_0_pred).detach()

    if debug_flowbp_lagrange_connector:
        with torch.no_grad():
            err_j = (xj_conn.detach() - x_j.detach()).abs().max()
            err_0 = (x0_conn.detach() - x_0.detach()).abs().max()
            if err_j > 1e-3 or err_0 > 1e-3:
                raise RuntimeError(
                    "Connector identity failed: "
                    f"err_j={err_j.item()}, err_0={err_0.item()}"
                )

    d_j = torch.abs(x_j.double() - xhat_j_pred.double()).mean(
        dim=tuple(range(1, x_j.ndim)), keepdim=False,
    )
    d_0 = torch.abs(x_0.double() - xhat_0_pred.double()).mean(
        dim=tuple(range(1, x_0.ndim)), keepdim=False,
    )
    d_j_euler_per_sample = torch.abs(x_j.double() - xj_euler.double()).mean(
        dim=tuple(range(1, x_j.ndim)), keepdim=False,
    )
    d_j_lagrange_per_sample = torch.abs(
        x_j.double() - xj_lagrange.double()
    ).mean(dim=tuple(range(1, x_j.ndim)), keepdim=False)

    connector_payload = {
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
        "flowbp_lagrange_anchor_lambda": float(lambda_anchor),
        "dj_euler_per_sample": d_j_euler_per_sample.detach(),
        "dj_lagrange_per_sample": d_j_lagrange_per_sample.detach(),
        "connector_scheme": "flowbp_lagrange_flux2",
    }
    _maybe_save_connector_dump(args, "flowbp_lagrange_flux2", connector_payload)
    args._last_connector_payload = connector_payload

    tau = float(args.tau)
    denom = d_j.clip(min=float(tau)) + d_0.clip(min=float(tau))
    w_sim = 1.0 / denom

    # Optional per-sample masks: drop samples whose connector approximation
    # errors d_j or d_0 exceed their thresholds from the reward gradient.
    # The sample's forward still happens (it lives in the same batch and
    # goes through the transformer), but its contribution to the reward
    # loss is zeroed via ``sample_weights``, so it produces no gradient.
    # Useful when reward fine-tuning makes the velocity field rough on some
    # batches and you want to drop the outliers instead of clipping
    # gradients globally.
    #
    # The two clips are independent:
    #   * ``clip_dj_threshold > 0``  enables the d_j clip (default off).
    #   * ``clip_d0`` flag           enables the d_0 clip (default off);
    #     when on, the threshold defaults to ``clip_d0_threshold = 0.2``.
    # When both are enabled the per-sample masks are AND'd together.
    batch_size = w_sim.shape[0]

    clip_dj_threshold = float(getattr(args, "clip_dj_threshold", 0.0) or 0.0)
    if clip_dj_threshold > 0.0:
        keep_mask_dj = (d_j <= clip_dj_threshold).to(w_sim.dtype)
        kept_count = keep_mask_dj.sum().detach()
        keep_ratio = keep_mask_dj.mean().detach()
    else:
        keep_mask_dj = torch.ones(
            batch_size, device=w_sim.device, dtype=w_sim.dtype,
        )
        kept_count = torch.tensor(
            float(batch_size), device=w_sim.device, dtype=torch.float32,
        )
        keep_ratio = torch.tensor(1.0, device=w_sim.device, dtype=torch.float32)

    clip_d0_enabled = bool(getattr(args, "clip_d0", False))
    clip_d0_threshold = float(getattr(args, "clip_d0_threshold", 0.2) or 0.2)
    if clip_d0_enabled and clip_d0_threshold > 0.0:
        keep_mask_d0 = (d_0 <= clip_d0_threshold).to(w_sim.dtype)
        kept_count_d0 = keep_mask_d0.sum().detach()
        keep_ratio_d0 = keep_mask_d0.mean().detach()
    else:
        keep_mask_d0 = torch.ones(
            batch_size, device=w_sim.device, dtype=w_sim.dtype,
        )
        kept_count_d0 = torch.tensor(
            float(batch_size), device=w_sim.device, dtype=torch.float32,
        )
        keep_ratio_d0 = torch.tensor(
            1.0, device=w_sim.device, dtype=torch.float32,
        )

    w_sim = w_sim * keep_mask_dj * keep_mask_d0

    def _support_stat_tensor(indices: list[int], stat: str) -> torch.Tensor:
        if stat == "count":
            value = len(indices)
        elif not indices:
            value = -1
        elif stat == "min":
            value = min(indices)
        elif stat == "max":
            value = max(indices)
        elif stat == "span":
            value = max(indices) - min(indices)
        else:
            raise ValueError(f"Unknown support stat: {stat}")
        return torch.tensor(value, device=x_0.device, dtype=torch.float32)

    traj_metrics = {
        "d0": d_0.detach().mean(),
        "dj": d_j.detach().mean(),
        "w_sim": w_sim.detach().mean(),
        "clip_dj_kept_count": kept_count,
        "clip_dj_keep_ratio": keep_ratio,
        "clip_dj_threshold": torch.tensor(
            clip_dj_threshold,
            device=x_0.device,
            dtype=torch.float32,
        ),
        "clip_d0_kept_count": kept_count_d0,
        "clip_d0_keep_ratio": keep_ratio_d0,
        "clip_d0_threshold": torch.tensor(
            clip_d0_threshold if clip_d0_enabled else 0.0,
            device=x_0.device,
            dtype=torch.float32,
        ),
        "clip_d0_enabled": torch.tensor(
            1.0 if clip_d0_enabled else 0.0,
            device=x_0.device,
            dtype=torch.float32,
        ),
    }
    traj_metrics.update({
        "kj_start_weight": info_kj["start_weight"].float(),
        "j0_start_weight": info_j0["start_weight"].float(),
        "kj_weight_abs_sum": info_kj["weight_abs_sum"].float(),
        "j0_weight_abs_sum": info_j0["weight_abs_sum"].float(),
        "kj_support_count": torch.tensor(
            len(info_kj["support_indices"]),
            device=x_0.device, dtype=torch.float32,
        ),
        "j0_support_count": torch.tensor(
            len(info_j0["support_indices"]),
            device=x_0.device, dtype=torch.float32,
        ),
        "kj_active_support_count": _support_stat_tensor(active_indices_kj, "count"),
        "j0_active_support_count": _support_stat_tensor(active_indices_j0, "count"),
        "kj_active_support_min": _support_stat_tensor(active_indices_kj, "min"),
        "kj_active_support_max": _support_stat_tensor(active_indices_kj, "max"),
        "j0_active_support_min": _support_stat_tensor(active_indices_j0, "min"),
        "j0_active_support_max": _support_stat_tensor(active_indices_j0, "max"),
        "flowbp_lagrange_grad_support_scale": torch.tensor(
            float(getattr(args, "flowbp_lagrange_grad_support_scale", 1.0)),
            device=x_0.device, dtype=torch.float32,
        ),
        "kj_active_support_span": _support_stat_tensor(active_indices_kj, "span"),
        "j0_active_support_span": _support_stat_tensor(active_indices_j0, "span"),
        "kj_grad_rescale": info_kj["grad_rescale_factor"],
        "j0_grad_rescale": info_j0["grad_rescale_factor"],
        "flowbp_lagrange_weight_scheme_is_uniform": torch.tensor(
            1.0 if str(weight_scheme).lower() == "uniform" else 0.0,
            device=x_0.device, dtype=torch.float32,
        ),
        "flowbp_lagrange_anchor_lambda": torch.tensor(
            float(lambda_anchor),
            device=x_0.device, dtype=torch.float32,
        ),
        "jk_truncated": torch.tensor(
            1.0 if bool(getattr(args, "_last_jk_truncated", False)) else 0.0,
            device=x_0.device, dtype=torch.float32,
        ),
        "dj_euler": d_j_euler_per_sample.detach().float().mean(),
        "dj_lagrange": d_j_lagrange_per_sample.detach().float().mean(),
    })
    return x0_conn, w_sim.detach(), traj_metrics


def get_flowbp_lagrange_reward_loss(
    args,
    latents_pred_x0,
    sample_weights,
    vae,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
):
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
            "Only HPSv2 is currently supported for reward computation."
        )

    raw_loss = F.relu(-reward_pred_x0 + args.loss_relu_clip) * args.loss_grad_scale
    loss = torch.mean(sample_weights.to(raw_loss.device, raw_loss.dtype) * raw_loss)
    return loss, reward_pred_x0.detach().mean()


def train_flowbp_lagrange_one_step_flux2(
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

    latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_lagrange_trajectory(
        args,
        device,
        transformer,
        fm_scheduler,
        prompt_embeds,
        text_ids,
        select_idx_generator,
    )
    _maybe_log_connector_wandb(
        args, vae, "flowbp_lagrange_flux2",
        getattr(args, "_last_connector_payload", None),
    )

    loss, avg_reward_pred_x0 = get_flowbp_lagrange_reward_loss(
        args,
        latents_pred_x0,
        sample_weights,
        vae,
        reward_model,
        tokenizer,
        caption,
        preprocess_val,
    )

    # Capture the un-scaled loss before applying both grad-accum averaging
    # and the CFG gradient compensation. With ``_cfg_aware_forward`` detaching
    # the negative branch, the effective gradient through theta is multiplied
    # by ``cfg_guidance``; dividing the backward loss by ``cfg_guidance``
    # recovers the non-CFG gradient magnitude and keeps grad_norm /
    # max_grad_norm thresholds comparable across cfg settings. Disable
    # ``cfg_grad_norm_compensate`` to use the raw CFG-amplified gradient.
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


class FlowBPLagrangeFlux2Trainer(LeapAlignFlux2Trainer):
    """FlowBP-Lagrange two-segment trainer for FLUX.2."""

    def train(self):
        self.args.trainer = "flowbp_lagrange"
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
        return train_flowbp_lagrange_one_step_flux2(
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
