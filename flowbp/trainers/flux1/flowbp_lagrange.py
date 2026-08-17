
from __future__ import annotations

import os

import numpy as np
import torch

from flowbp.trainers.flux1.leapalign import _maybe_log_connector_wandb, _maybe_save_connector_dump
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from flowbp.trainers.common.flowbp_lagrange_utils import (
    _lagrange_integral_weights,
    _lagrange_quadrature_predict,
    _scale_nonstart_gradient_forward_value,
    _select_active_support_indices,
    _select_interval_support_indices,
    _uniform_integral_weights,
)
from flowbp.trainers.common.jk_sampling import sample_jk_indices
from flowbp.trainers.flux1.leapalign import (
    LeapAlignFluxTrainer,
    pack_latents,
    prepare_latent_image_ids,
    run_training,
    unpack_latents,
)


def _model_forward(
    args,
    transformer,
    latents,
    step_t,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    cfg_guidance,
):
    timestep = step_t.expand(latents.shape[0]).to(latents.dtype)
    with torch.autocast("cuda", torch.bfloat16):
        return transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep / 1000,
            guidance=cfg_guidance,
            txt_ids=text_ids,
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]








def _build_active_support_velocities(
    args,
    transformer,
    cache,
    support_indices: list[int],
    start_idx: int,
    current_v: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
):
    """
    Build a dict mapping active support index -> differentiable velocity.
    The start_idx uses current_v directly.
    Non-start active supports are re-forwarded with detached cached latents.
    """
    mode = args.flowbp_lagrange_grad_support_mode
    max_active = getattr(args, "flowbp_lagrange_max_active_supports", None)
    grad_scale = float(args.flowbp_lagrange_grad_support_scale)

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
            pooled_prompt_embeds,
            text_ids,
            cache["image_ids"],
            cache["cfg_guidance"],
        )
        active_velocities[idx] = _scale_nonstart_gradient_forward_value(
            v_i,
            grad_scale,
        )
    return active_velocities, active_indices








def _decode_packed_latents_for_wandb(args, vae, packed_latents, max_samples: int):
    if packed_latents is None:
        return []
    n = min(int(max_samples), packed_latents.shape[0])
    if n <= 0:
        return []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latents = unpack_latents(packed_latents[:n].to(vae.device), args.h, args.w, 8)
        latents = (latents / 0.3611) + 0.1159
        images = vae.decode(latents, return_dict=False)[0]
        images = (images * 0.5 + 0.5).clamp(0, 1)
    return [
        image.detach().float().cpu().permute(1, 2, 0).numpy()
        for image in images
    ]






def _sample_initial_latents(args, device, batch_size):
    spatial_downsample = 8
    in_channels = 16
    latent_w = args.w // spatial_downsample
    latent_h = args.h // spatial_downsample
    latents = torch.randn(
        (batch_size, in_channels, latent_h, latent_w),
        device=device,
        dtype=torch.bfloat16,
    )
    return pack_latents(
        latents,
        batch_size,
        in_channels,
        latent_h,
        latent_w,
    ), latent_h, latent_w


@torch.no_grad()
def _rollout_with_cache(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
):
    sample_steps = int(getattr(args, "rollout_steps", None) or args.sampling_steps or 25)
    batch_size = encoder_hidden_states.shape[0]
    latents_xt, latent_h, latent_w = _sample_initial_latents(args, device, batch_size)
    image_ids = prepare_latent_image_ids(
        batch_size,
        latent_h // 2,
        latent_w // 2,
        device,
        torch.bfloat16,
    )
    cfg_guidance = torch.tensor(
        [args.cfg_guidance],
        device=latents_xt.device,
        dtype=torch.bfloat16,
    ).expand(latents_xt.shape[0])

    sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
    mu = calculate_shift(
        latents_xt.shape[1],
        fm_scheduler.config.base_image_seq_len,
        fm_scheduler.config.max_image_seq_len,
        fm_scheduler.config.base_shift,
        fm_scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        fm_scheduler,
        sample_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    assert num_inference_steps == sample_steps

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
            pooled_prompt_embeds,
            text_ids,
            image_ids,
            cfg_guidance,
        )
        v_history.append(v_t.detach())
        sigma = fm_scheduler.sigmas[idx].to(device=latents_xt.device, dtype=torch.float32)
        sigma_next = fm_scheduler.sigmas[idx + 1].to(
            device=latents_xt.device,
            dtype=torch.float32,
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
        "cfg_guidance": cfg_guidance,
    }


def sample_flowbp_lagrange_trajectory(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    generator,
):
    """Two FlowBP-Lagrange connector steps with Lagrange quadrature predictors."""
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
    assert len(cache["sigmas"]) == total_steps + 1

    # ``sample_jk_indices`` draws j/k uniformly from the trainable tail
    # window and validates the sampled ordering.
    select_indices, k_idx, j_idx = sample_jk_indices(
        args, total_steps, generator,
    )

    x_k = cache["x_history"][k_idx]
    x_j = cache["x_history"][j_idx]
    x_0 = cache["x0"]
    connector_order = int(args.flowbp_lagrange_connector_order)
    detach_history = bool(args.flowbp_lagrange_detach_history)
    weight_scheme = args.flowbp_lagrange_weight_scheme
    debug_flowbp_lagrange_connector = bool(
        args.debug_flowbp_lagrange_connector
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
        pooled_prompt_embeds,
        text_ids,
        cache["image_ids"],
        cache["cfg_guidance"],
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
        pooled_prompt_embeds=pooled_prompt_embeds,
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
        weight_scheme=weight_scheme,
    )
    # Keep fp32 throughout the connector to avoid precision drift in the
    # straight-through surrogate path.
    # autocast at the transformer / VAE boundary handles dtype.

    # Euler jump baseline from v_k: xj_euler = x_k + (sigma_j - sigma_k) * v_k.
    # Keep gradient through v_k; do NOT detach pred / v_k here.
    sigma_k_kj = cache["sigmas"][k_idx].to(device=x_k.device, dtype=torch.float32)
    sigma_j_kj = cache["sigmas"][j_idx].to(device=x_k.device, dtype=torch.float32)
    xj_euler = x_k.float() + (sigma_j_kj - sigma_k_kj) * v_k.float()

    lambda_anchor = float(args.flowbp_lagrange_anchor_lambda)
    if lambda_anchor < 0.0 or lambda_anchor > 1.0:
        raise ValueError(
            f"flowbp_lagrange_anchor_lambda must be in [0, 1], got {lambda_anchor}"
        )

    # Anchored FlowBP-Lagrange connector: Euler jump from v_k is the anchor
    # path; Lagrange integration is used as a controlled correction.
    # lambda=1 uses the full Lagrange correction, lambda=0 keeps the Euler anchor.
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
        pooled_prompt_embeds,
        text_ids,
        cache["image_ids"],
        cache["cfg_guidance"],
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
        pooled_prompt_embeds=pooled_prompt_embeds,
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
        weight_scheme=weight_scheme,
    )
    # `xhat_0_pred` is already fp32 from `_lagrange_quadrature_predict`; keep
    # it in fp32 through the straight-through endpoint connector.
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
        dim=tuple(range(1, x_j.ndim)),
        keepdim=False,
    )
    d_0 = torch.abs(x_0.double() - xhat_0_pred.double()).mean(
        dim=tuple(range(1, x_0.ndim)),
        keepdim=False,
    )
    # Per-path diagnostic distances for the anchored FlowBP-Lagrange connector.
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
        "connector_scheme": "flowbp_lagrange",
    }
    _maybe_save_connector_dump(args, "flowbp_lagrange", connector_payload)
    args._last_connector_payload = connector_payload
    tau = float(args.tau)
    denom = d_j.clip(min=float(tau)) + d_0.clip(min=float(tau))
    w_sim = 1.0 / denom

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
    }
    traj_metrics.update({
        "kj_start_weight": info_kj["start_weight"].float(),
        "j0_start_weight": info_j0["start_weight"].float(),
        "kj_weight_abs_sum": info_kj["weight_abs_sum"].float(),
        "j0_weight_abs_sum": info_j0["weight_abs_sum"].float(),
        "kj_support_count": torch.tensor(
            len(info_kj["support_indices"]),
            device=x_0.device,
            dtype=torch.float32,
        ),
        "j0_support_count": torch.tensor(
            len(info_j0["support_indices"]),
            device=x_0.device,
            dtype=torch.float32,
        ),
        "kj_active_support_count": _support_stat_tensor(active_indices_kj, "count"),
        "j0_active_support_count": _support_stat_tensor(active_indices_j0, "count"),
        "kj_active_support_min": _support_stat_tensor(active_indices_kj, "min"),
        "kj_active_support_max": _support_stat_tensor(active_indices_kj, "max"),
        "j0_active_support_min": _support_stat_tensor(active_indices_j0, "min"),
        "j0_active_support_max": _support_stat_tensor(active_indices_j0, "max"),
        "flowbp_lagrange_grad_support_scale": torch.tensor(
            float(args.flowbp_lagrange_grad_support_scale),
            device=x_0.device,
            dtype=torch.float32,
        ),
        "kj_active_support_span": _support_stat_tensor(active_indices_kj, "span"),
        "j0_active_support_span": _support_stat_tensor(active_indices_j0, "span"),
        "flowbp_lagrange_weight_scheme_is_uniform": torch.tensor(
            1.0 if str(weight_scheme).lower() == "uniform" else 0.0,
            device=x_0.device,
            dtype=torch.float32,
        ),
        "flowbp_lagrange_anchor_lambda": torch.tensor(
            float(lambda_anchor),
            device=x_0.device,
            dtype=torch.float32,
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
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latents_pred_x0 = unpack_latents(latents_pred_x0, args.h, args.w, 8)
        latents_pred_x0 = (latents_pred_x0 / 0.3611) + 0.1159
        image_pred_x0 = vae.decode(latents_pred_x0, return_dict=False)[0]
        image_pred_x0 = (image_pred_x0 * 0.5 + 0.5).clamp(0, 1)

    if args.use_hpsv2:
        image_pred_x0 = preprocess_val(image_pred_x0)
        text = tokenizer(caption).to(
            device=image_pred_x0.device,
            non_blocking=True,
        )
        with torch.amp.autocast("cuda"):
            outputs = reward_model(image_pred_x0, text)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            reward_pred_x0 = torch.einsum(
                "bc,bc->b",
                image_features,
                text_features,
            )
    else:
        raise NotImplementedError("Only HPSv2 is currently supported for reward computation.")

    raw_loss = F.relu(-reward_pred_x0 + args.loss_relu_clip) * args.loss_grad_scale
    loss = torch.mean(sample_weights.to(raw_loss.device, raw_loss.dtype) * raw_loss)
    return loss, reward_pred_x0.detach().mean()


def train_flowbp_lagrange_one_step(
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
        encoder_hidden_states.shape[1], 3
    ).to(device=device, dtype=text_ids.dtype)

    latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_lagrange_trajectory(
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
        "flowbp_lagrange",
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


class FlowBPLagrangeFluxTrainer(LeapAlignFluxTrainer):
    """FlowBP-Lagrange two-segment trainer using Lagrange quadrature."""

    def train(self):
        self.args.trainer = "flowbp_lagrange"
        return run_training(self.args)

    def train_one_step(self, inner_step, device, transformer, vae, fm_scheduler, reward_model, tokenizer, optimizer, lr_scheduler, loader, preprocess_val, select_idx_generator):
        return train_flowbp_lagrange_one_step(
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
