from __future__ import annotations

import os
import time
from collections import deque
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils import check_min_version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from flowbp.dataset.latent_sd35_rl_datasets import (
    LatentSD35Dataset,
    latent_sd35_collate_function,
)
from flowbp.eval.runners.online import run_online_sd35_eval
from flowbp.schedulers.fm_euler_discrete_with_pred_x0 import (
    FlowMatchEulerDiscreteWithPredX0Scheduler,
)
from flowbp.trainers.common.flowbp_lagrange_utils import (
    _jk_histogram_buffer,
    save_jk_sampling_csv,
)
from flowbp.trainers.common.jk_sampling import sample_jk_indices
from flowbp.utils.checkpoint import save_checkpoint
from flowbp.utils.debug_utils import setup_debug
from flowbp.utils.ema_utils import save_sharded_ema_checkpoint
from flowbp.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
from flowbp.utils.hpsv2_transforms import HPSV2TransformsWithGrad
from flowbp.utils.logging_ import main_print
from flowbp.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
)

check_min_version("0.32.0")


def calculate_shift(
    image_seq_len,
    base_seq_len=256,
    max_seq_len=4096,
    base_shift=0.5,
    max_shift=1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _iter_sd35_loader(dataloader, device):
    while True:
        for data_item in dataloader:
            (
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                caption,
            ) = data_item
            yield (
                prompt_embeds.to(device),
                pooled_prompt_embeds.to(device),
                negative_prompt_embeds.to(device),
                negative_pooled_prompt_embeds.to(device),
                caption,
            )
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(dataloader.sampler.epoch + 1)


def _sample_initial_latents(args, transformer, device, batch_size):
    spatial_downsample = 8
    in_channels = int(getattr(transformer.module if isinstance(transformer, FSDP) else transformer, "config").in_channels)
    latent_h = args.h // spatial_downsample
    latent_w = args.w // spatial_downsample
    latents = torch.randn(
        (batch_size, in_channels, latent_h, latent_w),
        device=device,
        dtype=torch.bfloat16,
    )
    return latents, latent_h, latent_w


def _prepare_timesteps(args, fm_scheduler, transformer, latents, device):
    sample_steps = int(getattr(args, "rollout_steps", None) or args.sampling_steps or 25)
    sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)

    # SD3.5 official schedulers ship with use_dynamic_shifting=False (static shift=3.0).
    # Only pass `mu` when the scheduler actually uses dynamic shifting, mirroring the
    # diffusers SD3 pipeline so we do not silently feed an ignored kwarg.
    scheduler_kwargs = {}
    if fm_scheduler.config.get("use_dynamic_shifting", False):
        model = transformer.module if isinstance(transformer, FSDP) else transformer
        patch_size = int(getattr(model.config, "patch_size", 2))
        image_seq_len = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
        scheduler_kwargs["mu"] = calculate_shift(
            image_seq_len,
            fm_scheduler.config.get("base_image_seq_len", 256),
            fm_scheduler.config.get("max_image_seq_len", 4096),
            fm_scheduler.config.get("base_shift", 0.5),
            fm_scheduler.config.get("max_shift", 1.15),
        )

    timesteps, num_inference_steps = retrieve_timesteps(
        fm_scheduler,
        sample_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    assert num_inference_steps == sample_steps
    return timesteps


def _should_apply_cfg(args, negative_prompt_embeds, negative_pooled_prompt_embeds) -> bool:
    """Single source of truth for whether to enable CFG this step.

    Used by both ``_model_forward_sd35`` (forward graph shape) and
    ``train_sd35_one_step`` (backward loss compensation) so the two stay in
    perfect sync.
    """
    return (
        negative_prompt_embeds is not None
        and negative_pooled_prompt_embeds is not None
        and float(getattr(args, "cfg_guidance", 1.0)) > 1.0
    )


def _model_forward_sd35(
    args,
    transformer,
    latents,
    step_t,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
):
    """SD3.5 transformer forward with classifier-free guidance.

    Unlike Flux (guidance distillation, scalar ``guidance`` embedding) SD3.5
    uses real CFG. The default training path runs two separate forwards:
    conditional with autograd, negative under ``torch.no_grad()`` with detached
    latents. Configs can disable that detach/no_grad path via
    ``cfg_detach_neg: false``.

    The combined output ``f = pred_neg + g * (pred_cond - pred_neg)`` is
    numerically identical to the diffusers SD3.5 sampler. The default backward
    path divides the loss by ``g (= cfg_guidance)`` to keep optimizer step size
    roughly invariant to guidance scale; this can be disabled with
    ``cfg_grad_norm_compensate: false``.
    """
    batch_size = latents.shape[0]
    timestep = step_t.expand(batch_size).to(latents.dtype)

    with torch.autocast("cuda", torch.bfloat16):
        pred_cond = transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    if not _should_apply_cfg(args, negative_prompt_embeds, negative_pooled_prompt_embeds):
        return pred_cond

    detach_negative_branch = bool(getattr(args, "cfg_detach_neg", True))
    neg_latents = latents.detach() if detach_negative_branch else latents

    if detach_negative_branch:
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                pred_neg = transformer(
                    hidden_states=neg_latents,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    pooled_projections=negative_pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
    else:
        with torch.autocast("cuda", torch.bfloat16):
            pred_neg = transformer(
                hidden_states=neg_latents,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                pooled_projections=negative_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
    return pred_neg + float(args.cfg_guidance) * (pred_cond - pred_neg)


def _decode_sd35_latents(args, vae, latents):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(vae.config, "shift_factor", 0.0)
        latents = (latents / scaling_factor) + shift_factor
        image_pred_x0 = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
        image_pred_x0 = (image_pred_x0 * 0.5 + 0.5).clamp(0, 1)
    return image_pred_x0


def _maybe_decode_sd35_for_wandb(args, vae, latents, max_samples):
    if latents is None:
        return []
    n = min(int(max_samples), latents.shape[0])
    if n <= 0:
        return []
    images = _decode_sd35_latents(args, vae, latents[:n].to(vae.device))
    return [
        image.detach().float().cpu().permute(1, 2, 0).numpy()
        for image in images
    ]


def _maybe_log_connector_wandb(args, vae, prefix, payload):
    interval = int(getattr(args, "connector_wandb_interval", getattr(args, "connector_dump_interval", 0)) or 0)
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
        images = _maybe_decode_sd35_for_wandb(args, vae, payload.get(name), max_samples)
        if images:
            log_payload[f"connector/{prefix}_{name}"] = [
                wandb.Image(image, caption=f"{prefix} {name} sample {idx}")
                for idx, image in enumerate(images)
            ]
    if log_payload:
        wandb.log(log_payload, step=int(step))


def _maybe_save_connector_dump(args, prefix, payload):
    interval = int(getattr(args, "connector_dump_interval", 0) or 0)
    step = getattr(args, "current_train_step", None)
    if interval <= 0 or step is None or int(step) % interval != 0:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    dump_dir = getattr(args, "connector_dump_dir", None) or os.path.join(args.output_dir, "connector_dumps")
    os.makedirs(dump_dir, exist_ok=True)
    saved = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in payload.items()
    }
    torch.save(saved, os.path.join(dump_dir, f"{prefix}_step_{int(step):06d}.pt"))


def _optional_positive_float(value):
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _build_traj_filter_mask(args, d_j: torch.Tensor, d_0: torch.Tensor):
    """Return keep mask for per-sample trajectory filtering, or None if disabled."""
    dj_max = _optional_positive_float(getattr(args, "traj_filter_dj_max", None))
    d0_max = _optional_positive_float(getattr(args, "traj_filter_d0_max", None))
    if dj_max is None and d0_max is None:
        args._last_sample_keep_mask = None
        return None, {}

    keep_mask = torch.ones_like(d_j, dtype=torch.bool)
    if dj_max is not None:
        keep_mask = keep_mask & (d_j <= dj_max)
    if d0_max is not None:
        keep_mask = keep_mask & (d_0 <= d0_max)

    args._last_sample_keep_mask = keep_mask.detach()
    keep_rate = keep_mask.float().mean()
    return keep_mask, {
        "traj_filter_enabled": torch.tensor(1.0, device=d_j.device, dtype=torch.float32),
        "traj_filter_keep_rate": keep_rate.detach(),
        "traj_filter_drop_rate": (1.0 - keep_rate).detach(),
        "traj_filter_kept": keep_mask.float().sum().detach(),
    }


def sample_trajectory_sd35(
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
    latents_xt, _, _ = _sample_initial_latents(args, transformer, device, prompt_embeds.shape[0])
    timesteps = _prepare_timesteps(args, fm_scheduler, transformer, latents_xt, device)
    select_indices, k_idx, j_idx = sample_jk_indices(
        args, timesteps.size(0), generator
    )
    jk_truncated = bool(getattr(args, "_last_jk_truncated", False))
    grad_on_ind_set = set(
        timesteps.size(0) - select_index.item()
        for select_index in select_indices
    )

    x_j_reference = None
    pred_xt2 = None
    latents_pred_x0 = None
    for idx, step_t in enumerate(timesteps):
        with nullcontext() if (idx in grad_on_ind_set) else torch.no_grad():
            if idx == timesteps.size(0) - select_indices[1].item():
                with torch.no_grad():
                    pred_x2_raw = torch.abs(
                        pred_xt2.double() - latents_xt.double()
                    ).mean(
                        dim=tuple(range(1, pred_xt2.ndim)),
                        keepdim=False,
                    )
                    pred_x2_weight_factor = pred_x2_raw.clip(min=args.tau)
                # Match Flux LeapAlign: stop grads through the Euler branch while keeping the jump branch for theta.
                latents_xt = pred_xt2 + (latents_xt - pred_xt2).detach()
                if args.alpha <= 0.0:
                    dit_latents_xt = latents_xt.detach()
                else:
                    dit_latents_xt = (
                        args.alpha * latents_xt
                        + (1 - args.alpha) * latents_xt.detach()
                    )
            else:
                dit_latents_xt = latents_xt

            pred = _model_forward_sd35(
                args,
                transformer,
                dit_latents_xt,
                step_t,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            )

            if idx == timesteps.size(0) - select_indices[0].item():
                t2_idx = timesteps.size(0) - select_indices[1].item()
                pred_xt2 = fm_scheduler.jump_to_step(pred, step_t, latents_xt, t2_idx)
                pred = pred.detach()
            if idx == timesteps.size(0) - select_indices[1].item():
                x_j_reference = latents_xt.detach()
                latents_xt, latents_pred_x0 = fm_scheduler.step(
                    pred,
                    step_t,
                    latents_xt,
                    with_pred_x0=True,
                )
                latents_xt = latents_xt.detach()
            else:
                latents_xt = fm_scheduler.step(
                    pred,
                    step_t,
                    latents_xt,
                    with_pred_x0=False,
                )[0]
            if latents_xt.dtype != torch.bfloat16:
                latents_xt = latents_xt.to(torch.bfloat16)

    with torch.no_grad():
        pred_x0_raw = torch.abs(latents_pred_x0.double() - latents_xt.double()).mean(
            dim=tuple(range(1, latents_pred_x0.ndim)), keepdim=False
        )
        pred_x0_weight_factor = pred_x0_raw.clip(min=args.tau)
        traj_dist = pred_x2_weight_factor + pred_x0_weight_factor
        traj_sim_weight_factor = 1.0 / traj_dist
        traj_metrics = {
            "d0": pred_x0_raw.detach().mean(),
            "dj": pred_x2_raw.detach().mean(),
            "d0_clipped": pred_x0_weight_factor.detach().mean(),
            "dj_clipped": pred_x2_weight_factor.detach().mean(),
            "d_sum": traj_dist.detach().mean(),
            "w_sim": traj_sim_weight_factor.detach().mean(),
        }
    traj_metrics.update({
        "j_idx": torch.tensor(float(j_idx), device=latents_xt.device, dtype=torch.float32),
        "k_idx": torch.tensor(float(k_idx), device=latents_xt.device, dtype=torch.float32),
        "jk_gap": torch.tensor(float(j_idx - k_idx), device=latents_xt.device, dtype=torch.float32),
        "j_rev": torch.tensor(float(timesteps.size(0) - j_idx), device=latents_xt.device, dtype=torch.float32),
        "k_rev": torch.tensor(float(timesteps.size(0) - k_idx), device=latents_xt.device, dtype=torch.float32),
        "jk_truncated": torch.tensor(1.0 if jk_truncated else 0.0, device=latents_xt.device, dtype=torch.float32),
    })

    latents_pred_x0 = latents_pred_x0 + (latents_xt - latents_pred_x0).detach()
    if x_j_reference is not None:
        d_j_dump = torch.abs(pred_xt2.double() - x_j_reference.double()).mean(
            dim=tuple(range(1, pred_xt2.ndim)),
            keepdim=False,
        )
        d_0_dump = torch.abs(latents_pred_x0.double() - latents_xt.double()).mean(
            dim=tuple(range(1, latents_pred_x0.ndim)),
            keepdim=False,
        )
        _, filter_metrics = _build_traj_filter_mask(args, d_j_dump, d_0_dump)
        traj_metrics.update(filter_metrics)
        connector_payload = {
            "step": getattr(args, "current_train_step", None),
            "k_idx": timesteps.size(0) - select_indices[0].item(),
            "j_idx": timesteps.size(0) - select_indices[1].item(),
            "select_indices": select_indices.detach(),
            "xj_pred": pred_xt2.detach(),
            "xj": x_j_reference.detach(),
            "x0_pred": latents_pred_x0.detach(),
            "x0": latents_xt.detach(),
            "dj_per_sample": d_j_dump.detach(),
            "d0_per_sample": d_0_dump.detach(),
            "weight_scheme": "euler_jump",
        }
        _maybe_save_connector_dump(args, "sd35_leapalign", connector_payload)
        args._last_connector_payload = connector_payload
    return latents_xt, latents_pred_x0, traj_sim_weight_factor, traj_metrics


def _repeat_captions(caption, repeats: int):
    if repeats <= 1:
        return caption
    if isinstance(caption, str):
        return [caption] * repeats
    return list(caption) * repeats


def get_reward_loss_sd35(
    args,
    latents_pred_x0,
    sample_weights,
    sample_keep_mask,
    vae,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
):
    image_pred_x0 = _decode_sd35_latents(args, vae, latents_pred_x0)
    if args.use_hpsv2:
        image_pred_x0 = preprocess_val(image_pred_x0)
        text = tokenizer(caption).to(device=image_pred_x0.device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            outputs = reward_model(image_pred_x0, text)
            reward_pred_x0 = torch.einsum(
                "bc,bc->b",
                outputs["image_features"],
                outputs["text_features"],
            )
    else:
        raise NotImplementedError("Only HPSv2 is currently supported for reward computation.")
    raw_loss = F.relu(-reward_pred_x0 + args.loss_relu_clip) * args.loss_grad_scale
    if sample_weights is not None:
        sample_weights = sample_weights.to(raw_loss.device, raw_loss.dtype)
        raw_loss = sample_weights * raw_loss
    if sample_keep_mask is not None:
        sample_keep_mask = sample_keep_mask.to(device=raw_loss.device, dtype=torch.bool)
        if sample_keep_mask.any():
            raw_loss = raw_loss[sample_keep_mask]
        else:
            raw_loss = raw_loss.sum().view(()) * 0.0
    return torch.mean(raw_loss), reward_pred_x0.detach().mean()


def _align_grad_dtype_to_param_dtype(model):
    """AdamW expects each param's grad to match the param/state dtype.

    FSDP mixed precision can leave a few gradients in bf16 while the optimizer
    state for the corresponding flat parameter is fp32 (or vice versa). The
    single-tensor AdamW path then fails at exp_avg.lerp_(grad, ...). Casting
    after clipping keeps optimizer math well-typed without changing the forward
    graph.
    """
    for param in model.parameters():
        if param.grad is not None and param.grad.dtype != param.dtype:
            param.grad.data = param.grad.data.to(param.dtype)


def train_sd35_one_step(
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
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        caption,
    ) = next(loader)
    caption_for_reward = caption
    # Function-local imports keep the per-algorithm trainer modules decoupled
    # from this module at import time (they import shared backend helpers from
    # here), avoiding circular imports while still dispatching by trainer name.
    from flowbp.trainers.sd35.flowbp_bridge import sample_flowbp_bridge_trajectory_sd35
    from flowbp.trainers.sd35.flowbp_sparse import sample_flowbp_sparse_trajectory_sd35
    from flowbp.trainers.sd35.draft_lv import sample_draft_lv_trajectory_sd35
    from flowbp.trainers.sd35.drtune import sample_drtune_trajectory_sd35
    from flowbp.trainers.sd35.flowbp_lagrange import sample_flowbp_lagrange_trajectory_sd35
    from flowbp.trainers.sd35.refl import sample_refl_trajectory_sd35

    if args.trainer == "flowbp_lagrange":
        latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_lagrange_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        _maybe_log_connector_wandb(args, vae, "sd35_flowbp_lagrange", getattr(args, "_last_connector_payload", None))
    elif args.trainer == "draft_lv":
        _, latents_pred_x0, sample_weights, traj_metrics = sample_draft_lv_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        repeat_factor = latents_pred_x0.shape[0] // prompt_embeds.shape[0]
        caption_for_reward = _repeat_captions(caption, repeat_factor)
    elif args.trainer == "drtune":
        _, latents_pred_x0, sample_weights, traj_metrics = sample_drtune_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
    elif args.trainer == "refl":
        _, latents_pred_x0, sample_weights, traj_metrics = sample_refl_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
    elif args.trainer == "flowbp_sparse":
        latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_sparse_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        _maybe_log_connector_wandb(args, vae, "sd35_flowbp_sparse", getattr(args, "_last_connector_payload", None))
    elif args.trainer == "flowbp_bridge":
        latents_pred_x0, sample_weights, traj_metrics = sample_flowbp_bridge_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        _maybe_log_connector_wandb(args, vae, "sd35_flowbp_bridge", getattr(args, "_last_connector_payload", None))
    else:
        _, latents_pred_x0, sample_weights, traj_metrics = sample_trajectory_sd35(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            select_idx_generator,
        )
        _maybe_log_connector_wandb(args, vae, "sd35_leapalign", getattr(args, "_last_connector_payload", None))

    if args.trainer == "flowbp_lagrange":
        payload = getattr(args, "_last_connector_payload", None)
        current_step = getattr(args, "current_train_step", None)
        if payload is not None and current_step is not None and dist.get_rank() == 0:
            _jk_histogram_buffer["k_idx"].append(float(payload["k_idx"]))
            _jk_histogram_buffer["j_idx"].append(float(payload["j_idx"]))
            _jk_histogram_buffer["jk_gap"].append(float(payload["j_idx"] - payload["k_idx"]))
            total = float(getattr(args, "sampling_steps", 25) or 25)
            _jk_histogram_buffer.setdefault("seg_a", []).append(total - float(payload["j_idx"]))
            _jk_histogram_buffer.setdefault("seg_b", []).append(float(payload["j_idx"] - payload["k_idx"]))
            _jk_histogram_buffer.setdefault("seg_c", []).append(float(payload["k_idx"]))

    loss, avg_reward_pred_x0 = get_reward_loss_sd35(
        args,
        latents_pred_x0,
        sample_weights,
        getattr(args, "_last_sample_keep_mask", None),
        vae,
        reward_model,
        tokenizer,
        caption_for_reward,
        preprocess_val,
    )
    # Capture the unscaled loss before applying CFG gradient compensation, so
    # the metric remains comparable across guidance scales.
    avg_loss = loss.detach()

    # CFG gradient compensation. By default this cancels the linear gradient
    # amplification from CFG, so guidance changes mostly affect the forward
    # rollout rather than the optimizer step size.
    cfg_compensation = float(getattr(args, "cfg_guidance", 1.0))
    use_cfg_compensation = bool(getattr(args, "cfg_grad_norm_compensate", True))
    backward_divisor = float(args.gradient_accumulation_steps) * (
        cfg_compensation
        if (
            use_cfg_compensation
            and _should_apply_cfg(args, negative_prompt_embeds, negative_pooled_prompt_embeds)
        )
        else 1.0
    )
    (loss / backward_divisor).backward()
    dist.all_reduce(avg_loss.div_(args.gradient_accumulation_steps), op=dist.ReduceOp.AVG)
    dist.all_reduce(avg_reward_pred_x0.div_(args.gradient_accumulation_steps), op=dist.ReduceOp.AVG)
    reduced_traj_metrics = {}
    for metric_name, metric_value in traj_metrics.items():
        if metric_value is None:
            reduced_traj_metrics[metric_name] = None
            continue
        metric_tensor = metric_value.detach().float().to(device)
        dist.all_reduce(metric_tensor.div_(args.gradient_accumulation_steps), op=dist.ReduceOp.AVG)
        reduced_traj_metrics[metric_name] = metric_tensor.item()

    if (inner_step + 1) % args.gradient_accumulation_steps == 0:
        grad_norm = transformer.clip_grad_norm_(args.max_grad_norm)
        _align_grad_dtype_to_param_dtype(transformer)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        grad_norm = grad_norm.item()
    else:
        grad_norm = None
    return avg_loss.item(), avg_reward_pred_x0.item(), grad_norm, reduced_traj_metrics


def _init_hpsv2(args, device):
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        "./hps_ckpt/open_clip_pytorch_model.bin",
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    preprocess_val = HPSV2TransformsWithGrad(preprocess_val)
    checkpoint = torch.load("./hps_ckpt/HPS_v2.1_compressed.pt", map_location=f"cuda:{device}")
    model.load_state_dict(checkpoint["state_dict"])
    processor = get_tokenizer("ViT-H-14")
    reward_model = model.to(device)
    reward_model.requires_grad_(False)
    reward_model.eval()
    return reward_model, processor, preprocess_val



def normalize_sd35_trainer_name(name: str | None) -> str:
    raw_name = str(name or "leapalign").lower().replace("-", "_")
    aliases = {
        "flowbp_sparse": "flowbp_sparse",
        "flowbp_bridge": "flowbp_bridge",
        "flowbp_lagrange": "flowbp_lagrange",
    }
    return aliases.get(raw_name, raw_name)

def run_training_sd35(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    trainer_name = normalize_sd35_trainer_name(getattr(args, "trainer", "leapalign"))
    args.trainer = trainer_name
    if trainer_name not in {
        "leapalign",
        "refl",
        "flowbp_lagrange",
        "flowbp_sparse",
        "flowbp_bridge",
        "draft_lv",
        "drtune",
    }:
        raise ValueError(
            "SD3.5 trainer supports FlowBP method names (flowbp_sparse, "
            "flowbp_bridge, flowbp_lagrange) and baselines "
            "(leapalign, refl, draft_lv, drtune)."
        )
    if int(getattr(args, "sp_size", 1)) != 1:
        raise NotImplementedError("SD3.5 trainer currently supports sp_size=1 only.")
    if args.h is None or args.w is None or args.h % 16 != 0 or args.w % 16 != 0:
        raise ValueError("SD3.5 training expects h/w to be set and divisible by 16.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if args.debug:
        setup_debug()
    main_print("------------------args------------------")
    for arg_k, arg_v in vars(args).items():
        main_print(f"{arg_k}: {arg_v}")
    main_print("------------------args------------------")
    if float(getattr(args, "cfg", 0.0)) > 0.0 and float(getattr(args, "cfg_guidance", 1.0)) > 1.0:
        main_print(
            "[warning] SD3.5 RL training usually wants data.cfg=0.0. "
            "With cfg>0, dropped samples become unconditional under CFG."
        )
    if args.seed is not None:
        set_seed(args.seed + rank)
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if not args.use_hpsv2:
        raise NotImplementedError("Only HPSv2 is currently supported as the reward model.")
    reward_model, processor, preprocess_val = _init_hpsv2(args, device)

    main_print(f"--> loading SD3 model from {args.pretrained_model_name_or_path}")
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    fm_scheduler = FlowMatchEulerDiscreteWithPredX0Scheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    ema_model = deepcopy(transformer) if args.use_ema else None
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    transformer = FSDP(transformer, **fsdp_kwargs)
    if args.use_ema and ema_model is not None:
        ema_model = FSDP(ema_model, **fsdp_kwargs)
        ema_model.eval()
        ema_model.requires_grad_(False)
    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules, args.selective_checkpointing)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    transformer.train()

    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
        fused=False,
        foreach=False,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=-1,
    )

    train_dataset = LatentSD35Dataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_sd35_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    select_idx_generator = (
        torch.Generator(device="cpu").manual_seed(args.select_idx_seed)
        if args.select_idx_seed is not None
        else None
    )
    if rank <= 0:
        wandb.init(project=args.project, config=args, name=args.run_name)

    total_batch_size = world_size * args.gradient_accumulation_steps * args.train_batch_size
    main_print("***** Running SD3.5 training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size = {total_batch_size}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(f"  Master weight dtype: {next(transformer.parameters()).dtype}")

    loader = _iter_sd35_loader(train_dataloader, device)
    progress_bar = tqdm(range(1, args.max_train_steps + 1), desc="Steps", disable=local_rank > 0)
    step_times = deque(maxlen=100)

    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for step in range(1, args.max_train_steps + 1):
            args.current_train_step = step
            start_time = time.time()
            total_loss = 0.0
            total_reward_pred_x0 = 0.0
            grad_norm = None
            traj_metrics = {}
            for inner_step in range(args.gradient_accumulation_steps):
                avg_loss, avg_reward_pred_x0, grad_norm, traj_metrics = train_sd35_one_step(
                    args,
                    inner_step,
                    device,
                    transformer,
                    vae,
                    fm_scheduler,
                    reward_model,
                    processor,
                    optimizer,
                    lr_scheduler,
                    loader,
                    preprocess_val,
                    select_idx_generator,
                )
                total_loss += avg_loss
                total_reward_pred_x0 += avg_reward_pred_x0

            if args.use_ema and ema_model is not None:
                for tgt, src in zip(ema_model.parameters(), transformer.parameters()):
                    tgt.data.lerp_(src.data.to(tgt), 1 - args.ema_decay)
            if step % args.checkpointing_steps == 0:
                save_checkpoint(transformer, rank, args.output_dir, step, epoch)
                if args.use_ema:
                    save_sharded_ema_checkpoint(ema_model, rank, args.output_dir, step, epoch)
                dist.barrier()

            step_time = time.time() - start_time
            step_times.append(step_time)
            progress_bar.set_postfix({"loss": f"{total_loss:.4f}", "step_time": f"{step_time:.2f}s", "grad_norm": grad_norm})
            if rank <= 0:
                log_dict = {
                    "train_loss": total_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": sum(step_times) / len(step_times),
                    "grad_norm": grad_norm,
                    "reward": total_reward_pred_x0,
                }
                log_dict.update({f"train/{name}": value for name, value in traj_metrics.items() if value is not None})
                wandb.log(log_dict, step=step)
            progress_bar.update(1)

            if step % args.evaluation_interval == 0 or step == 1:
                transformer.eval()
                run_online_sd35_eval(
                    step,
                    args,
                    ema_model if args.use_ema else transformer,
                    vae,
                    rank,
                    world_size,
                    device,
                )
                transformer.train()

    if trainer_name == "flowbp_lagrange" and rank <= 0:
        csv_path = save_jk_sampling_csv(args.output_dir)
        if csv_path:
            print(f"[jk_sampling] Saved: {csv_path}")
    progress_bar.close()
    if dist.is_initialized():
        dist.barrier()
    if rank <= 0:
        wandb.finish()
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()
    elif dist.is_initialized():
        dist.destroy_process_group()


class LeapAlignSD35Trainer:
    def __init__(self, config):
        self.args = config

    def train(self):
        return run_training_sd35(self.args)
