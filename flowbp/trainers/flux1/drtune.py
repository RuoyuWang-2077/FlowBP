
"""Deep Reward Tuning trainer for FLUX.

Implements "Deep Reward Supervisions for Tuning Text-to-Image Diffusion
Models" (DRTune): detach the denoising-network input, train an equally spaced
subset of sampling steps, and supervise a final/intermediate x0 prediction with
the differentiable reward.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from flowbp.trainers.flux1.leapalign import (
    LeapAlignFluxTrainer,
    get_reward_loss,
    pack_latents,
    prepare_latent_image_ids,
    run_training,
)


DEFAULT_DRTUNE_NUM_TRAIN_STEPS = 3
DEFAULT_DRTUNE_EARLY_STOP_RATIO = 0.4


def _sample_equally_spaced_train_indices(
    total_steps: int,
    num_train_steps: int,
    generator: torch.Generator | None,
) -> list[int]:
    """Sample DRTune's equally spaced training-step subset.

    The returned indices use this codebase's denoising-loop convention:
    ``0`` is the first/noisiest step, ``total_steps - 1`` is closest to x0.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if num_train_steps <= 0:
        raise ValueError(
            f"drtune_num_train_steps must be positive, got {num_train_steps}"
        )
    num_train_steps = min(int(num_train_steps), int(total_steps))
    if num_train_steps == total_steps:
        return list(range(total_steps))

    stride = max(1, total_steps // num_train_steps)
    max_start = total_steps - 1 - (num_train_steps - 1) * stride
    if max_start < 0:
        stride = max(1, (total_steps - 1) // max(1, num_train_steps - 1))
        max_start = total_steps - 1 - (num_train_steps - 1) * stride
    max_start = max(0, int(max_start))

    start = torch.randint(
        0,
        max_start + 1,
        (1,),
        device="cpu",
        generator=generator,
    ).long()[0].item()
    return [int(start + i * stride) for i in range(num_train_steps)]


def _sample_early_stop_idx(
    args,
    total_steps: int,
    generator: torch.Generator | None,
) -> int:
    """Return the denoising-loop index where x0 supervision is taken."""
    early_stop_steps = getattr(args, "drtune_early_stop_steps", None)
    if early_stop_steps is None:
        ratio = float(
            getattr(args, "drtune_early_stop_ratio", DEFAULT_DRTUNE_EARLY_STOP_RATIO)
        )
        early_stop_steps = int(round(total_steps * ratio))
    early_stop_steps = int(early_stop_steps)

    # m <= 0 disables early stop and supervises the final denoising step.
    if early_stop_steps <= 0:
        return total_steps - 1
    early_stop_steps = min(early_stop_steps, total_steps)
    reverse_index = torch.randint(
        1,
        early_stop_steps + 1,
        (1,),
        device="cpu",
        generator=generator,
    ).long()[0].item()
    return total_steps - int(reverse_index)


def sample_drtune_trajectory(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    generator,
):
    """Sample a DRTune trajectory and return an x0 prediction with gradients."""
    w, h = args.w, args.h
    sample_steps = int(
        getattr(args, "rollout_steps", None)
        or getattr(args, "sampling_steps", None)
        or 25
    )
    if sample_steps <= 0:
        raise ValueError(f"rollout/sampling steps must be positive, got {sample_steps}")

    batch_size = encoder_hidden_states.shape[0]
    spatial_downsample = 8
    in_channels = 16
    latent_w, latent_h = w // spatial_downsample, h // spatial_downsample

    latents_xt = torch.randn(
        (batch_size, in_channels, latent_h, latent_w),
        device=device,
        dtype=torch.bfloat16,
    )
    latents_xt = pack_latents(
        latents_xt,
        batch_size,
        in_channels,
        latent_h,
        latent_w,
    )
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

    early_stop_idx = _sample_early_stop_idx(args, timesteps.size(0), generator)
    num_train_steps = int(
        getattr(args, "drtune_num_train_steps", DEFAULT_DRTUNE_NUM_TRAIN_STEPS)
    )
    # Early stop shortens the supervised trajectory for this iteration. Sample
    # inside the executed prefix so every update carries at least one trainable
    # DRTune branch.
    train_indices = _sample_equally_spaced_train_indices(
        total_steps=early_stop_idx + 1,
        num_train_steps=num_train_steps,
        generator=generator,
    )
    train_index_set = set(train_indices)
    executed_train_indices: list[int] = []
    latents_pred_x0 = None

    for idx, step_t in enumerate(timesteps):
        timestep = step_t.expand(latents_xt.shape[0]).to(latents_xt.dtype)
        model_input = latents_xt.detach()
        is_train_step = idx in train_index_set
        if is_train_step:
            with torch.autocast("cuda", torch.bfloat16):
                pred = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep / 1000,
                    guidance=cfg_guidance,
                    txt_ids=text_ids,
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            executed_train_indices.append(idx)
        else:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                pred = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep / 1000,
                    guidance=cfg_guidance,
                    txt_ids=text_ids,
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            pred = pred.detach()

        if idx == early_stop_idx:
            _, latents_pred_x0 = fm_scheduler.step(
                pred,
                step_t,
                latents_xt,
                with_pred_x0=True,
            )
            break

        latents_xt = fm_scheduler.step(
            pred,
            step_t,
            latents_xt,
            with_pred_x0=False,
        )[0]
        if latents_xt.dtype != torch.bfloat16:
            latents_xt = latents_xt.to(torch.bfloat16)

    if latents_pred_x0 is None:
        raise RuntimeError("DRTune rollout ended without producing an x0 prediction.")

    traj_metrics = {
        "d0": None,
        "dj": None,
        "w_sim": None,
        "drtune_num_train_steps": torch.tensor(
            len(train_indices), device=device, dtype=torch.float32
        ),
        "drtune_num_executed_train_steps": torch.tensor(
            len(executed_train_indices), device=device, dtype=torch.float32
        ),
        "drtune_early_stop_idx": torch.tensor(
            early_stop_idx, device=device, dtype=torch.float32
        ),
    }
    args._last_connector_payload = {
        "step": getattr(args, "current_train_step", None),
        "train_indices": torch.tensor(train_indices),
        "executed_train_indices": torch.tensor(executed_train_indices),
        "early_stop_idx": int(early_stop_idx),
        "connector_scheme": "drtune_stop_gradient",
    }
    return latents_xt, latents_pred_x0, None, traj_metrics


def train_drtune_one_step(
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

    _, latents_pred_x0, traj_sim_weight_factor, traj_metrics = sample_drtune_trajectory(
        args,
        device,
        transformer,
        fm_scheduler,
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        select_idx_generator,
    )
    loss, avg_reward_pred_x0 = get_reward_loss(
        args,
        latents_pred_x0,
        vae,
        reward_model,
        tokenizer,
        caption,
        preprocess_val,
        traj_sim_weight_factor,
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


class LeapAlignDRTuneFluxTrainer(LeapAlignFluxTrainer):
    """DRTune trainer wrapper for FLUX."""

    def train(self):
        self.args.trainer = "drtune"
        return run_training(self.args)

    def sample_trajectory(
        self,
        device,
        transformer,
        fm_scheduler,
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        generator,
    ):
        return sample_drtune_trajectory(
            self.args,
            device,
            transformer,
            fm_scheduler,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            generator,
        )

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
        return train_drtune_one_step(
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
