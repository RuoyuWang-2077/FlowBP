
"""DRaFT-LV trainer for FLUX.

Implements the low-variance K=1 variant from "Directly Fine-Tuning Diffusion
Models on Differentiable Rewards": back-prop through the last denoising step,
then noise the generated x0 back to the last-step sigma multiple times and
average the reward gradients over those extra last-step examples.
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


DEFAULT_DRAFT_LV_NUM_NOISED_SAMPLES = 2


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


def _repeat_captions(caption, repeats: int):
    if repeats <= 1:
        return caption
    if isinstance(caption, str):
        return [caption] * repeats
    return list(caption) * repeats


def sample_draft_lv_trajectory(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    generator,
):
    """Return stacked x0 predictions for base + low-variance last-step samples."""
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

    final_idx = timesteps.size(0) - 1
    with torch.no_grad():
        for idx in range(final_idx):
            step_t = timesteps[idx]
            pred = _model_forward(
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
            latents_xt = fm_scheduler.step(
                pred,
                step_t,
                latents_xt,
                with_pred_x0=False,
            )[0]
            if latents_xt.dtype != torch.bfloat16:
                latents_xt = latents_xt.to(torch.bfloat16)

    final_step_t = timesteps[final_idx]
    latents_last = latents_xt.detach()
    pred = _model_forward(
        args,
        transformer,
        latents_last,
        final_step_t,
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        image_ids,
        cfg_guidance,
    )
    _, base_x0 = fm_scheduler.step(
        pred,
        final_step_t,
        latents_last,
        with_pred_x0=True,
    )

    num_noised = int(
        getattr(args, "draft_lv_num_noised_samples", DEFAULT_DRAFT_LV_NUM_NOISED_SAMPLES)
    )
    if num_noised < 0:
        raise ValueError(
            f"draft_lv_num_noised_samples must be >= 0, got {num_noised}"
        )

    x0_predictions = [base_x0]
    sigma_last = fm_scheduler.sigmas[final_idx].to(
        device=base_x0.device, dtype=torch.float32,
    )
    alpha_last = 1.0 - sigma_last
    for _ in range(num_noised):
        eps = torch.randn(
            base_x0.shape,
            device=base_x0.device,
            dtype=torch.float32,
        )
        x_last = (alpha_last * base_x0.detach().float() + sigma_last * eps).to(
            torch.bfloat16
        )
        pred_lv = _model_forward(
            args,
            transformer,
            x_last,
            final_step_t,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            image_ids,
            cfg_guidance,
        )
        x0_lv = x_last.float() - sigma_last * pred_lv.float()
        x0_predictions.append(x0_lv)

    latents_pred_x0 = torch.cat(x0_predictions, dim=0)
    traj_metrics = {
        "d0": None,
        "dj": None,
        "w_sim": None,
        "draft_lv_num_noised_samples": torch.tensor(
            float(num_noised), device=device, dtype=torch.float32,
        ),
        "draft_lv_total_reward_examples": torch.tensor(
            float(len(x0_predictions)), device=device, dtype=torch.float32,
        ),
        "draft_lv_sigma_last": sigma_last.detach().float(),
    }
    args._last_connector_payload = {
        "step": getattr(args, "current_train_step", None),
        "final_idx": int(final_idx),
        "num_noised_samples": int(num_noised),
        "sigma_last": float(sigma_last.item()),
        "connector_scheme": "draft_lv_last_step",
    }
    return latents_last, latents_pred_x0, None, traj_metrics


def train_draft_lv_one_step(
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

    _, latents_pred_x0, traj_sim_weight_factor, traj_metrics = (
        sample_draft_lv_trajectory(
            args,
            device,
            transformer,
            fm_scheduler,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            select_idx_generator,
        )
    )
    repeat_factor = latents_pred_x0.shape[0] // encoder_hidden_states.shape[0]
    loss, avg_reward_pred_x0 = get_reward_loss(
        args,
        latents_pred_x0,
        vae,
        reward_model,
        tokenizer,
        _repeat_captions(caption, repeat_factor),
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


class LeapAlignDRaFTLVFluxTrainer(LeapAlignFluxTrainer):
    """DRaFT-LV trainer wrapper for FLUX."""

    def train(self):
        self.args.trainer = "draft_lv"
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
        return sample_draft_lv_trajectory(
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
        return train_draft_lv_one_step(
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
