
"""DRaFT-LV trainer for FLUX.2."""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from flowbp.trainers.flux2.leapalign import (
    FLUX2_PATCH_CHANNELS,
    FLUX2_TOTAL_DOWNSAMPLE,
    LeapAlignFlux2Trainer,
    _cfg_aware_forward,
    _maybe_save_connector_dump,
    compute_empirical_mu,
    encode_or_load_prompts,
    get_reward_loss,
    pack_latents,
    prepare_latent_ids,
    run_training,
)


DEFAULT_DRAFT_LV_NUM_NOISED_SAMPLES = 2


def _repeat_captions(caption, repeats: int):
    if repeats <= 1:
        return caption
    if isinstance(caption, str):
        return [caption] * repeats
    return list(caption) * repeats


def sample_draft_lv_trajectory_flux2(
    args,
    device,
    transformer,
    fm_scheduler,
    prompt_embeds,
    text_ids,
    generator,
):
    """Sample full trajectory, then average last-step reward gradients."""
    h, w = args.h, args.w
    sample_steps = int(
        getattr(args, "rollout_steps", None)
        or getattr(args, "sampling_steps", None)
        or 25
    )
    if sample_steps <= 0:
        raise ValueError(f"rollout/sampling steps must be positive, got {sample_steps}")
    if h % FLUX2_TOTAL_DOWNSAMPLE or w % FLUX2_TOTAL_DOWNSAMPLE:
        raise ValueError(
            f"FLUX.2 requires h, w divisible by {FLUX2_TOTAL_DOWNSAMPLE}; "
            f"got h={h}, w={w}"
        )

    batch_size = prompt_embeds.shape[0]
    h_tokens = h // FLUX2_TOTAL_DOWNSAMPLE
    w_tokens = w // FLUX2_TOTAL_DOWNSAMPLE

    latents_xt = torch.randn(
        (batch_size, FLUX2_PATCH_CHANNELS, h_tokens, w_tokens),
        device=device,
        dtype=torch.bfloat16,
    )
    latents_xt = pack_latents(latents_xt)
    img_ids = prepare_latent_ids(
        batch_size,
        h_tokens,
        w_tokens,
        device,
        torch.int64,
    )

    sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
    image_seq_len = latents_xt.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=sample_steps)
    fm_scheduler.set_timesteps(sample_steps, device=device, sigmas=sigmas, mu=mu)
    timesteps = fm_scheduler.timesteps
    assert timesteps.shape[0] == sample_steps

    final_idx = timesteps.size(0) - 1
    with torch.no_grad():
        for idx in range(final_idx):
            step_t = timesteps[idx]
            timestep = step_t.expand(latents_xt.shape[0]).to(latents_xt.dtype)
            pred = _cfg_aware_forward(
                args,
                transformer,
                latents_xt,
                timestep,
                prompt_embeds,
                text_ids,
                img_ids,
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
    timestep = final_step_t.expand(latents_xt.shape[0]).to(latents_xt.dtype)
    latents_last = latents_xt.detach()
    pred = _cfg_aware_forward(
        args,
        transformer,
        latents_last,
        timestep,
        prompt_embeds,
        text_ids,
        img_ids,
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
        pred_lv = _cfg_aware_forward(
            args,
            transformer,
            x_last,
            timestep,
            prompt_embeds,
            text_ids,
            img_ids,
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
    connector_payload = {
        "step": getattr(args, "current_train_step", None),
        "final_idx": int(final_idx),
        "num_noised_samples": int(num_noised),
        "sigma_last": float(sigma_last.item()),
        "connector_scheme": "draft_lv_last_step_flux2",
    }
    _maybe_save_connector_dump(args, "draft_lv_flux2", connector_payload)
    args._last_connector_payload = connector_payload
    return latents_last, latents_pred_x0, None, traj_metrics


def train_draft_lv_one_step_flux2(
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

    _, latents_pred_x0, traj_sim_weight_factor, traj_metrics = (
        sample_draft_lv_trajectory_flux2(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            text_ids,
            select_idx_generator,
        )
    )
    repeat_factor = latents_pred_x0.shape[0] // prompt_embeds.shape[0]
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


class LeapAlignDRaFTLVFlux2Trainer(LeapAlignFlux2Trainer):
    """DRaFT-LV trainer wrapper for FLUX.2."""

    def train(self):
        self.args.trainer = "draft_lv"
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
        return train_draft_lv_one_step_flux2(
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
