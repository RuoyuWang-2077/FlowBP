
"""Deep Reward Tuning trainer for FLUX.2.

Implements "Deep Reward Supervisions for Tuning Text-to-Image Diffusion
Models" (DRTune) on the FLUX.2 pipeline: Qwen3 prompt embeddings, packed
32-channel VAE latents, 4D position ids, and explicit CFG via
``_cfg_aware_forward``.
"""

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


DEFAULT_DRTUNE_NUM_TRAIN_STEPS = 3
DEFAULT_DRTUNE_EARLY_STOP_RATIO = 0.4


def _sample_equally_spaced_train_indices(
    total_steps: int,
    num_train_steps: int,
    generator: torch.Generator | None,
) -> list[int]:
    """Sample DRTune's equally spaced training-step subset."""
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


def sample_drtune_trajectory_flux2(
    args,
    device,
    transformer,
    fm_scheduler,
    prompt_embeds,
    text_ids,
    generator,
):
    """Sample a FLUX.2 DRTune trajectory and return an x0 prediction."""
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

    early_stop_idx = _sample_early_stop_idx(args, timesteps.size(0), generator)
    num_train_steps = int(
        getattr(args, "drtune_num_train_steps", DEFAULT_DRTUNE_NUM_TRAIN_STEPS)
    )
    # Sample inside the executed prefix so every update has a trainable branch.
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
            pred = _cfg_aware_forward(
                args,
                transformer,
                model_input,
                timestep,
                prompt_embeds,
                text_ids,
                img_ids,
            )
            executed_train_indices.append(idx)
        else:
            with torch.no_grad():
                pred = _cfg_aware_forward(
                    args,
                    transformer,
                    model_input,
                    timestep,
                    prompt_embeds,
                    text_ids,
                    img_ids,
                )
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
        raise RuntimeError("DRTune FLUX.2 rollout ended without x0 prediction.")

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
    connector_payload = {
        "step": getattr(args, "current_train_step", None),
        "train_indices": torch.tensor(train_indices),
        "executed_train_indices": torch.tensor(executed_train_indices),
        "early_stop_idx": int(early_stop_idx),
        "connector_scheme": "drtune_stop_gradient_flux2",
    }
    _maybe_save_connector_dump(args, "drtune_flux2", connector_payload)
    args._last_connector_payload = connector_payload
    return latents_xt, latents_pred_x0, None, traj_metrics


def train_drtune_one_step_flux2(
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
        sample_drtune_trajectory_flux2(
            args,
            device,
            transformer,
            fm_scheduler,
            prompt_embeds,
            text_ids,
            select_idx_generator,
        )
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


class LeapAlignDRTuneFlux2Trainer(LeapAlignFlux2Trainer):
    """DRTune trainer wrapper for FLUX.2."""

    def train(self):
        self.args.trainer = "drtune"
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
        return train_drtune_one_step_flux2(
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
