
"""Standard ReFL trainer for FLUX.2 (Xu et al., NeurIPS 2023).

Mirrors :mod:`flowbp.trainers.flux1.refl` but targets the FLUX.2
architecture:

* Qwen3 text encoder (online or precomputed embeddings)
* 32-channel VAE latents with a 2x2 patchify + BatchNorm-based normalization
* 4D positional ids ``(T, H, W, L)`` for image and text tokens
* No pooled projections; CFG handled explicitly via ``_cfg_aware_forward``
  for undistilled klein-base checkpoints.

Algorithm (standard ReFL):
  1. Roll out the diffusion from x_T with NO gradient until a randomly chosen
     step t in the last ``refl_last_n_steps`` of the rollout.
  2. Run the transformer once at step t WITH gradient enabled.
  3. Predict x_0 in a single Euler step:  x0_hat = x_t - sigma_t * v_theta.
  4. Decode x0_hat through the VAE and back-prop the reward loss.

No LeapAlign-specific weighting (no traj_sim_weight_factor, no LeapAlign
reparameterization, no nested_grad mixing) is involved.
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
    compute_empirical_mu,
    encode_or_load_prompts,
    get_reward_loss,
    pack_latents,
    prepare_latent_ids,
    run_training,
)

# ---------------------------------------------------------------------------
# ReFL defaults (kept in sync with refl_flux_trainer.py)
# ---------------------------------------------------------------------------
# Standard ReFL fine-tunes ONE randomly chosen denoising step from the tail of
# a fixed-length rollout. The defaults below realize the recipe:
#     * Total rollout length: 25 denoising steps  (DEFAULT_REFL_ROLLOUT_STEPS)
#     * Gradient step is sampled uniformly from the LAST 11 of those 25 steps
#       (DEFAULT_REFL_LAST_N_STEPS), i.e. indices [14, 24] inclusive.
DEFAULT_REFL_ROLLOUT_STEPS = 25
DEFAULT_REFL_LAST_N_STEPS = 11


def sample_refl_trajectory_flux2(
    args,
    device,
    transformer,
    fm_scheduler,
    prompt_embeds,
    text_ids,
    generator,
):
    """Standard ReFL trajectory sampling for FLUX.2.

    Default behavior: roll out 25 denoising steps and back-prop through one
    randomly chosen step from the last 11 (i.e. fine-tune the tail 11/25).
    Both knobs are configurable via ``args.rollout_steps`` and
    ``args.refl_last_n_steps``.

    Returns ``(latents_xt, latents_pred_x0, None, traj_metrics)`` where
    ``traj_metrics`` is a dict of ``None`` placeholders kept for signature
    compatibility with the LeapAlign step's metric reducer.
    """
    h, w = args.h, args.w
    sample_steps = int(
        getattr(args, "rollout_steps", None)
        or getattr(args, "sampling_steps", None)
        or DEFAULT_REFL_ROLLOUT_STEPS
    )
    last_n_steps = int(
        getattr(args, "refl_last_n_steps", None) or DEFAULT_REFL_LAST_N_STEPS
    )
    if last_n_steps < 1:
        raise ValueError(f"refl_last_n_steps must be >= 1, got {last_n_steps}")
    if last_n_steps > sample_steps:
        raise ValueError(
            f"refl_last_n_steps={last_n_steps} cannot exceed "
            f"rollout/sampling steps={sample_steps}"
        )

    # selected_index follows the LeapAlign reverse-index convention:
    # 1 = the final denoising step, n = the n-th step before the end.
    selected_index = torch.randint(
        1,
        last_n_steps + 1,
        (1,),
        device="cpu",
        generator=generator,
    ).long()[0]

    batch_size = prompt_embeds.shape[0]
    h_tokens = h // FLUX2_TOTAL_DOWNSAMPLE
    w_tokens = w // FLUX2_TOTAL_DOWNSAMPLE
    if h % FLUX2_TOTAL_DOWNSAMPLE or w % FLUX2_TOTAL_DOWNSAMPLE:
        raise ValueError(
            f"FLUX.2 requires h, w divisible by {FLUX2_TOTAL_DOWNSAMPLE}; "
            f"got h={h}, w={w}"
        )

    latents_xt = torch.randn(
        (batch_size, FLUX2_PATCH_CHANNELS, h_tokens, w_tokens),
        device=device,
        dtype=torch.bfloat16,
    )
    latents_xt = pack_latents(latents_xt)  # (B, n_tokens, 128)

    # int64 image ids to match the inference pipeline; bf16 would silently
    # quantize positions >= 256 inside Flux2PosEmbed. See the matching comment
    # in leapalign_flux2_trainer.sample_trajectory.
    img_ids = prepare_latent_ids(
        batch_size, h_tokens, w_tokens, device, torch.int64,
    )

    sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
    image_seq_len = latents_xt.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=sample_steps)
    fm_scheduler.set_timesteps(sample_steps, device=device, sigmas=sigmas, mu=mu)
    timesteps = fm_scheduler.timesteps
    assert timesteps.shape[0] == sample_steps

    grad_idx = timesteps.size(0) - selected_index.item()

    # Phase 1: no-grad rollout from x_T up to (but not including) step grad_idx.
    # We rely on `_cfg_aware_forward` to apply CFG correctly when
    # args.cfg_guidance > 1 (the negative branch is detached internally).
    with torch.no_grad():
        for idx in range(grad_idx):
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

    # Phase 2: single gradient-enabled transformer call at step grad_idx.
    # latents_xt at this point is a no-grad tensor; the only gradient path
    # is through this transformer call.
    latents_xt = latents_xt.detach()
    step_t = timesteps[grad_idx]
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

    # One-step x_0 prediction:  x0_hat = x_t - sigma_t * v_theta(x_t, t).
    _, latents_pred_x0 = fm_scheduler.step(
        pred,
        step_t,
        latents_xt,
        with_pred_x0=True,
    )

    return latents_xt, latents_pred_x0, None, {
        "d0": None,
        "dj": None,
        "w_sim": None,
    }


def train_refl_one_step_flux2(
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
    """Single ReFL training step on FLUX.2.

    Loss path:  prompt_embeds -> sample_refl_trajectory_flux2 -> VAE decode
    -> HPSv2 reward -> ReLU(-r + clip) * scale -> backward.

    The backward is divided by ``cfg_guidance`` when CFG is active so that
    grad_norm / max_grad_norm thresholds stay comparable across cfg values
    (the detached negative branch in ``_cfg_aware_forward`` would otherwise
    inflate the gradient by exactly ``cfg_guidance``).
    """
    batch = next(loader)
    prompt_embeds, text_ids, caption = encode_or_load_prompts(
        args, batch, device, text_encoder, text_tokenizer,
    )

    _, latents_pred_x0, traj_sim_weight_factor, traj_metrics = (
        sample_refl_trajectory_flux2(
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
        traj_sim_weight_factor,  # always None for standard ReFL
    )

    # Capture the un-scaled loss for wandb display before applying the CFG
    # gradient compensation below.
    avg_loss = loss.detach()

    cfg_scale = float(getattr(args, "cfg_guidance", 1.0))
    cfg_compensate = bool(getattr(args, "cfg_grad_norm_compensate", True))
    if cfg_scale > 1.0 and cfg_compensate:
        backward_loss = loss / cfg_scale
    else:
        backward_loss = loss
    backward_loss.backward()

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


class LeapAlignReFLFlux2Trainer(LeapAlignFlux2Trainer):
    """Standard ReFL trainer for FLUX.2.

    By default fine-tunes one random denoising step out of the last
    ``DEFAULT_REFL_LAST_N_STEPS`` (= 11) steps of a
    ``DEFAULT_REFL_ROLLOUT_STEPS`` (= 25)-step rollout. Override via
    ``args.rollout_steps`` / ``args.refl_last_n_steps`` (CLI flags
    ``--rollout_steps`` / ``--refl_last_n_steps`` or YAML keys
    ``sampling.rollout_steps`` / ``refl.last_n_steps``).
    """

    def train(self):
        self.args.trainer = "refl"
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
        return train_refl_one_step_flux2(
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
