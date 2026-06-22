
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

# ---------------------------------------------------------------------------
# ReFL defaults
# ---------------------------------------------------------------------------
# Standard ReFL fine-tunes ONE randomly chosen denoising step from the tail of
# a fixed-length rollout. The defaults below realize the recipe:
#     * Total rollout length: 25 denoising steps  (DEFAULT_REFL_ROLLOUT_STEPS)
#     * Gradient step is sampled uniformly from the LAST 11 of those 25 steps
#       (DEFAULT_REFL_LAST_N_STEPS), i.e. indices [14, 24] inclusive.
# These values match the CLI defaults in ``train_flowbp_flux.py``
# (``--rollout_steps 25``, ``--refl_last_n_steps 11``) and the YAML config
# ``configs/finetune/flux_refl_hpdv2_hpsv2_eval5.yaml``.
DEFAULT_REFL_ROLLOUT_STEPS = 25
DEFAULT_REFL_LAST_N_STEPS = 11


def sample_refl_trajectory(
    args,
    device,
    transformer,
    fm_scheduler,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    generator,
):
    """Standard ReFL trajectory sampling (Xu et al., NeurIPS 2023).

    Default behavior: roll out 25 denoising steps and back-prop through one
    randomly chosen step from the last 11 (i.e. fine-tune the tail 11/25).
    Both knobs are configurable via ``args.rollout_steps`` and
    ``args.refl_last_n_steps``.

    Steps:
      1. Run a no-grad rollout from x_T until we reach a randomly-chosen step
         t in the last ``refl_last_n_steps`` denoising steps.
      2. Run the transformer once at step t WITH gradient enabled.
      3. Predict x_0 in a single Euler step:  x0_hat = x_t - sigma_t * v_theta.
      4. Return that one-step x0_hat directly to the reward loss.

    The gradient path is exactly one transformer call -- we do not continue
    the rollout after the gradient step, and we do NOT use the LeapAlign
    "x0_hat + (x_T_final - x0_hat).detach()" reparameterization trick.
    """
    w, h = args.w, args.h
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

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    latents_xt = torch.randn(
        (B, IN_CHANNELS, latent_h, latent_w),
        device=device,
        dtype=torch.bfloat16,
    )
    latents_xt = pack_latents(
        latents_xt,
        B,
        IN_CHANNELS,
        latent_h,
        latent_w,
    )
    image_ids = prepare_latent_image_ids(
        B,
        latent_h // 2,
        latent_w // 2,
        device,
        torch.bfloat16,
    )
    cfg_guidance = torch.tensor(
        [args.cfg_guidance],
        device=latents_xt.device,
        dtype=torch.bfloat16,
    )
    cfg_guidance = cfg_guidance.expand(latents_xt.shape[0])

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

    grad_idx = timesteps.size(0) - selected_index.item()

    # Phase 1: no-grad rollout from x_T up to (but not including) step grad_idx.
    with torch.no_grad():
        for idx in range(grad_idx):
            step_t = timesteps[idx]
            timestep = step_t.expand(latents_xt.shape[0]).to(latents_xt.dtype)
            with torch.autocast("cuda", torch.bfloat16):
                pred = transformer(
                    hidden_states=latents_xt,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep / 1000,
                    guidance=cfg_guidance,
                    txt_ids=text_ids,
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            latents_xt = fm_scheduler.step(
                pred,
                step_t,
                latents_xt,
                with_pred_x0=False,
            )[0]
            if latents_xt.dtype != torch.bfloat16:
                latents_xt = latents_xt.to(torch.bfloat16)

    # Phase 2: single gradient-enabled transformer call at step grad_idx.
    # latents_xt at this point is purely a no-grad tensor; the only path that
    # carries gradient is through the transformer call below.
    latents_xt = latents_xt.detach()
    step_t = timesteps[grad_idx]
    timestep = step_t.expand(latents_xt.shape[0]).to(latents_xt.dtype)
    with torch.autocast("cuda", torch.bfloat16):
        pred = transformer(
            hidden_states=latents_xt,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep / 1000,
            guidance=cfg_guidance,
            txt_ids=text_ids,
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

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


def train_refl_one_step(
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

    _, latents_pred_x0, traj_sim_weight_factor, traj_metrics = sample_refl_trajectory(
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

    loss.backward()
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


class LeapAlignReFLFluxTrainer(LeapAlignFluxTrainer):
    """Standard ReFL trainer for FLUX.

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

    def sample_trajectory(self, device, transformer, fm_scheduler, encoder_hidden_states, pooled_prompt_embeds, text_ids, generator):
        return sample_refl_trajectory(
            self.args,
            device,
            transformer,
            fm_scheduler,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            generator,
        )

    def train_one_step(self, inner_step, device, transformer, vae, fm_scheduler, reward_model, tokenizer, optimizer, lr_scheduler, loader, preprocess_val, select_idx_generator):
        return train_refl_one_step(
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
