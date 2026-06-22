"""Vanilla ReFL trainer for SD3.5.

Mirrors the ReFL recipe: roll out without grad until one random timestep
from the rollout tail, run exactly one transformer forward with grad, and use
that step's x0 prediction for the reward loss. No LeapAlign weighting, nested
gradient mixing, or FlowBP-Lagrange connector is involved.
"""

from __future__ import annotations

import torch

from flowbp.trainers.sd35.leapalign import (
    LeapAlignSD35Trainer,
    _model_forward_sd35,
    _prepare_timesteps,
    _sample_initial_latents,
    run_training_sd35,
)


def sample_refl_trajectory_sd35(
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
    latents_xt, _, _ = _sample_initial_latents(
        args, transformer, device, prompt_embeds.shape[0]
    )
    timesteps = _prepare_timesteps(args, fm_scheduler, transformer, latents_xt, device)
    total_steps = timesteps.size(0)
    last_n_steps = int(getattr(args, "refl_last_n_steps", 11))
    if last_n_steps < 1:
        raise ValueError(f"refl_last_n_steps must be >= 1, got {last_n_steps}")
    if last_n_steps > total_steps:
        raise ValueError(
            f"refl_last_n_steps={last_n_steps} cannot exceed rollout steps={total_steps}"
        )

    # Reverse-index convention: 1 = final denoising step, n = n-th before end.
    selected_rev = torch.randint(
        1,
        last_n_steps + 1,
        (1,),
        device="cpu",
        generator=generator,
    ).long()[0]
    grad_idx = total_steps - selected_rev.item()

    with torch.no_grad():
        for idx in range(grad_idx):
            step_t = timesteps[idx]
            pred = _model_forward_sd35(
                args,
                transformer,
                latents_xt,
                step_t,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            )
            latents_xt = fm_scheduler.step(
                pred,
                step_t,
                latents_xt,
                with_pred_x0=False,
            )[0]
            if latents_xt.dtype != torch.bfloat16:
                latents_xt = latents_xt.to(torch.bfloat16)

    latents_xt = latents_xt.detach()
    step_t = timesteps[grad_idx]
    pred = _model_forward_sd35(
        args,
        transformer,
        latents_xt,
        step_t,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )
    _, latents_pred_x0 = fm_scheduler.step(
        pred,
        step_t,
        latents_xt,
        with_pred_x0=True,
    )
    if latents_pred_x0.dtype != torch.bfloat16:
        latents_pred_x0 = latents_pred_x0.to(torch.bfloat16)

    traj_metrics = {
        "d0": None,
        "dj": None,
        "w_sim": None,
        "refl_selected_idx": torch.tensor(
            float(grad_idx), device=device, dtype=torch.float32
        ),
        "refl_selected_rev": torch.tensor(
            float(selected_rev.item()), device=device, dtype=torch.float32
        ),
        "refl_last_n_steps": torch.tensor(
            float(last_n_steps), device=device, dtype=torch.float32
        ),
    }
    args._last_connector_payload = {
        "step": getattr(args, "current_train_step", None),
        "selected_idx": int(grad_idx),
        "selected_rev": int(selected_rev.item()),
        "x0_pred": latents_pred_x0.detach(),
        "connector_scheme": "sd35_refl",
    }
    return latents_xt, latents_pred_x0, None, traj_metrics


class LeapAlignReFLSD35Trainer(LeapAlignSD35Trainer):
    def train(self):
        self.args.trainer = "refl"
        return run_training_sd35(self.args)
