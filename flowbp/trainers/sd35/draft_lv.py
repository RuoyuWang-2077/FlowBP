"""DRaFT-LV trainer for SD3.5: last-step x0 plus noised last-step variants."""

from __future__ import annotations

import torch

from flowbp.trainers.sd35.leapalign import (
    LeapAlignSD35Trainer,
    _maybe_save_connector_dump,
    _model_forward_sd35,
    _prepare_timesteps,
    _sample_initial_latents,
    run_training_sd35,
)

DEFAULT_DRAFT_LV_NUM_NOISED_SAMPLES = 2


def sample_draft_lv_trajectory_sd35(
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
    final_idx = timesteps.size(0) - 1

    with torch.no_grad():
        for idx in range(final_idx):
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

    final_step_t = timesteps[final_idx]
    latents_last = latents_xt.detach()
    pred = _model_forward_sd35(
        args,
        transformer,
        latents_last,
        final_step_t,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )
    _, base_x0 = fm_scheduler.step(
        pred,
        final_step_t,
        latents_last,
        with_pred_x0=True,
    )
    if base_x0.dtype != torch.bfloat16:
        base_x0 = base_x0.to(torch.bfloat16)

    num_noised = int(
        getattr(args, "draft_lv_num_noised_samples", DEFAULT_DRAFT_LV_NUM_NOISED_SAMPLES)
    )
    if num_noised < 0:
        raise ValueError(
            f"draft_lv_num_noised_samples must be >= 0, got {num_noised}"
        )

    x0_predictions = [base_x0]
    sigma_last = fm_scheduler.sigmas[final_idx].to(
        device=base_x0.device,
        dtype=torch.float32,
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
        pred_lv = _model_forward_sd35(
            args,
            transformer,
            x_last,
            final_step_t,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        x0_lv = x_last.float() - sigma_last * pred_lv.float()
        x0_predictions.append(x0_lv.to(torch.bfloat16))

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
        "connector_scheme": "sd35_draft_lv_last_step",
    }
    _maybe_save_connector_dump(args, "sd35_draft_lv", args._last_connector_payload)
    return latents_last, latents_pred_x0, None, traj_metrics


class LeapAlignDRaFTLVSD35Trainer(LeapAlignSD35Trainer):
    def train(self):
        self.args.trainer = "draft_lv"
        return run_training_sd35(self.args)
