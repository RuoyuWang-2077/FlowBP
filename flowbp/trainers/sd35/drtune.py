"""DRTune trainer for SD3.5: train K spaced steps in an early-stopped prefix."""

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

DEFAULT_DRTUNE_NUM_TRAIN_STEPS = 3
DEFAULT_DRTUNE_EARLY_STOP_RATIO = 0.4


def _sample_equally_spaced_train_indices(
    total_steps: int,
    num_train_steps: int,
    generator: torch.Generator | None,
) -> list[int]:
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


def _sample_drtune_early_stop_idx(
    args,
    total_steps: int,
    generator: torch.Generator | None,
) -> int:
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


def sample_drtune_trajectory_sd35(
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

    early_stop_idx = _sample_drtune_early_stop_idx(args, total_steps, generator)
    num_train_steps = int(
        getattr(args, "drtune_num_train_steps", DEFAULT_DRTUNE_NUM_TRAIN_STEPS)
    )
    train_indices = _sample_equally_spaced_train_indices(
        total_steps=early_stop_idx + 1,
        num_train_steps=num_train_steps,
        generator=generator,
    )
    train_index_set = set(train_indices)
    executed_train_indices: list[int] = []
    latents_pred_x0 = None

    for idx, step_t in enumerate(timesteps):
        model_input = latents_xt.detach()
        if idx in train_index_set:
            pred = _model_forward_sd35(
                args,
                transformer,
                model_input,
                step_t,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            )
            executed_train_indices.append(idx)
        else:
            with torch.no_grad():
                pred = _model_forward_sd35(
                    args,
                    transformer,
                    model_input,
                    step_t,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
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
        raise RuntimeError("DRTune SD3.5 rollout ended without x0 prediction.")
    if latents_pred_x0.dtype != torch.bfloat16:
        latents_pred_x0 = latents_pred_x0.to(torch.bfloat16)

    traj_metrics = {
        "d0": None,
        "dj": None,
        "w_sim": None,
        "drtune_num_train_steps": torch.tensor(
            len(train_indices), device=device, dtype=torch.float32,
        ),
        "drtune_num_executed_train_steps": torch.tensor(
            len(executed_train_indices), device=device, dtype=torch.float32,
        ),
        "drtune_early_stop_idx": torch.tensor(
            early_stop_idx, device=device, dtype=torch.float32,
        ),
    }
    args._last_connector_payload = {
        "step": getattr(args, "current_train_step", None),
        "train_indices": train_indices,
        "executed_train_indices": executed_train_indices,
        "early_stop_idx": int(early_stop_idx),
        "connector_scheme": "sd35_drtune_stop_gradient",
    }
    _maybe_save_connector_dump(args, "sd35_drtune", args._last_connector_payload)
    return latents_xt, latents_pred_x0, None, traj_metrics


class LeapAlignDRTuneSD35Trainer(LeapAlignSD35Trainer):
    def train(self):
        self.args.trainer = "drtune"
        return run_training_sd35(self.args)
