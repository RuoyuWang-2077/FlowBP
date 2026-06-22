
"""Online eval runner for FLUX.2 LeapAlign training.

Same control flow as :func:`flowbp.eval.runners.online.run_online_flux_eval`
but uses ``Flux2KleinPipeline`` for image generation.
"""

from __future__ import annotations

import gc
import json
import os
import os.path as osp
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import wandb

from flowbp.eval.config import EvalConfig, configure_hpsv3_env
from flowbp.eval.generation.flux2 import (
    GeneratedEvalItem,
    sample_flux2_images_distributed,
)
from flowbp.eval.logging_wandb import build_wandb_eval_log
from flowbp.eval.metrics import as_float_list, collect_metric_values
from flowbp.eval.rewards.multi_scorer import build_reward_scorer
from flowbp.utils.logging_ import main_print


def _score_local_items(
    scorer: Any,
    local_items: list[GeneratedEvalItem],
) -> list[dict[str, Any]]:
    if not local_items:
        return []

    local_images = [item.image for item in local_items]
    local_prompts = [item.prompt for item in local_items]
    score_details, _ = scorer(local_images, local_prompts, metadata={})
    score_lists = {
        name: as_float_list(scores)
        for name, scores in score_details.items()
    }

    local_scores: list[dict[str, Any]] = []
    for item_idx, item in enumerate(local_items):
        score_item: dict[str, Any] = {
            "prompt_idx": item.prompt_idx,
            "img_idx": item.img_idx,
            "prompt": item.prompt,
        }
        for reward_name, reward_values in score_lists.items():
            if item_idx < len(reward_values):
                score_item[reward_name] = reward_values[item_idx]
        local_scores.append(score_item)
    return local_scores


@torch.no_grad()
def run_online_flux2_eval(
    step: int,
    args: Any,
    transformer,
    vae,
    rank: int,
    world_size: int,
    device: int | str | torch.device,
) -> None:
    """Distributed FLUX.2 eval: generate images, score with reward fns, log to wandb."""

    eval_cfg = EvalConfig.from_args(args)
    main_print(f"--> Online Evaluation (FLUX.2) step {step}")

    hpsv3_paths = configure_hpsv3_env(eval_cfg)
    if hpsv3_paths:
        main_print(f"--> HPSv3 paths: {hpsv3_paths}")

    local_items = sample_flux2_images_distributed(
        eval_cfg=eval_cfg,
        transformer=transformer,
        vae=vae,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    scorer, active_rewards = build_reward_scorer(
        reward_fn=eval_cfg.reward_fn,
        device=device,
        reward_ckpt_path=eval_cfg.reward_ckpt_path,
        allow_unavailable=True,
    )
    try:
        local_scores = _score_local_items(scorer, local_items)
    finally:
        scorer.to(torch.device("cpu"))
        del scorer

    all_scores_gathered = [None] * world_size
    dist.all_gather_object(all_scores_gathered, local_scores)

    if rank == 0:
        all_scores: list[dict[str, Any]] = []
        for rank_scores in all_scores_gathered:
            if rank_scores is not None:
                all_scores.extend(rank_scores)

        metric_names = list(active_rewards) + ["mean"]
        metric_values = collect_metric_values(all_scores, metric_names)
        metric_summary = " | ".join(
            f"{name}={float(np.mean(values)):.4f}"
            for name, values in metric_values.items()
        )
        num_eval_images = len(all_scores)
        main_print(
            f"--> Step {step} | Eval {metric_summary} | "
            f"num_images: {num_eval_images}"
        )

        wandb.log(
            build_wandb_eval_log(
                metric_values=metric_values,
                local_items=local_items,
                local_scores=local_scores,
                active_rewards=active_rewards,
                num_eval_images=num_eval_images,
            ),
            step=step,
        )

        if eval_cfg.output_dir:
            eval_log_dir = osp.join(eval_cfg.output_dir, "eval_logs")
            os.makedirs(eval_log_dir, exist_ok=True)
            eval_log_path = osp.join(eval_log_dir, f"eval_step_{step}.json")
            eval_log = {
                "step": step,
                "avg_scores": {
                    name: float(np.mean(values))
                    for name, values in metric_values.items()
                },
                "num_images": num_eval_images,
                "scores": all_scores,
            }
            with open(eval_log_path, "w", encoding="utf-8") as f:
                json.dump(eval_log, f, indent=2)

    torch.cuda.empty_cache()
    gc.collect()
    if world_size > 1:
        dist.barrier()
