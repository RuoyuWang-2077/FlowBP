from __future__ import annotations

from typing import Any

import numpy as np
import wandb

from flowbp.eval.generation.flux import GeneratedEvalItem


def build_wandb_eval_log(
    *,
    metric_values: dict[str, list[float]],
    local_items: list[GeneratedEvalItem],
    local_scores: list[dict[str, Any]],
    active_rewards: list[str],
    num_eval_images: int,
    max_media_images: int = 16,
) -> dict[str, Any]:
    log_dict: dict[str, Any] = {
        "eval/num_eval_images": num_eval_images,
    }
    for reward_name, values in metric_values.items():
        log_dict[f"eval/{reward_name}_avg"] = float(np.mean(values))
        log_dict[f"eval/{reward_name}_std"] = float(np.std(values))

    media = []
    for item, score_item in zip(local_items[:max_media_images], local_scores[:max_media_images]):
        caption_parts = [
            f"{name}={score_item[name]:.4f}"
            for name in active_rewards
            if name in score_item
        ]
        caption_parts.append(item.prompt[:80])
        media.append(
            wandb.Image(
                np.array(item.image),
                caption=" | ".join(caption_parts),
            )
        )
    if media:
        log_dict["eval/sample_images"] = media
    return log_dict
