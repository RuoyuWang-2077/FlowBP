from __future__ import annotations

from typing import Any

import numpy as np
import torch


def as_float_list(scores: Any) -> list[float]:
    if isinstance(scores, torch.Tensor):
        return scores.detach().float().cpu().view(-1).tolist()
    if isinstance(scores, np.ndarray):
        return scores.reshape(-1).astype(float).tolist()
    return [float(score) for score in scores]


def collect_metric_values(
    all_scores: list[dict[str, Any]],
    metric_names: list[str],
    *,
    invalid_sentinel: float = -10.0,
) -> dict[str, list[float]]:
    metric_values: dict[str, list[float]] = {}
    for metric_name in metric_names:
        values = [
            float(item[metric_name])
            for item in all_scores
            if metric_name in item and float(item[metric_name]) != invalid_sentinel
        ]
        if values:
            metric_values[metric_name] = values
    return metric_values


def summarize_metric_values(metric_values: dict[str, list[float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name, values in metric_values.items():
        summary[f"{metric_name}_avg"] = float(np.mean(values))
        summary[f"{metric_name}_std"] = float(np.std(values))
    return summary
