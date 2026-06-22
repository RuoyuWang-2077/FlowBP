"""Utilities for limiting FlowBP trainable steps to the rollout tail."""

from __future__ import annotations

import copy
import math
from typing import Any


def get_train_step_tail_ratio(args: Any) -> float:
    """Return the fraction of final rollout steps exposed to FlowBP training."""
    raw_ratio = getattr(args, "train_step_tail_ratio", 1.0)
    if raw_ratio is None:
        return 1.0

    ratio = float(raw_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(
            f"train_step_tail_ratio must be in (0, 1], got {raw_ratio!r}"
        )
    return ratio


def resolve_train_step_window(
    args: Any,
    total_steps: int,
    *,
    min_window: int = 1,
) -> tuple[int, int]:
    """Resolve the trainable forward-index window ``[start_idx, end_idx)``.

    FlowBP rollout caches use forward indices where ``0`` is the noisy end and
    larger indices are closer to the clean endpoint. A tail ratio of ``0.6`` on
    a 25-step rollout therefore exposes indices ``[10, 25)``.
    """
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    min_window = max(1, min(int(min_window), total_steps))
    window_len = max(
        min_window,
        int(math.ceil(total_steps * get_train_step_tail_ratio(args))),
    )
    window_len = min(window_len, total_steps)
    return total_steps - window_len, total_steps


def resolve_reverse_index_window(
    args: Any,
    total_steps: int,
    *,
    min_span: int = 2,
) -> tuple[int, int]:
    """Resolve reverse-index bounds ``[min_idx, max_idx)`` for j-k sampling."""
    min_span = max(1, int(min_span))
    start_idx, end_idx = resolve_train_step_window(
        args,
        total_steps,
        min_window=min_span,
    )

    original_min = int(getattr(args, "min_idx", 1))
    original_max = int(getattr(args, "max_idx", total_steps + 1))

    # reverse_idx = total_steps - forward_idx. ``max_idx`` is exclusive.
    tail_min = total_steps - (end_idx - 1)
    tail_max = total_steps - start_idx + 1
    min_idx = max(original_min, tail_min)
    max_idx = min(original_max, tail_max)

    if max_idx - min_idx < min_span:
        raise ValueError(
            "Trainable rollout window is too small for j-k sampling: "
            f"min_idx={min_idx}, max_idx={max_idx}, min_span={min_span}, "
            f"total_steps={total_steps}, "
            f"train_step_tail_ratio={get_train_step_tail_ratio(args)}"
        )
    return min_idx, max_idx


def make_jk_window_args(
    args: Any,
    total_steps: int,
    *,
    min_span: int = 3,
) -> Any:
    """Return an args copy whose ``min_idx/max_idx`` are clipped to the tail."""
    min_idx, max_idx = resolve_reverse_index_window(
        args,
        total_steps,
        min_span=min_span,
    )
    scoped_args = copy.copy(args)
    scoped_args.min_idx = min_idx
    scoped_args.max_idx = max_idx
    return scoped_args
