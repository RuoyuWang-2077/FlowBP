"""Shared j-k index sampler for the LeapAlign / FlowBP trainers.

This module factors the ``select_indices`` sampling logic out of every trainer
into a single helper: two distinct reverse indices are drawn uniformly from
``[min_idx, max_idx)``, clipped to the trainable tail window.

The function returns a tuple ``(select_indices, k_idx, j_idx)`` where
``select_indices`` is a length-2 ``torch.long`` tensor sorted descending
(``select_indices[0] = k_rev > select_indices[1] = j_rev``) so existing
trainer code that indexes ``timesteps.size(0) - select_indices[X].item()``
continues to work unchanged. ``k_idx`` and ``j_idx`` are the forward-time
indices satisfying ``0 <= k_idx < j_idx < total_steps``.

All randomness flows through the supplied ``generator`` so the sampler stays
deterministic across ranks when ``args.select_idx_seed`` is set.
"""

from __future__ import annotations

import torch

from flowbp.trainers.common.rollout_window import make_jk_window_args


def sample_jk_indices(args, total_steps: int, generator):
    """Draw ``(select_indices, k_idx, j_idx)`` uniformly from the tail window."""
    sample_args = make_jk_window_args(args, total_steps, min_span=3)
    select_indices = (
        torch.randperm(
            sample_args.max_idx - sample_args.min_idx,
            device="cpu",
            generator=generator,
        )[:2]
        + sample_args.min_idx
    ).long()
    select_indices = torch.sort(select_indices, descending=True).values
    k_idx = total_steps - select_indices[0].item()
    j_idx = total_steps - select_indices[1].item()
    if not (0 <= k_idx < j_idx < total_steps):
        raise RuntimeError(
            f"Invalid jk indices: k_idx={k_idx}, j_idx={j_idx}, "
            f"total_steps={total_steps}, "
            f"select_indices={select_indices.tolist()}, "
            f"min_idx={sample_args.min_idx}, max_idx={sample_args.max_idx}"
        )
    return select_indices, k_idx, j_idx
