
"""Shared j-k index sampler for LeapAlign / High-Order LeapAlign trainers.

This module factors the ``select_indices`` sampling logic out of every trainer
into a single helper so all four trainers (FLUX.1 / FLUX.2 × LeapAlign /
High-Order) share the same set of sampling modes:

* ``"uniform"`` — the original ``torch.randperm`` sampler (high variance,
  CV ≈ 66%).
* ``"dirichlet"`` — Dirichlet stabilisation that partitions
  ``[0, N=max_idx-min_idx]`` into three segments ``(A, B, C)`` via
  ``Dir(α_a, α_b, α_c)`` (sampled as normalised Gammas). The three segment
  lengths are deterministic functions of the proportions, so the resulting
  ``(k_idx, j_idx)`` placement has variance shrunk by ``α``. Optionally,
  ``jk_dirichlet_max_j_rev`` caps ``j_rev`` and moves overflow to segment C,
  preserving the sampled ``k_rev - j_rev`` gap.
* ``"midpoint"`` — sample ``k_rev`` uniformly, then place ``j_idx`` at the
  midpoint between ``k_idx`` and the end of the trajectory.
* ``"midpoint_j"`` — sample ``j_rev`` uniformly, then place ``k_idx`` at the
  midpoint between the start of the trajectory and ``j_idx``.

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


def _uniform_sample(args, total_steps: int, generator):
    max_j_rev = getattr(args, "jk_dirichlet_max_j_rev", None)
    if max_j_rev is not None:
        j_rev_lo = args.min_idx
        j_rev_hi = min(int(max_j_rev), args.max_idx - 2)
        if j_rev_hi < j_rev_lo:
            raise ValueError(
                "uniform jcap requires "
                f"min_idx <= jk_dirichlet_max_j_rev <= max_idx - 2; got "
                f"min_idx={args.min_idx}, max_idx={args.max_idx}, "
                f"jk_dirichlet_max_j_rev={max_j_rev}"
            )
        j_rev = torch.randint(
            j_rev_lo, j_rev_hi + 1, (1,), generator=generator,
        ).item()
        k_rev = torch.randint(
            j_rev + 1, args.max_idx, (1,), generator=generator,
        ).item()
        select_indices = torch.tensor([k_rev, j_rev], dtype=torch.long)
        k_idx = total_steps - k_rev
        j_idx = total_steps - j_rev
        return select_indices, k_idx, j_idx

    select_indices = (
        torch.randperm(
            args.max_idx - args.min_idx,
            device="cpu",
            generator=generator,
        )[:2]
        + args.min_idx
    ).long()
    select_indices = torch.sort(select_indices, descending=True).values
    k_idx = total_steps - select_indices[0].item()
    j_idx = total_steps - select_indices[1].item()
    return select_indices, k_idx, j_idx


def _dirichlet_sample(args, total_steps: int, generator):
    """Partition ``N = max_idx - min_idx`` into three segments via Dir(α_a, α_b, α_c).

    Reverse-index layout (smaller reverse-idx -> later in forward time):
        [min_idx, j_rev)          length=lengths[0]     "A"
        [j_rev,   k_rev)          length=lengths[1]     "B" (the gap)
        [k_rev,   min_idx + N)    length=lengths[2]     "C"

    So ``α_a`` controls how close ``j_idx`` is to the noise end,
    ``α_b`` controls the gap between ``k`` and ``j``, and ``α_c`` controls
    how close ``k_idx`` is to the clean end. Setting them all equal recovers
    the symmetric Dirichlet over the simplex.
    """
    n = args.max_idx - args.min_idx
    alpha = float(getattr(args, "jk_dirichlet_alpha", 4.2))
    alpha_a = float(getattr(args, "jk_dirichlet_alpha_a", None) or alpha)
    alpha_b = float(getattr(args, "jk_dirichlet_alpha_b", None) or alpha)
    alpha_c = float(getattr(args, "jk_dirichlet_alpha_c", None) or alpha)
    conc = torch.tensor([alpha_a, alpha_b, alpha_c])

    # Re-seed the gamma sampler from ``generator`` so a fresh Generator is
    # used here without contaminating ``generator``'s own state. This keeps
    # the gamma draw reproducible AND independent of how many gammas the
    # caller has already drawn.
    seed = torch.randint(0, 2**31, (1,), generator=generator).item()
    dir_gen = torch.Generator(device="cpu").manual_seed(seed)
    gamma_samples = torch._standard_gamma(
        conc.unsqueeze(0), generator=dir_gen,
    ).squeeze(0)
    props = gamma_samples / gamma_samples.sum()

    lengths = (props * n).round().long()
    diff = n - lengths.sum()
    lengths[props.argmax()] += diff
    lengths = lengths.clamp(min=1)
    diff = n - lengths.sum()
    if diff != 0:
        lengths[lengths.argmax()] += diff

    truncated = False
    max_j_rev = getattr(args, "jk_dirichlet_max_j_rev", None)
    if max_j_rev is not None:
        cap_l0 = max(1, int(max_j_rev) - args.min_idx)
        if lengths[0].item() > cap_l0:
            overflow = int(lengths[0].item() - cap_l0)
            lengths[0] = cap_l0
            lengths[2] += overflow
            truncated = True

    setattr(args, "_last_jk_truncated", truncated)
    j_rev = lengths[0].item() + args.min_idx
    k_rev = (lengths[0] + lengths[1]).item() + args.min_idx
    select_indices = torch.tensor([k_rev, j_rev], dtype=torch.long)
    k_idx = total_steps - k_rev
    j_idx = total_steps - j_rev
    return select_indices, k_idx, j_idx


def _midpoint_sample(args, total_steps: int, generator):
    """Sample ``k_rev`` uniformly, place ``j_idx`` at midpoint of ``[k_idx, total_steps]``."""
    k_rev_lo = args.min_idx + 1
    k_rev_hi = args.max_idx - 1
    if k_rev_hi < k_rev_lo:
        raise ValueError(
            f"jk_sampling_mode='midpoint' requires max_idx >= min_idx + 2, "
            f"got min_idx={args.min_idx}, max_idx={args.max_idx}"
        )
    k_rev = torch.randint(
        k_rev_lo, k_rev_hi + 1, (1,), generator=generator,
    ).item()
    k_idx = total_steps - k_rev
    j_idx = (k_idx + total_steps) // 2
    if j_idx <= k_idx:
        j_idx = k_idx + 1
    j_rev = total_steps - j_idx
    select_indices = torch.tensor([k_rev, j_rev], dtype=torch.long)
    return select_indices, k_idx, j_idx


def _midpoint_j_sample(args, total_steps: int, generator):
    """Sample ``j_rev`` uniformly, place ``k_idx`` at midpoint of ``[0, j_idx]``."""
    j_rev_lo = args.min_idx
    j_rev_hi = args.max_idx - 2
    if j_rev_hi < j_rev_lo:
        raise ValueError(
            f"jk_sampling_mode='midpoint_j' requires max_idx >= min_idx + 2, "
            f"got min_idx={args.min_idx}, max_idx={args.max_idx}"
        )
    j_rev = torch.randint(
        j_rev_lo, j_rev_hi + 1, (1,), generator=generator,
    ).item()
    j_idx = total_steps - j_rev
    k_idx = j_idx // 2
    if k_idx >= j_idx:
        k_idx = j_idx - 1
    if k_idx < 0:
        k_idx = 0
    k_rev = total_steps - k_idx
    select_indices = torch.tensor([k_rev, j_rev], dtype=torch.long)
    return select_indices, k_idx, j_idx


_DISPATCH = {
    "uniform": _uniform_sample,
    "dirichlet": _dirichlet_sample,
    "midpoint": _midpoint_sample,
    "midpoint_j": _midpoint_j_sample,
}


def sample_jk_indices(args, total_steps: int, generator):
    """Dispatch on ``args.jk_sampling_mode`` and return ``(select_indices, k_idx, j_idx)``.

    ``select_indices`` is a ``(2,)`` long tensor sorted descending in
    reverse-index space (``select_indices[0]`` corresponds to the earlier
    ``k`` step in forward time, ``select_indices[1]`` to the later ``j``).
    ``k_idx`` / ``j_idx`` are the forward-time indices, with
    ``0 <= k_idx < j_idx < total_steps`` guaranteed.

    Unknown modes fall back to ``"uniform"`` after logging a warning via
    ``RuntimeError`` (so misconfiguration fails loudly instead of silently).
    """
    sample_args = make_jk_window_args(args, total_steps, min_span=3)
    mode = str(getattr(sample_args, "jk_sampling_mode", "uniform")).lower()
    sampler = _DISPATCH.get(mode)
    if sampler is None:
        raise ValueError(
            f"Unknown jk_sampling_mode={mode!r}; expected one of "
            f"{sorted(_DISPATCH.keys())}."
        )
    setattr(args, "_last_jk_truncated", False)
    setattr(sample_args, "_last_jk_truncated", False)
    select_indices, k_idx, j_idx = sampler(sample_args, total_steps, generator)
    setattr(
        args,
        "_last_jk_truncated",
        bool(getattr(sample_args, "_last_jk_truncated", False)),
    )
    if not (0 <= k_idx < j_idx < total_steps):
        raise RuntimeError(
            f"Invalid jk indices: k_idx={k_idx}, j_idx={j_idx}, "
            f"total_steps={total_steps}, mode={mode!r}, "
            f"select_indices={select_indices.tolist()}, "
            f"min_idx={sample_args.min_idx}, max_idx={sample_args.max_idx}"
        )
    return select_indices, k_idx, j_idx
