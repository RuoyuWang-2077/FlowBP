"""Connector diagnostics shared by every FlowBP backbone.

Saving a connector dump, logging decoded connector pairs to W&B and building
the per-sample trajectory filter are all backend-agnostic: the only part that
differs is how latents are turned into images, which callers inject as
``decode_fn``.

These helpers previously existed as eight near-identical copies spread over
the FLUX.1, FLUX.2 and SD3.5 trainers, and only the SD3.5 copy had the
trajectory filter.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.distributed as dist
import wandb

CONNECTOR_IMAGE_KEYS = ("xj_pred", "xj", "x0_pred", "x0")


def _is_logging_rank() -> bool:
    return not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)


def _due(args, interval_attr: str, fallback_attr: str | None = None) -> int | None:
    """Return the current step when ``interval_attr`` fires on it, else None."""
    interval = int(getattr(args, interval_attr, 0) or 0)
    if interval <= 0 and fallback_attr is not None:
        interval = int(getattr(args, fallback_attr, 0) or 0)
    step = getattr(args, "current_train_step", None)
    if interval <= 0 or step is None or int(step) % interval != 0:
        return None
    return int(step)


def maybe_save_connector_dump(args, prefix: str, payload: dict[str, Any]) -> None:
    """Persist a connector payload to disk every ``connector_dump_interval`` steps."""
    step = _due(args, "connector_dump_interval")
    if step is None or not _is_logging_rank():
        return

    dump_dir = getattr(args, "connector_dump_dir", None) or os.path.join(
        args.output_dir, "connector_dumps"
    )
    os.makedirs(dump_dir, exist_ok=True)
    saved = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in payload.items()
    }
    torch.save(saved, os.path.join(dump_dir, f"{prefix}_step_{step:06d}.pt"))


def maybe_log_connector_wandb(
    args,
    prefix: str,
    payload: dict[str, Any] | None,
    decode_fn: Callable[[torch.Tensor, int], list],
) -> None:
    """Log decoded connector latents to W&B.

    ``decode_fn(latents, max_samples)`` returns a list of HWC float arrays; it
    owns the backend-specific unpack + VAE denormalization.
    """
    step = _due(args, "connector_wandb_interval", "connector_dump_interval")
    if step is None or not _is_logging_rank():
        return
    if payload is None or wandb.run is None:
        return

    max_samples = int(getattr(args, "connector_wandb_num_samples", 2) or 2)
    log_payload = {}
    for name in CONNECTOR_IMAGE_KEYS:
        latents = payload.get(name)
        if latents is None:
            continue
        images = decode_fn(latents, max_samples)
        if images:
            log_payload[f"connector/{prefix}_{name}"] = [
                wandb.Image(image, caption=f"{prefix} {name} sample {idx}")
                for idx, image in enumerate(images)
            ]
    if log_payload:
        wandb.log(log_payload, step=step)


def images_to_wandb_arrays(images: torch.Tensor) -> list:
    """Convert a decoded CHW image batch into the HWC arrays W&B expects."""
    return [
        image.detach().float().cpu().permute(1, 2, 0).numpy() for image in images
    ]


def _positive_or_none(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def build_traj_filter_mask(args, d_j: torch.Tensor, d_0: torch.Tensor):
    """Keep-mask for per-sample trajectory filtering, or ``(None, {})`` if off.

    Samples whose connector residual ``d_j`` or endpoint residual ``d_0``
    exceeds the configured maximum are dropped from the reward loss. The
    forward rollout still runs for them; only their gradient is removed.
    """
    dj_max = _positive_or_none(getattr(args, "traj_filter_dj_max", None))
    d0_max = _positive_or_none(getattr(args, "traj_filter_d0_max", None))
    if dj_max is None and d0_max is None:
        args._last_sample_keep_mask = None
        return None, {}

    keep_mask = torch.ones_like(d_j, dtype=torch.bool)
    if dj_max is not None:
        keep_mask = keep_mask & (d_j <= dj_max)
    if d0_max is not None:
        keep_mask = keep_mask & (d_0 <= d0_max)

    args._last_sample_keep_mask = keep_mask.detach()
    keep_rate = keep_mask.float().mean()
    return keep_mask, {
        "traj_filter_enabled": torch.tensor(1.0, device=d_j.device, dtype=torch.float32),
        "traj_filter_keep_rate": keep_rate.detach(),
        "traj_filter_drop_rate": (1.0 - keep_rate).detach(),
        "traj_filter_kept": keep_mask.float().sum().detach(),
    }
