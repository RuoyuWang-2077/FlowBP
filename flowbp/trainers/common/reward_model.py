"""Shared training-time reward model construction.

The FLUX.1 / FLUX.2 / SD3.5 trainers all fine-tune against the same
differentiable HPSv2.1 reward, so the checkpoint lookup lives here instead of
being duplicated (and hardcoded) in every backend module.

Resolution order for the checkpoint directory:

1. ``args.reward_ckpt_path`` (``reward.reward_ckpt_path`` in YAML)
2. ``args.eval_reward_ckpt_path`` (``eval.reward_ckpt_path`` in YAML)
3. ``FLOWBP_REWARD_CKPT_PATH``
4. ``<repo>/models/reward_ckpts`` then ``<repo>/reward_ckpts``
5. ``./hps_ckpt`` and ``<repo>/hps_ckpt`` (layout used by the original
   LeapAlign codebase)

Each candidate directory is probed independently for the OpenCLIP backbone and
the HPSv2 reward weights, so a partially populated directory still resolves as
long as some directory on the list provides each file.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from flowbp.utils.logging_ import main_print

_REPO_ROOT = Path(__file__).resolve().parents[3]

HPSV2_BACKBONE_FILENAMES = ("open_clip_pytorch_model.bin",)
HPSV2_REWARD_FILENAMES = ("HPS_v2.1_compressed.pt", "HPS_v2_compressed.pt")


def reward_ckpt_search_dirs(args) -> list[str]:
    """Ordered, de-duplicated list of directories to probe for reward weights."""
    candidates: list[str] = []

    def add(value) -> None:
        if not value:
            return
        path = os.path.expanduser(str(value).strip())
        if path and path not in candidates:
            candidates.append(path)

    add(getattr(args, "reward_ckpt_path", None))
    add(getattr(args, "eval_reward_ckpt_path", None))
    add(os.environ.get("FLOWBP_REWARD_CKPT_PATH", ""))
    add(_REPO_ROOT / "models" / "reward_ckpts")
    add(_REPO_ROOT / "reward_ckpts")
    add("./hps_ckpt")
    add(_REPO_ROOT / "hps_ckpt")
    return candidates


def _find_first(search_dirs: list[str], filenames: tuple[str, ...]) -> str | None:
    for directory in search_dirs:
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                return path
    return None


def resolve_hpsv2_checkpoints(args) -> tuple[str, str]:
    """Locate the OpenCLIP backbone and the HPSv2 reward checkpoint."""
    search_dirs = reward_ckpt_search_dirs(args)
    backbone_path = _find_first(search_dirs, HPSV2_BACKBONE_FILENAMES)
    reward_path = _find_first(search_dirs, HPSV2_REWARD_FILENAMES)

    missing = []
    if backbone_path is None:
        missing.append(f"OpenCLIP backbone ({' or '.join(HPSV2_BACKBONE_FILENAMES)})")
    if reward_path is None:
        missing.append(f"HPSv2 reward weights ({' or '.join(HPSV2_REWARD_FILENAMES)})")
    if missing:
        searched = "\n  ".join(search_dirs)
        raise FileNotFoundError(
            "Could not locate the HPSv2 training reward checkpoints.\n"
            f"Missing: {'; '.join(missing)}\n"
            f"Searched:\n  {searched}\n"
            "Set reward.reward_ckpt_path in the YAML config, pass "
            "--reward_ckpt_path, or export FLOWBP_REWARD_CKPT_PATH."
        )
    return backbone_path, reward_path


def init_hpsv2_reward_model(args, device):
    """Build the frozen HPSv2.1 reward model used as the training objective.

    Returns ``(reward_model, tokenizer, preprocess_val)`` where
    ``preprocess_val`` is the gradient-preserving image transform.
    """
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    from flowbp.utils.hpsv2_transforms import HPSV2TransformsWithGrad

    backbone_path, reward_path = resolve_hpsv2_checkpoints(args)
    main_print(f"--> HPSv2 OpenCLIP backbone: {backbone_path}")
    main_print(f"--> HPSv2 reward checkpoint: {reward_path}")

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        backbone_path,
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    preprocess_val = HPSV2TransformsWithGrad(preprocess_val)

    checkpoint = torch.load(reward_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")

    reward_model = model.to(device)
    reward_model.requires_grad_(False)
    reward_model.eval()
    return reward_model, tokenizer, preprocess_val
