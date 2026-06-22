from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass
from typing import Any


# Default location of the FlowBP-managed HPSv3 inference config that
# overrides `model_name_or_path` to a locally cached Qwen2-VL-7B-Instruct
# checkpoint (see assets/eval/hpsv3/HPSv3_7B.yaml). Resolved relative to
# the FlowBP repo root.
_FLOWBP_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), os.pardir, os.pardir))
_DEFAULT_HPSV3_CONFIG_PATH = osp.join(
    _FLOWBP_REPO_ROOT, "assets", "eval", "hpsv3", "HPSv3_7B.yaml"
)


@dataclass
class EvalConfig:
    prompts_file: str
    num_imgs_per_prompt: int
    seed: int
    num_steps: int
    guidance_scale: float
    reward_fn: dict[str, float]
    reward_ckpt_path: str
    output_dir: str | None
    height: int
    width: int
    pretrained_model_name_or_path: str
    # Optional explicit overrides for HPSv3. When empty we auto-discover from
    # `reward_ckpt_path` (e.g. `<reward_ckpt_path>/HPSv3/HPSv3.safetensors`)
    # and fall back to the bundled `_DEFAULT_HPSV3_CONFIG_PATH` YAML.
    hpsv3_config_path: str = ""
    hpsv3_checkpoint_path: str = ""

    @classmethod
    def from_args(cls, args: Any) -> "EvalConfig":
        reward_fn = getattr(args, "eval_reward_fn", {"hpsv2": 1.0})
        if isinstance(reward_fn, dict):
            reward_weights = {str(k): float(v) for k, v in reward_fn.items()}
        else:
            reward_weights = {str(name): 1.0 for name in reward_fn}

        return cls(
            prompts_file=getattr(args, "eval_prompts_file", "./assets/eval_prompts.txt"),
            num_imgs_per_prompt=int(getattr(args, "eval_num_imgs_per_prompt", 4)),
            seed=int(getattr(args, "eval_seed", 42)),
            num_steps=int(getattr(args, "eval_num_steps", 50)),
            guidance_scale=float(getattr(args, "eval_guidance_scale", 3.5)),
            reward_fn=reward_weights,
            reward_ckpt_path=getattr(args, "eval_reward_ckpt_path", ""),
            output_dir=getattr(args, "output_dir", None),
            height=int(getattr(args, "h")),
            width=int(getattr(args, "w")),
            pretrained_model_name_or_path=getattr(args, "pretrained_model_name_or_path"),
            hpsv3_config_path=str(getattr(args, "eval_hpsv3_config_path", "") or ""),
            hpsv3_checkpoint_path=str(
                getattr(args, "eval_hpsv3_checkpoint_path", "") or ""
            ),
        )


def _first_existing(*candidates: str | None) -> str:
    for candidate in candidates:
        if candidate and osp.isfile(candidate):
            return candidate
    return ""


def configure_hpsv3_env(eval_cfg: EvalConfig) -> dict[str, str]:
    """Populate FLOWBP_HPSV3_* env vars so an optional HPSv3 scorer can find
    a working local config/checkpoint pair before model construction.

    This is a no-op when `hpsv3` is not in `eval_cfg.reward_fn`, and never
    overwrites env vars that the caller already set. Returns the resolved
    paths for logging.
    """

    if "hpsv3" not in {str(k).lower() for k in eval_cfg.reward_fn}:
        return {}

    config_path = _first_existing(
        os.environ.get("FLOWBP_HPSV3_CONFIG_PATH"),
        eval_cfg.hpsv3_config_path,
        # Common local layouts under `<reward_ckpt_path>/HPSv3/`.
        osp.join(eval_cfg.reward_ckpt_path or "", "HPSv3", "HPSv3_7B.yaml"),
        osp.join(eval_cfg.reward_ckpt_path or "", "HPSv3", "config.yaml"),
        # Bundled fallback config, if the repository provides one.
        _DEFAULT_HPSV3_CONFIG_PATH,
    )
    checkpoint_path = _first_existing(
        os.environ.get("FLOWBP_HPSV3_CHECKPOINT_PATH"),
        eval_cfg.hpsv3_checkpoint_path,
        osp.join(eval_cfg.reward_ckpt_path or "", "HPSv3", "HPSv3.safetensors"),
        osp.join(eval_cfg.reward_ckpt_path or "", "HPSv3", "HPS_v3.1_compressed.pt"),
        osp.join(eval_cfg.reward_ckpt_path or "", "HPSv3", "HPSv3.pt"),
    )

    resolved: dict[str, str] = {}
    if config_path:
        os.environ.setdefault("FLOWBP_HPSV3_CONFIG_PATH", config_path)
        resolved["FLOWBP_HPSV3_CONFIG_PATH"] = os.environ["FLOWBP_HPSV3_CONFIG_PATH"]
    if checkpoint_path:
        os.environ.setdefault("FLOWBP_HPSV3_CHECKPOINT_PATH", checkpoint_path)
        resolved["FLOWBP_HPSV3_CHECKPOINT_PATH"] = os.environ[
            "FLOWBP_HPSV3_CHECKPOINT_PATH"
        ]
    return resolved
