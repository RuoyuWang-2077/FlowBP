from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import yaml


@dataclass
class FlowBPEvalConfig:
    prompts_file: str = "./assets/eval_prompts.txt"
    num_imgs_per_prompt: int = 4
    seed: int = 42
    num_steps: int = 50
    guidance_scale: float = 3.5
    reward_fn: dict[str, float] = field(default_factory=lambda: {"hpsv2": 1.0})
    reward_ckpt_path: str = ""


@dataclass
class FlowBPConfig:
    """Structured FlowBP config.

    The trainer still consumes an argparse-style namespace for compatibility.
    This dataclass documents the intended config surface while
    ``load_config_defaults`` maps YAML values onto parser defaults.
    """

    raw: dict[str, Any] = field(default_factory=dict)


_SECTION_KEY_MAP: dict[str, dict[str, str]] = {
    "trainer": {
        "name": "trainer",
    },
    "model": {
        "pretrained_model_name_or_path": "pretrained_model_name_or_path",
        "gradient_checkpointing": "gradient_checkpointing",
        "master_weight_type": "master_weight_type",
        "use_ema": "use_ema",
        "ema_decay": "ema_decay",
    },
    "data": {
        "data_json_path": "data_json_path",
        "num_latent_t": "num_latent_t",
        "cfg": "cfg",
        "dataloader_num_workers": "dataloader_num_workers",
    },
    "train": {
        "seed": "seed",
        "train_batch_size": "train_batch_size",
        "train_sp_batch_size": "train_sp_batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "max_train_steps": "max_train_steps",
        "checkpointing_steps": "checkpointing_steps",
        "output_dir": "output_dir",
        "allow_tf32": "allow_tf32",
        "debug": "debug",
        "connector_dump_interval": "connector_dump_interval",
        "connector_dump_dir": "connector_dump_dir",
        "connector_wandb_interval": "connector_wandb_interval",
        "connector_wandb_num_samples": "connector_wandb_num_samples",
        "resume_from_checkpoint": "resume_from_checkpoint",
    },
    "optimizer": {
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "lr_scheduler": "lr_scheduler",
        "lr_warmup_steps": "lr_warmup_steps",
        "lr_num_cycles": "lr_num_cycles",
        "lr_power": "lr_power",
        "max_grad_norm": "max_grad_norm",
    },
    "distributed": {
        "sp_size": "sp_size",
        "fsdp_sharding_startegy": "fsdp_sharding_startegy",
        "selective_checkpointing": "selective_checkpointing",
        "use_cpu_offload": "use_cpu_offload",
    },
    "sampling": {
        "h": "h",
        "w": "w",
        "sampling_steps": "sampling_steps",
        "rollout_steps": "rollout_steps",
        "sampler_seed": "sampler_seed",
        "cfg_guidance": "cfg_guidance",
    },
    "reward": {
        "use_hpsv2": "use_hpsv2",
        "loss_grad_scale": "loss_grad_scale",
        "loss_relu_clip": "loss_relu_clip",
    },
    "flowbp": {
        "select_idx_seed": "select_idx_seed",
        "min_idx": "min_idx",
        "max_idx": "max_idx",
        "train_step_tail_ratio": "train_step_tail_ratio",
        "alpha": "alpha",
        "tau": "tau",
        # j-k index sampling shared by FlowBP-Bridge/Lagrange
        "jk_sampling_mode": "jk_sampling_mode",
        "jk_dirichlet_alpha": "jk_dirichlet_alpha",
        "jk_dirichlet_alpha_a": "jk_dirichlet_alpha_a",
        "jk_dirichlet_alpha_b": "jk_dirichlet_alpha_b",
        "jk_dirichlet_alpha_c": "jk_dirichlet_alpha_c",
        "jk_dirichlet_max_j_rev": "jk_dirichlet_max_j_rev",
    },
    "refl": {
        "last_n_steps": "refl_last_n_steps",
        "refl_last_n_steps": "refl_last_n_steps",
    },
    "flowbp_sparse": {
        "num_active_steps": "flowbp_sparse_num_active_steps",
        "late_bias": "flowbp_sparse_late_bias",
        "grad_rescale": "flowbp_sparse_grad_rescale",
        "flowbp_sparse_num_active_steps": "flowbp_sparse_num_active_steps",
        "flowbp_sparse_late_bias": "flowbp_sparse_late_bias",
        "flowbp_sparse_grad_rescale": "flowbp_sparse_grad_rescale",
    },
    "flowbp_bridge": {
        "alpha": "alpha",
    },
    "drtune": {
        "num_train_steps": "drtune_num_train_steps",
        "early_stop_steps": "drtune_early_stop_steps",
        "early_stop_ratio": "drtune_early_stop_ratio",
        "drtune_num_train_steps": "drtune_num_train_steps",
        "drtune_early_stop_steps": "drtune_early_stop_steps",
        "drtune_early_stop_ratio": "drtune_early_stop_ratio",
    },
    "draft_lv": {
        "num_noised_samples": "draft_lv_num_noised_samples",
        "draft_lv_num_noised_samples": "draft_lv_num_noised_samples",
    },
    "flowbp_lagrange": {
        "alpha": "alpha",
        "clip_dj_threshold": "clip_dj_threshold",
        "clip_d0": "clip_d0",
        "clip_d0_threshold": "clip_d0_threshold",
        "tau": "tau",
        "connector_order": "flowbp_lagrange_connector_order",
        "detach_history": "flowbp_lagrange_detach_history",
        "grad_support_mode": "flowbp_lagrange_grad_support_mode",
        "grad_support_scale": "flowbp_lagrange_grad_support_scale",
        "max_active_supports": "flowbp_lagrange_max_active_supports",
        "grad_rescale": "flowbp_lagrange_grad_rescale",
        "weight_scheme": "flowbp_lagrange_weight_scheme",
        "anchor_lambda": "flowbp_lagrange_anchor_lambda",
        "debug_connector": "debug_flowbp_lagrange_connector",
        "flowbp_lagrange_connector_order": "flowbp_lagrange_connector_order",
        "flowbp_lagrange_detach_history": "flowbp_lagrange_detach_history",
        "flowbp_lagrange_grad_support_mode": "flowbp_lagrange_grad_support_mode",
        "flowbp_lagrange_grad_support_scale": "flowbp_lagrange_grad_support_scale",
        "flowbp_lagrange_max_active_supports": "flowbp_lagrange_max_active_supports",
        "flowbp_lagrange_grad_rescale": "flowbp_lagrange_grad_rescale",
        "flowbp_lagrange_weight_scheme": "flowbp_lagrange_weight_scheme",
        "flowbp_lagrange_anchor_lambda": "flowbp_lagrange_anchor_lambda",
        "debug_flowbp_lagrange_connector": "debug_flowbp_lagrange_connector",
    },
    "logging": {
        "project": "project",
        "run_name": "run_name",
    },
    "flux2": {
        "max_sequence_length": "max_sequence_length",
        "text_encoder_out_layers": "text_encoder_out_layers",
        "caption_key": "caption_key",
        "use_precomputed_embeds": "use_precomputed_embeds",
        "cfg_grad_norm_compensate": "cfg_grad_norm_compensate",
        "cfg_detach_neg": "cfg_detach_neg",
    },
}

# Backward-compatible spelling from the original LeapAlign codebase.
_SECTION_KEY_MAP["leapalign"] = _SECTION_KEY_MAP["flowbp"]

_SECTION_KEY_MAP["flowbp-sparse"] = _SECTION_KEY_MAP["flowbp_sparse"]
_SECTION_KEY_MAP["flowbp-bridge"] = _SECTION_KEY_MAP["flowbp_bridge"]
_SECTION_KEY_MAP["flowbp-lagrange"] = _SECTION_KEY_MAP["flowbp_lagrange"]

TRAINER_ALIASES = {
    "flowbp-sparse": "flowbp_sparse",
    "flowbp-bridge": "flowbp_bridge",
    "flowbp-lagrange": "flowbp_lagrange",
}

FLOWBP_TRAINER_CHOICES = [
    "leapalign",
    "refl",
    "draft_lv",
    "drtune",
    "flowbp_sparse",
    "flowbp_bridge",
    "flowbp_lagrange",
]

SUPPORTED_INTERNAL_TRAINERS = {
    "leapalign",
    "refl",
    "flowbp_lagrange",
    "flowbp_sparse",
    "flowbp_bridge",
    "drtune",
    "draft_lv",
}


def normalize_trainer_name(name: str | None) -> str:
    """Map paper-facing FlowBP method names to implementation names."""

    raw_name = str(name or "leapalign").lower()
    normalized = TRAINER_ALIASES.get(raw_name, raw_name.replace("-", "_"))
    return TRAINER_ALIASES.get(normalized, normalized)


def _sanitize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def infer_model_name(config_path: str | Path | None, config_data: dict[str, Any] | None = None) -> str:
    """Infer the model family/name used in a config.

    Prefer the config path because final configs are intentionally named by
    backbone and scale, e.g. ``flux2_klein_base_9b_flowbp_lagrange.yaml``.
    """

    if config_path is not None:
        parts = Path(config_path).with_suffix("").parts
        if "flux1" in parts:
            return "flux1"
        if "flux2" in parts:
            return "flux2_klein_base_9b"
        if "sd35" in parts:
            return "sd3_5"

    model_path = ((config_data or {}).get("model") or {}).get(
        "pretrained_model_name_or_path", ""
    )
    model_name = _sanitize_name(Path(str(model_path)).name)
    if "flux" in model_name and "2" in model_name:
        return model_name
    if "flux" in model_name:
        return "flux1"
    if "sd3" in model_name or "3_5" in model_name:
        return "sd3_5"
    return model_name or "model"


def infer_reward_name(config_data: dict[str, Any] | None = None) -> str:
    reward = (config_data or {}).get("reward") or {}
    if reward.get("use_hpsv2", False):
        return "hpsv2"
    eval_reward = ((config_data or {}).get("eval") or {}).get("reward_fn")
    if isinstance(eval_reward, dict) and eval_reward:
        return "_".join(_sanitize_name(name) for name in eval_reward)
    if isinstance(eval_reward, list) and eval_reward:
        return "_".join(_sanitize_name(name) for name in eval_reward)
    return "reward"


def build_default_run_name(config_path: str | Path | None, config_data: dict[str, Any]) -> str:
    model_name = infer_model_name(config_path, config_data)
    method_name = _sanitize_name(((config_data.get("trainer") or {}).get("name")) or "method")
    reward_name = infer_reward_name(config_data)
    batch_size = ((config_data.get("train") or {}).get("train_batch_size")) or "unknown"
    return f"{model_name}_{method_name}_{reward_name}_bs{batch_size}"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"FlowBP config must be a mapping, got {type(data).__name__}")
    return data


def _flatten_section(section_name: str, section: dict[str, Any], defaults: dict[str, Any]) -> None:
    key_map = _SECTION_KEY_MAP.get(section_name)
    if key_map is None:
        return
    for src_key, dst_key in key_map.items():
        if src_key in section:
            defaults[dst_key] = section[src_key]


def load_config_defaults(path: str | Path) -> dict[str, Any]:
    """Load a FlowBP YAML config and return argparse defaults.

    CLI flags are parsed after these defaults are installed, so any explicit
    command-line argument still wins over the YAML value.
    """

    data = _load_yaml(path)
    defaults: dict[str, Any] = {}
    for section_name, section in data.items():
        if isinstance(section, dict):
            _flatten_section(section_name, section, defaults)
        elif section_name in _SECTION_KEY_MAP:
            raise ValueError(
                f"Config section {section_name!r} must be a mapping, "
                f"got {type(section).__name__}"
            )

    eval_section = data.get("eval", {})
    if eval_section:
        if not isinstance(eval_section, dict):
            raise ValueError(
                f"Config section 'eval' must be a mapping, got {type(eval_section).__name__}"
            )
        eval_key_map = {
            "prompts_file": "eval_prompts_file",
            "num_imgs_per_prompt": "eval_num_imgs_per_prompt",
            "seed": "eval_seed",
            "num_steps": "eval_num_steps",
            "guidance_scale": "eval_guidance_scale",
            "reward_fn": "eval_reward_fn",
            "reward_ckpt_path": "eval_reward_ckpt_path",
            "hpsv3_config_path": "eval_hpsv3_config_path",
            "hpsv3_checkpoint_path": "eval_hpsv3_checkpoint_path",
            "interval": "evaluation_interval",
            "evaluation_interval": "evaluation_interval",
        }
        for src_key, dst_key in eval_key_map.items():
            if src_key in eval_section:
                defaults[dst_key] = eval_section[src_key]

    defaults.setdefault("project", "flowbp")
    if not (((data.get("logging") or {}).get("run_name"))):
        defaults["run_name"] = build_default_run_name(path, data)

    return defaults
