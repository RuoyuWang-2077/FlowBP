"""Single source of truth for FlowBP hyperparameter defaults.

Before this module every default lived in two places at once: the argparse
definition in ``train_flowbp_flux.py`` / ``train_flowbp_flux2.py``, and a
``getattr(args, "name", <literal>)`` fallback inside each of the 18 method
trainers. The two parsers had drifted apart on 22 shared arguments and the
per-method fallbacks had drifted on several of them.

Resolution order, lowest priority first:

1. ``BASE_DEFAULTS``      - canonical value for every argument
2. ``BACKEND_DEFAULTS``   - genuine per-backbone differences
3. ``METHOD_DEFAULTS``    - per-method recipe values
4. YAML config            - ``flowbp/config/flowbp.py``
5. explicit CLI flag

``BASE_DEFAULTS`` follows the values that all 21 shipped configs in
``configs/final/**`` agree on, so an ad-hoc run without a config now behaves
like the paper recipe instead of the stale FLUX.1 defaults.
"""

from __future__ import annotations

from typing import Any

BACKENDS = ("flux1", "flux2", "sd35")

FLOWBP_METHOD_CHOICES = [
    "leapalign",
    "refl",
    "draft_lv",
    "drtune",
    "flowbp_sparse",
    "flowbp_bridge",
    "flowbp_lagrange",
]


# --------------------------------------------------------------------------
# Canonical defaults
# --------------------------------------------------------------------------
BASE_DEFAULTS: dict[str, Any] = {
    # entrypoint / trainer selection
    "config": None,
    "trainer": "leapalign",
    # data
    "data_json_path": None,
    "dataloader_num_workers": 4,
    "num_latent_t": 1,
    "cfg": 0.0,
    "caption_key": "caption",
    # model
    "pretrained_model_name_or_path": None,
    "gradient_checkpointing": False,
    "master_weight_type": "bf16",
    "use_ema": False,
    "ema_decay": 0.995,
    # train loop
    "seed": None,
    "output_dir": None,
    "train_batch_size": 8,
    "train_sp_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "max_train_steps": None,
    "checkpointing_steps": 500,
    "resume_from_checkpoint": None,
    "save_optimizer_state": False,
    "allow_tf32": False,
    "debug": False,
    # connector diagnostics
    "connector_dump_interval": 0,
    "connector_dump_dir": None,
    "connector_wandb_interval": 0,
    "connector_wandb_num_samples": 2,
    # optimizer / schedule
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "lr_scheduler": "constant_with_warmup",
    "lr_warmup_steps": 10,
    "lr_num_cycles": 1,
    "lr_power": 1.0,
    "max_grad_norm": 1.0,
    # distributed
    "sp_size": 1,
    "fsdp_sharding_strategy": "full",
    "selective_checkpointing": 1.0,
    "use_cpu_offload": False,
    # sampling
    "h": 512,
    "w": 512,
    "sampling_steps": 25,
    "rollout_steps": 25,
    "sampler_seed": None,
    "cfg_guidance": 4.0,
    "cfg_detach_neg": True,
    "cfg_grad_norm_compensate": True,
    # reward
    "use_hpsv2": False,
    "reward_ckpt_path": "",
    "loss_grad_scale": 1.0,
    "loss_relu_clip": 0.55,
    # shared FlowBP knobs
    "select_idx_seed": None,
    "min_idx": 1,
    # ``None`` means "derive as sampling_steps + 1"; see resolve_derived_args.
    "max_idx": None,
    "train_step_tail_ratio": 1.0,
    "alpha": 0.3,
    "tau": 0.1,
    "num_active_steps": 3,
    # per-sample trajectory filtering (0/None disables)
    "traj_filter_dj_max": 0.0,
    "traj_filter_d0_max": 0.0,
    # j-k index sampling
    # ReFL
    "refl_last_n_steps": 11,
    # DRTune
    "drtune_num_train_steps": 3,
    "drtune_early_stop_steps": None,
    "drtune_early_stop_ratio": 0.4,
    # DRaFT-LV
    "draft_lv_num_noised_samples": 2,
    # FlowBP-Lagrange
    "flowbp_lagrange_connector_order": 3,
    "flowbp_lagrange_detach_history": True,
    "flowbp_lagrange_grad_support_mode": "midpoint",
    "flowbp_lagrange_grad_support_scale": 0.25,
    "flowbp_lagrange_max_active_supports": 2,
    "flowbp_lagrange_weight_scheme": "lagrange",
    "flowbp_lagrange_anchor_lambda": 1.0,
    "debug_flowbp_lagrange_connector": False,
    # FLUX.2 text stack
    "max_sequence_length": 512,
    "text_encoder_out_layers": [9, 18, 27],
    "use_precomputed_embeds": False,
    # logging
    "project": "flowbp",
    "run_name": None,
    # evaluation
    "evaluation_interval": 50,
    "eval_prompts_file": "./assets/eval_prompts.txt",
    "eval_num_imgs_per_prompt": 4,
    "eval_seed": 42,
    "eval_num_steps": 50,
    "eval_guidance_scale": 4.0,
    "eval_reward_fn": ["hpsv2"],
    "eval_reward_ckpt_path": "",
    "eval_hpsv3_config_path": "",
    "eval_hpsv3_checkpoint_path": "",
    "evaluation_prompts": [
        "A man lying down on green grass, gazing at the stars during an evening at a countryside villa",
        "A grey tabby cat with yellow eyes rests on a weathered wooden log under bright sunlight.",
        "A photo of four giraffes",
        "A photo of a red rabbit on the left of a white stop sign",
    ],
}


# --------------------------------------------------------------------------
# Genuine per-backbone differences
# --------------------------------------------------------------------------
BACKEND_DEFAULTS: dict[str, dict[str, Any]] = {
    "flux1": {
        "pretrained_model_name_or_path": "data/flux",
        "cfg_guidance": 4.0,
        "eval_num_steps": 50,
        "eval_guidance_scale": 4.0,
        "project": "flux_rlhf",
    },
    "flux2": {
        "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-klein-base-4B",
        "cfg_guidance": 4.0,
        "eval_num_steps": 50,
        "eval_guidance_scale": 4.0,
        "project": "flux2_rlhf",
        # FLUX.2-Klein 9B does not fit the FLUX.1 per-device batch.
        "train_batch_size": 4,
        "train_sp_batch_size": 4,
    },
    "sd35": {
        "pretrained_model_name_or_path": "data/sd3.5_medium",
        # SD3.5-M uses real CFG at a lower guidance scale than the FLUX models.
        "cfg_guidance": 3.5,
        "eval_num_steps": 40,
        "eval_guidance_scale": 3.5,
        "project": "sd35_rlhf",
    },
}


# --------------------------------------------------------------------------
# Per-method recipe values
# --------------------------------------------------------------------------
METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "leapalign": {},
    "refl": {},
    "draft_lv": {},
    "drtune": {},
    "flowbp_sparse": {
        "num_active_steps": 3,
    },
    "flowbp_bridge": {
        # Bridge splits its budget across two segments, so it needs a larger
        # budget than Sparse.
        "num_active_steps": 4,
        "alpha": 0.5,
    },
    "flowbp_lagrange": {
        "alpha": 0.1,
    },
}


# --------------------------------------------------------------------------
# Backward-compatible argument names
# --------------------------------------------------------------------------
# Old name -> canonical name. Old names stay accepted on the CLI and in YAML.
ARG_ALIASES: dict[str, str] = {
    # The Sparse-prefixed knobs are also used by FlowBP-Bridge.
    "flowbp_sparse_num_active_steps": "num_active_steps",
    # Long-standing typo, baked into every shipped config.
    "fsdp_sharding_startegy": "fsdp_sharding_strategy",
    # FLUX.2 spelled per-sample filtering differently from SD3.5.
    "clip_dj_threshold": "traj_filter_dj_max",
    "clip_d0_threshold": "traj_filter_d0_max",
}

# ``clip_d0`` was a separate on/off switch guarding ``clip_d0_threshold``.
# It has no canonical counterpart: a positive ``traj_filter_d0_max`` is the
# switch. Handled by resolve_derived_args.
LEGACY_SWITCHES = ("clip_d0",)


def canonical_name(name: str) -> str:
    """Map a possibly-legacy argument name to its canonical name."""
    return ARG_ALIASES.get(name, name)


def defaults_for(backend: str, trainer: str | None = None) -> dict[str, Any]:
    """Merge base + backend + method defaults into one flat mapping."""
    if backend not in BACKEND_DEFAULTS:
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of {sorted(BACKEND_DEFAULTS)}"
        )
    merged = dict(BASE_DEFAULTS)
    merged.update(BACKEND_DEFAULTS[backend])
    if trainer:
        merged.update(METHOD_DEFAULTS.get(trainer, {}))
    return merged


def resolve_derived_args(args) -> None:
    """Fill in values that depend on other arguments. Mutates ``args``.

    Applied after parsing so it sees the final merged configuration.
    """
    # ``max_idx`` is an exclusive reverse-index bound and must track the
    # rollout length. Leaving it stale silently shrinks the j-k sampling range.
    total_steps = int(getattr(args, "rollout_steps", None) or getattr(args, "sampling_steps", 25) or 25)
    if getattr(args, "max_idx", None) is None:
        args.max_idx = total_steps + 1

    # Legacy FLUX.2 switch: --no-clip_d0 forced d0 filtering off regardless of
    # the threshold. Tri-state, so only an explicit False overrides.
    if getattr(args, "clip_d0", None) is False:
        args.traj_filter_d0_max = 0.0

    if getattr(args, "run_name", None) in (None, ""):
        args.run_name = str(getattr(args, "trainer", None) or "flowbp")
