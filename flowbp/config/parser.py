"""One argument parser for all three FlowBP backbones.

``train_flowbp_flux.py`` and ``train_flowbp_flux2.py`` used to carry two
independently maintained parsers that had drifted apart on 22 shared arguments
and disagreed on which arguments existed at all (``--eval_num_steps`` and
``--eval_guidance_scale`` were FLUX.2-only, so FLUX.1 and SD3.5 accepted them
from YAML but rejected them on the command line).

Defaults come from :mod:`flowbp.config.defaults`; this module only owns the
argument *shapes* (type, nargs, choices, help) plus the legacy spellings that
stay accepted for backward compatibility.
"""

from __future__ import annotations

import argparse
from typing import Any

from flowbp.config.defaults import (
    ARG_ALIASES,
    FLOWBP_METHOD_CHOICES,
    defaults_for,
    resolve_derived_args,
)


def _legacy_flags(dest: str) -> list[str]:
    """Old CLI spellings that should still resolve to ``dest``."""
    return [f"--{old}" for old, new in ARG_ALIASES.items() if new == dest]


def _add(parser: argparse.ArgumentParser, dest: str, **kwargs: Any) -> None:
    """Register ``--dest`` plus any legacy aliases, sharing one destination."""
    flags = [f"--{dest}", *_legacy_flags(dest)]
    parser.add_argument(*flags, dest=dest, **kwargs)


def build_parser(backend: str) -> argparse.ArgumentParser:
    """Build the FlowBP parser for one backbone.

    Defaults are the merged base+backend values. Method-specific defaults are
    layered on later by :func:`parse_args`, once ``--trainer`` is known.
    """
    d = defaults_for(backend)
    p = argparse.ArgumentParser(description=f"FlowBP {backend} fine-tuning entrypoint.")

    # ---- entrypoint -------------------------------------------------------
    _add(p, "config", type=str, default=d["config"],
         help="Path to a FlowBP YAML config. CLI flags override values from this file.")
    _add(p, "trainer", type=str, default=d["trainer"], choices=FLOWBP_METHOD_CHOICES,
         help="FlowBP method or direct-gradient baseline to run.")

    # ---- data -------------------------------------------------------------
    _add(p, "data_json_path", type=str, default=d["data_json_path"],
         help="Index JSON produced by flowbp/data_preprocess/*.")
    _add(p, "dataloader_num_workers", type=int, default=d["dataloader_num_workers"])
    _add(p, "num_latent_t", type=int, default=d["num_latent_t"])
    _add(p, "cfg", type=float, default=d["cfg"],
         help="Probability of dropping the prompt embedding during training.")
    _add(p, "caption_key", type=str, default=d["caption_key"],
         help="JSON key holding the prompt string in each dataset entry.")

    # ---- model ------------------------------------------------------------
    _add(p, "pretrained_model_name_or_path", type=str, default=d["pretrained_model_name_or_path"])
    _add(p, "gradient_checkpointing", action="store_true", default=d["gradient_checkpointing"])
    _add(p, "master_weight_type", type=str, default=d["master_weight_type"], choices=("fp32", "bf16"))
    _add(p, "use_ema", action="store_true", default=d["use_ema"])
    _add(p, "ema_decay", type=float, default=d["ema_decay"])

    # ---- train loop -------------------------------------------------------
    _add(p, "seed", type=int, default=d["seed"])
    _add(p, "output_dir", type=str, default=d["output_dir"])
    _add(p, "train_batch_size", type=int, default=d["train_batch_size"])
    _add(p, "train_sp_batch_size", type=int, default=d["train_sp_batch_size"])
    _add(p, "gradient_accumulation_steps", type=int, default=d["gradient_accumulation_steps"])
    _add(p, "max_train_steps", type=int, default=d["max_train_steps"])
    _add(p, "checkpointing_steps", type=int, default=d["checkpointing_steps"])
    _add(p, "resume_from_checkpoint", type=str, default=d["resume_from_checkpoint"],
         help='Checkpoint directory to resume from, or "latest" to pick the highest step under output_dir.')
    _add(p, "save_optimizer_state", action="store_true", default=d["save_optimizer_state"],
         help="Also write AdamW state into each checkpoint so resume is exact. Roughly doubles checkpoint size.")
    _add(p, "allow_tf32", action="store_true", default=d["allow_tf32"])
    _add(p, "debug", action="store_true", default=d["debug"])

    # ---- connector diagnostics -------------------------------------------
    _add(p, "connector_dump_interval", type=int, default=d["connector_dump_interval"],
         help="Save connector latent pairs every N steps; 0 disables.")
    _add(p, "connector_dump_dir", type=str, default=d["connector_dump_dir"])
    _add(p, "connector_wandb_interval", type=int, default=d["connector_wandb_interval"])
    _add(p, "connector_wandb_num_samples", type=int, default=d["connector_wandb_num_samples"])

    # ---- optimizer / schedule --------------------------------------------
    _add(p, "learning_rate", type=float, default=d["learning_rate"])
    _add(p, "weight_decay", type=float, default=d["weight_decay"])
    _add(p, "lr_scheduler", type=str, default=d["lr_scheduler"])
    _add(p, "lr_warmup_steps", type=int, default=d["lr_warmup_steps"])
    _add(p, "lr_num_cycles", type=int, default=d["lr_num_cycles"])
    _add(p, "lr_power", type=float, default=d["lr_power"])
    _add(p, "max_grad_norm", type=float, default=d["max_grad_norm"])

    # ---- distributed ------------------------------------------------------
    _add(p, "sp_size", type=int, default=d["sp_size"], help="Sequence-parallel group size.")
    _add(p, "fsdp_sharding_strategy", type=str, default=d["fsdp_sharding_strategy"],
         choices=("full", "hybrid_full", "none", "hybrid_zero2"))
    _add(p, "selective_checkpointing", type=float, default=d["selective_checkpointing"])
    _add(p, "use_cpu_offload", action="store_true", default=d["use_cpu_offload"])

    # ---- sampling ---------------------------------------------------------
    _add(p, "h", type=int, default=d["h"])
    _add(p, "w", type=int, default=d["w"])
    _add(p, "sampling_steps", type=int, default=d["sampling_steps"])
    _add(p, "rollout_steps", type=int, default=d["rollout_steps"])
    _add(p, "sampler_seed", type=int, default=d["sampler_seed"])
    _add(p, "cfg_guidance", type=float, default=d["cfg_guidance"])
    _add(p, "cfg_detach_neg", action=argparse.BooleanOptionalAction, default=d["cfg_detach_neg"],
         help="Detach the negative-prompt CFG branch (memory-efficient). Ignored by FLUX.1.")
    _add(p, "cfg_grad_norm_compensate", action=argparse.BooleanOptionalAction,
         default=d["cfg_grad_norm_compensate"],
         help="Divide the backward loss by cfg_guidance to undo CFG gradient amplification.")

    # ---- reward -----------------------------------------------------------
    _add(p, "use_hpsv2", action="store_true", default=d["use_hpsv2"])
    _add(p, "reward_ckpt_path", type=str, default=d["reward_ckpt_path"],
         help="Directory holding the HPSv2 training reward checkpoints.")
    _add(p, "loss_grad_scale", type=float, default=d["loss_grad_scale"])
    _add(p, "loss_relu_clip", type=float, default=d["loss_relu_clip"])

    # ---- shared FlowBP knobs ---------------------------------------------
    _add(p, "select_idx_seed", type=int, default=d["select_idx_seed"])
    _add(p, "min_idx", type=int, default=d["min_idx"])
    _add(p, "max_idx", type=int, default=d["max_idx"],
         help="Exclusive reverse-index bound for j-k sampling. Defaults to rollout_steps + 1.")
    _add(p, "train_step_tail_ratio", type=float, default=d["train_step_tail_ratio"],
         help="Expose only this final fraction of rollout steps to gradient-carrying active steps.")
    _add(p, "alpha", type=float, default=d["alpha"],
         help="Bridge coupling coefficient for LeapAlign and the bridged FlowBP variants.")
    _add(p, "tau", type=float, default=d["tau"],
         help="Residual floor for trajectory-similarity weighting.")
    _add(p, "num_active_steps", type=int, default=d["num_active_steps"],
         help="Active steps re-forwarded with grad per rollout (FlowBP-Sparse and FlowBP-Bridge).")
    _add(p, "traj_filter_dj_max", type=float, default=d["traj_filter_dj_max"],
         help="Drop samples whose connector residual d_j exceeds this. 0 disables.")
    _add(p, "traj_filter_d0_max", type=float, default=d["traj_filter_d0_max"],
         help="Drop samples whose endpoint residual d_0 exceeds this. 0 disables.")
    p.add_argument("--clip_d0", action=argparse.BooleanOptionalAction, default=None,
                   help="Deprecated FLUX.2 switch; --no-clip_d0 forces d_0 filtering off.")


    # ---- baselines --------------------------------------------------------
    _add(p, "refl_last_n_steps", type=int, default=d["refl_last_n_steps"])
    _add(p, "drtune_num_train_steps", type=int, default=d["drtune_num_train_steps"])
    _add(p, "drtune_early_stop_steps", type=int, default=d["drtune_early_stop_steps"])
    _add(p, "drtune_early_stop_ratio", type=float, default=d["drtune_early_stop_ratio"])
    _add(p, "draft_lv_num_noised_samples", type=int, default=d["draft_lv_num_noised_samples"])

    # ---- FlowBP-Lagrange --------------------------------------------------
    _add(p, "flowbp_lagrange_connector_order", type=int, default=d["flowbp_lagrange_connector_order"])
    _add(p, "flowbp_lagrange_detach_history", action=argparse.BooleanOptionalAction,
         default=d["flowbp_lagrange_detach_history"])
    _add(p, "flowbp_lagrange_grad_support_mode", type=str,
         default=d["flowbp_lagrange_grad_support_mode"],
         choices=("none", "start", "midpoint", "all"))
    _add(p, "flowbp_lagrange_grad_support_scale", type=float,
         default=d["flowbp_lagrange_grad_support_scale"])
    _add(p, "flowbp_lagrange_max_active_supports", type=int,
         default=d["flowbp_lagrange_max_active_supports"])
    _add(p, "flowbp_lagrange_weight_scheme", type=str, default=d["flowbp_lagrange_weight_scheme"],
         choices=("lagrange", "uniform", "adams_bashforth"))
    _add(p, "flowbp_lagrange_anchor_lambda", type=float, default=d["flowbp_lagrange_anchor_lambda"],
         help="1.0 = Lagrange connector, 0.0 = Euler/LeapAlign jump, in between blends the two.")
    _add(p, "debug_flowbp_lagrange_connector", action="store_true",
         default=d["debug_flowbp_lagrange_connector"])

    # ---- FLUX.2 text stack ------------------------------------------------
    _add(p, "max_sequence_length", type=int, default=d["max_sequence_length"])
    _add(p, "text_encoder_out_layers", type=int, nargs="+", default=d["text_encoder_out_layers"])
    _add(p, "use_precomputed_embeds", action=argparse.BooleanOptionalAction,
         default=d["use_precomputed_embeds"])

    # ---- logging ----------------------------------------------------------
    _add(p, "project", type=str, default=d["project"])
    _add(p, "run_name", type=str, default=d["run_name"])

    # ---- evaluation -------------------------------------------------------
    _add(p, "evaluation_interval", type=int, default=d["evaluation_interval"],
         help="Run online evaluation every N steps; 0 disables.")
    _add(p, "eval_prompts_file", type=str, default=d["eval_prompts_file"])
    _add(p, "eval_num_imgs_per_prompt", type=int, default=d["eval_num_imgs_per_prompt"])
    _add(p, "eval_seed", type=int, default=d["eval_seed"])
    _add(p, "eval_num_steps", type=int, default=d["eval_num_steps"])
    _add(p, "eval_guidance_scale", type=float, default=d["eval_guidance_scale"])
    _add(p, "eval_reward_fn", type=str, nargs="+", default=d["eval_reward_fn"])
    _add(p, "eval_reward_ckpt_path", type=str, default=d["eval_reward_ckpt_path"])
    _add(p, "eval_hpsv3_config_path", type=str, default=d["eval_hpsv3_config_path"])
    _add(p, "eval_hpsv3_checkpoint_path", type=str, default=d["eval_hpsv3_checkpoint_path"])
    _add(p, "evaluation_prompts", type=str, nargs="+", default=d["evaluation_prompts"])

    return p


def parse_args(backend: str, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse FlowBP arguments with the full precedence chain.

    base < backend < method < YAML config < explicit CLI flag.
    """
    from flowbp.config.flowbp import load_config_defaults, normalize_trainer_name

    parser = build_parser(backend)

    # Probe for --trainer / --config so method defaults and YAML can be layered
    # in before the real parse, keeping explicit CLI flags on top.
    probe, _ = parser.parse_known_args(argv)
    trainer = normalize_trainer_name(probe.trainer)

    layered = defaults_for(backend, trainer)
    layered["trainer"] = trainer
    if probe.config:
        layered.update(load_config_defaults(probe.config))
    parser.set_defaults(**layered)

    args = parser.parse_args(argv)
    args.trainer = normalize_trainer_name(args.trainer)
    resolve_derived_args(args)
    return args
