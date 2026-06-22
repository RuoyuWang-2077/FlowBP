
"""Entrypoint for FlowBP training on FLUX.2.

Parallels :mod:`flowbp.train_flowbp_flux` but loads the Flux2 trainer
classes. Shares the same YAML config schema; FLUX.2-specific knobs live under
the optional ``flux2:`` section.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flowbp.config.flowbp import (
    FLOWBP_TRAINER_CHOICES,
    load_config_defaults,
    normalize_trainer_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a FlowBP YAML config. CLI flags override values from this file.",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="leapalign",
        choices=FLOWBP_TRAINER_CHOICES,
        help=(
            "Training algorithm to run (FLUX.2 supports leapalign, "
            "flowbp_sparse, flowbp_bridge, flowbp_lagrange, refl, drtune, "
            "and draft_lv)."
        ),
    )

    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, default=None)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="Kept for parity with the Flux1 trainer; FLUX.2 ignores it.",
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="caption",
        help="JSON key that holds the prompt string in each dataset entry.",
    )

    # model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
    )

    # diffusion / training
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--connector_dump_interval", type=int, default=0)
    parser.add_argument("--connector_dump_dir", type=str, default=None)
    parser.add_argument("--connector_wandb_interval", type=int, default=0)
    parser.add_argument("--connector_wandb_num_samples", type=int, default=2)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # optimizer / scheduler
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_cpu_offload", action="store_true")

    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--train_sp_batch_size", type=int, default=4)
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--master_weight_type", type=str, default="bf16")

    # sampling
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=512)
    # Defaults target the undistilled FLUX.2 training recipe.
    parser.add_argument("--sampling_steps", type=int, default=25)
    parser.add_argument("--rollout_steps", type=int, default=25)
    parser.add_argument("--sampler_seed", type=int, default=None)
    parser.add_argument("--cfg_guidance", type=float, default=4.0)

    # reward
    parser.add_argument("--use_hpsv2", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--project", type=str, default="flux2_rlhf")
    parser.add_argument("--run_name", type=str, default="flux2_leapalign")

    # leapalign parameters
    parser.add_argument("--select_idx_seed", type=int, default=None)
    parser.add_argument("--min_idx", type=int, default=1)
    # max_idx = sampling_steps + 1 keeps reverse-index sampling in range
    # (25 steps -> max_idx=26). Keep these in sync when changing sampling_steps.
    parser.add_argument("--max_idx", type=int, default=26)
    parser.add_argument(
        "--train_step_tail_ratio",
        type=float,
        default=1.0,
        help=(
            "For FlowBP variants, expose only this final fraction of rollout "
            "steps to gradient-carrying active-step sampling. 1.0 keeps the "
            "full rollout."
        ),
    )
    parser.add_argument("--loss_grad_scale", type=float, default=1.0)
    parser.add_argument("--loss_relu_clip", type=float, default=0.55)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--cfg_grad_norm_compensate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to divide the backward loss by ``cfg_guidance`` to undo "
            "the gradient amplification introduced by the detached negative "
            "branch in ``_cfg_aware_forward``. Default True (recommended). "
            "Pass --no-cfg_grad_norm_compensate to use the raw CFG-scaled gradient; "
            "effective gradient magnitude then scales linearly with cfg, "
            "equivalent to setting LR = base_lr * cfg_guidance."
        ),
    )
    parser.add_argument(
        "--cfg_detach_neg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to detach the negative-prompt branch in "
            "``_cfg_aware_forward``. Default True (memory-efficient, gradient "
            "flows only through the conditional branch). Pass "
            "--no-cfg_detach_neg to compute pred_neg with grad enabled; the "
            "effective gradient becomes the mathematically standard CFG "
            "gradient ``(1-cfg)*d(pred_neg)/dθ + cfg*d(pred_cond)/dθ``. This "
            "costs roughly 1.8x activation memory due to the second backward "
            "graph."
        ),
    )

    # j-k index sampling (shared by FlowBP-Bridge/Lagrange trainers)
    parser.add_argument(
        "--jk_sampling_mode",
        type=str,
        default="uniform",
        choices=("uniform", "dirichlet", "midpoint", "midpoint_j"),
        help=(
            "How to draw the (j, k) step indices for FlowBP bridge/lagrange rollouts. "
            "'uniform' is the original randperm sampler (high variance); "
            "'dirichlet' partitions [min_idx, max_idx] into 3 segments via a "
            "Dirichlet(alpha_a, alpha_b, alpha_c) and shrinks the variance "
            "of the (k, j) placement by alpha; "
            "'midpoint' samples k uniformly and places j at the midpoint of "
            "[k, end]; 'midpoint_j' samples j and places k at the midpoint of "
            "[start, j]."
        ),
    )
    parser.add_argument(
        "--jk_dirichlet_alpha",
        type=float,
        default=4.2,
        help=(
            "Symmetric Dirichlet concentration when "
            "jk_sampling_mode='dirichlet'. Larger alpha -> tighter, less "
            "variant (k, j) placement. alpha=7.4 cuts the variance by ~7x "
            "vs the uniform sampler."
        ),
    )
    parser.add_argument(
        "--jk_dirichlet_alpha_a",
        type=float,
        default=None,
        help=(
            "Per-axis concentration for the segment that controls j_idx. "
            "Falls back to --jk_dirichlet_alpha when None."
        ),
    )
    parser.add_argument(
        "--jk_dirichlet_alpha_b",
        type=float,
        default=None,
        help=(
            "Per-axis concentration for the gap segment (between k and j). "
            "Falls back to --jk_dirichlet_alpha when None."
        ),
    )
    parser.add_argument(
        "--jk_dirichlet_alpha_c",
        type=float,
        default=None,
        help=(
            "Per-axis concentration for the segment that controls k_idx. "
            "Falls back to --jk_dirichlet_alpha when None."
        ),
    )
    parser.add_argument(
        "--jk_dirichlet_max_j_rev",
        type=int,
        default=None,
        help=(
            "Soft cap on j_rev for jk_sampling_mode='dirichlet'. Overflow "
            "above the cap is moved to the j->end segment, preserving the "
            "sampled k_rev - j_rev gap while preventing j from landing too "
            "close to the noise side."
        ),
    )

    # refl parameters
    parser.add_argument(
        "--refl_last_n_steps",
        type=int,
        default=11,
        help=(
            "For ReFL, randomly back-prop one denoising step among the final "
            "N rollout steps. Default = 11 (fine-tune the tail 11/25)."
        ),
    )

    # FlowBP-Lagrange parameters
    parser.add_argument("--flowbp_lagrange_connector_order", type=int, default=3)
    parser.add_argument(
        "--flowbp_lagrange_detach_history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--flowbp_lagrange_grad_support_mode",
        type=str,
        default="midpoint",
        choices=("none", "start", "midpoint", "all"),
    )
    parser.add_argument("--flowbp_lagrange_grad_support_scale", type=float, default=0.25)
    parser.add_argument("--flowbp_lagrange_max_active_supports", type=int, default=2)
    parser.add_argument(
        "--debug_flowbp_lagrange_connector",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--flowbp_lagrange_weight_scheme",
        type=str,
        default="lagrange",
        choices=("lagrange", "uniform"),
    )
    parser.add_argument("--flowbp_lagrange_grad_rescale", type=float, default=0.5)
    parser.add_argument("--flowbp_lagrange_anchor_lambda", type=float, default=1.0)
    parser.add_argument(
        "--clip_dj_threshold",
        type=float,
        default=0.0,
        help=(
            "If > 0, drop samples whose connector d_j exceeds the threshold "
            "from the reward loss (their gradient contribution is zeroed). "
            "Forward still runs for those samples; only the reward signal is "
            "masked. Set 0 to disable (default)."
        ),
    )
    parser.add_argument(
        "--clip_d0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Switch: when on, drop samples whose one-step pred_x0 error d_0 "
            "exceeds --clip_d0_threshold from the reward loss (their "
            "gradient contribution is zeroed). Forward still runs for those "
            "samples; only the reward signal is masked. ANDs with the "
            "--clip_dj_threshold mask if both are active. Default: off."
        ),
    )
    parser.add_argument(
        "--clip_d0_threshold",
        type=float,
        default=0.2,
        help=(
            "Per-sample d_0 threshold used when --clip_d0 is enabled. "
            "Default: 0.2."
        ),
    )

    # FlowBP-Sparse trainer
    parser.add_argument(
        "--flowbp_sparse_num_active_steps",
        type=int,
        default=3,
        help=(
            "FlowBP-Sparse trainer: number of active sampling steps to "
            "re-forward with grad per rollout (default 3)."
        ),
    )
    parser.add_argument(
        "--flowbp_sparse_late_bias",
        type=float,
        default=2.0,
        help=(
            "FlowBP-Sparse trainer: power-law bias for active step "
            "sampling. w_i ∝ (i+1)^bias. 0 = uniform, larger = more weight "
            "on late (close-to-clean) indices."
        ),
    )
    parser.add_argument(
        "--flowbp_sparse_grad_rescale",
        type=float,
        default=0.0,
        help=(
            "FlowBP-Sparse trainer: gradient rescale strength (>= 0). "
            "Formula: factor = 1 + gr * (Σ|Δσ_all|/Σ|Δσ_active| - 1). "
            "0 = raw small per-step gradient, 1 = full Euler/ReFL-magnitude "
            "gradient, >1 = push beyond full Euler (each active velocity gets "
            "more than its fair share). Use values >1 when even gr=1 yields "
            "small grad_norm under strong late-bias sampling."
        ),
    )
    # DRTune trainer
    parser.add_argument(
        "--drtune_num_train_steps",
        type=int,
        default=3,
        help=(
            "DRTune: number of equally spaced denoising steps to train per "
            "rollout. Paper default for 25-step SDXL/FLUX.2-style sampling is K=3."
        ),
    )
    parser.add_argument(
        "--drtune_early_stop_steps",
        type=int,
        default=None,
        help=(
            "DRTune: maximum reverse-index early-stop step m. None uses "
            "--drtune_early_stop_ratio * rollout_steps; 0 disables early stop."
        ),
    )
    parser.add_argument(
        "--drtune_early_stop_ratio",
        type=float,
        default=0.4,
        help=(
            "DRTune: default early-stop ratio used when "
            "--drtune_early_stop_steps is None. Paper suggests m ~= 0.4T."
        ),
    )

    # DRaFT-LV trainer
    parser.add_argument(
        "--draft_lv_num_noised_samples",
        type=int,
        default=2,
        help=(
            "DRaFT-LV: number of extra noised last-step examples. "
            "The paper default is n=2."
        ),
    )

    # FLUX.2-specific
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Max Qwen3 tokenizer context (FLUX.2 default is 512).",
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[9, 18, 27],
        help="Qwen3 hidden-state layer indices fused into prompt_embeds (3 layers -> 3*hidden_dim=7680).",
    )
    parser.add_argument(
        "--use_precomputed_embeds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, load Qwen3 prompt embeddings precomputed by "
            "flowbp/data_preprocess/preprocess_flux2_embedding.py instead of "
            "running the text encoder on the fly. data_json_path should point "
            "to the resulting captions2embed.json."
        ),
    )

    parser.add_argument("--debug", action="store_true", default=False)

    # Online evaluation.
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=0,
        help="Run online eval every N training steps; 0 disables.",
    )
    parser.add_argument(
        "--eval_prompts_file",
        type=str,
        default="./assets/eval_prompts.txt",
        help="Prompt list (one per line) for online eval.",
    )
    parser.add_argument(
        "--eval_num_imgs_per_prompt",
        type=int,
        default=1,
        help="Number of images sampled per eval prompt.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Random seed for eval image sampling.",
    )
    parser.add_argument(
        "--eval_num_steps",
        type=int,
        default=50,
        help="Number of denoising steps for eval sampling.",
    )
    parser.add_argument(
        "--eval_guidance_scale",
        type=float,
        default=4.0,
        help="CFG scale for eval sampling (klein-base-4B default is 4.0).",
    )
    parser.add_argument(
        "--eval_reward_fn",
        type=str,
        nargs="+",
        default=["hpsv2", "hpsv3"],
        help=(
            "Reward metrics to compute during eval, e.g. "
            "clipscore aesthetic hpsv2 hpsv3 pickscore imagereward."
        ),
    )
    parser.add_argument(
        "--eval_reward_ckpt_path",
        type=str,
        default="",
        help="Directory containing eval reward model checkpoints.",
    )
    parser.add_argument(
        "--eval_hpsv3_config_path",
        type=str,
        default="",
        help=(
            "Optional HPSv3 inferencer YAML config path. When empty, the "
            "online eval runner auto-discovers a config under "
            "<eval_reward_ckpt_path>/HPSv3/ and falls back to "
            "assets/eval/hpsv3/HPSv3_7B.yaml (which points at the local "
            "Qwen2-VL-7B-Instruct mirror)."
        ),
    )
    parser.add_argument(
        "--eval_hpsv3_checkpoint_path",
        type=str,
        default="",
        help=(
            "Optional HPSv3 .safetensors / .pt checkpoint path override. "
            "Defaults to <eval_reward_ckpt_path>/HPSv3/HPSv3.safetensors "
            "when present."
        ),
    )

    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    probe_args, _ = parser.parse_known_args()
    if probe_args.config:
        parser.set_defaults(**load_config_defaults(probe_args.config))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Import FLUX.2 trainers directly so dependency errors surface with a
    # clear stack trace.
    from flowbp.trainers.flux2.leapalign import LeapAlignFlux2Trainer
    from flowbp.trainers.flux2.flowbp_lagrange import (
        FlowBPLagrangeFlux2Trainer,
    )
    from flowbp.trainers.flux2.refl import LeapAlignReFLFlux2Trainer
    from flowbp.trainers.flux2.flowbp_sparse import (
        FlowBPSparseFlux2Trainer,
    )
    from flowbp.trainers.flux2.flowbp_bridge import (
        FlowBPBridgeFlux2Trainer,
    )
    from flowbp.trainers.flux2.drtune import LeapAlignDRTuneFlux2Trainer
    from flowbp.trainers.flux2.draft_lv import LeapAlignDRaFTLVFlux2Trainer

    trainer_map = {
        "leapalign": LeapAlignFlux2Trainer,
        "flowbp_lagrange": FlowBPLagrangeFlux2Trainer,
        "refl": LeapAlignReFLFlux2Trainer,
        "flowbp_sparse": FlowBPSparseFlux2Trainer,
        "flowbp_bridge": FlowBPBridgeFlux2Trainer,
        "drtune": LeapAlignDRTuneFlux2Trainer,
        "draft_lv": LeapAlignDRaFTLVFlux2Trainer,
    }
    trainer_map[normalize_trainer_name(args.trainer)](args).train()


if __name__ == "__main__":
    main()
