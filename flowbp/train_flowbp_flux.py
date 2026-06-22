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
        help="Training algorithm to run.",
    )

    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, default=None)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # Model components.
    parser.add_argument("--pretrained_model_name_or_path", type=str)

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--cfg", type=float, default=0.0)

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--connector_dump_interval",
        type=int,
        default=0,
        help="Save connector latent pairs every N training steps; 0 disables.",
    )
    parser.add_argument(
        "--connector_dump_dir",
        type=str,
        default=None,
        help="Directory for connector latent dumps; defaults to output_dir/connector_dumps.",
    )
    parser.add_argument(
        "--connector_wandb_interval",
        type=int,
        default=0,
        help="Log decoded connector examples to wandb every N training steps; 0 disables.",
    )
    parser.add_argument(
        "--connector_wandb_num_samples",
        type=int,
        default=2,
        help="Number of samples to decode per connector wandb logging step.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=25,
        help="Number of rollout denoising steps for ReFL training.",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model",
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Enable Exponential Moving Average of model weights."
    )
    parser.add_argument(
        "--project",
        type=str,
        default="flux_rlhf",
        help="project name for wandb",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="flux_grpo",
        help="run name for wandb",
    )
    parser.add_argument(
        "--cfg_guidance",
        type=float,
        default=3.5,
        help="guidance scale for cfg",
    )
    
    # refl parameters:
    parser.add_argument(
        "--select_idx_seed",
        type=int,
        default=None,
        help="select index seed for refl",
    )
    parser.add_argument(
        "--min_idx",
        type=int,
        default=1,
        help="min index for refl",
    )
    parser.add_argument(
        "--max_idx",
        type=int,
        default=10,
        help="max index for refl",
    )
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
    parser.add_argument(
        "--loss_grad_scale",
        type=float,
        default=1.0,
        help="reward loss coefficient",
    )
    parser.add_argument(
        "--loss_relu_clip",
        type=float,
        default=0.7,
        help="relu clip for reward loss",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Bridge coupling coefficient for LeapAlign and FlowBP bridged variants.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Residual floor for trajectory similarity weighting.",
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
            "Dirichlet(alpha_a, alpha_b, alpha_c) and shrinks variance by "
            "alpha; 'midpoint' samples k uniformly and places j at the "
            "midpoint of [k, end]; 'midpoint_j' samples j and places k at "
            "the midpoint of [start, j]."
        ),
    )
    parser.add_argument(
        "--jk_dirichlet_alpha",
        type=float,
        default=4.2,
        help=(
            "Symmetric Dirichlet concentration when "
            "jk_sampling_mode='dirichlet'. Larger alpha -> tighter, less "
            "variant (k, j) placement."
        ),
    )
    parser.add_argument(
        "--jk_dirichlet_alpha_a",
        type=float,
        default=None,
        help=(
            "Per-axis concentration controlling j_idx position. "
            "Falls back to --jk_dirichlet_alpha when None."
        ),
    )
    parser.add_argument(
        "--jk_dirichlet_alpha_b",
        type=float,
        default=None,
        help=(
            "Per-axis concentration controlling the (k, j) gap. "
            "Falls back to --jk_dirichlet_alpha when None."
        ),
    )
    parser.add_argument(
        "--jk_dirichlet_alpha_c",
        type=float,
        default=None,
        help=(
            "Per-axis concentration controlling k_idx position. "
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

    parser.add_argument(
        "--refl_last_n_steps",
        type=int,
        default=11,
        help="For ReFL, randomly update one step among the final N rollout steps.",
    )
    parser.add_argument(
        "--flowbp_lagrange_connector_order",
        type=int,
        default=4,
        help="Number of trajectory-prior support velocities for FlowBP-Lagrange.",
    )
    parser.add_argument(
        "--flowbp_lagrange_detach_history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach cached history velocities in FlowBP-Lagrange.",
    )
    parser.add_argument(
        "--flowbp_lagrange_grad_support_mode",
        type=str,
        default="start",
        choices=("none", "start", "midpoint", "all"),
        help="Which FlowBP-Lagrange support velocities participate in backward.",
    )
    parser.add_argument(
        "--flowbp_lagrange_grad_support_scale",
        type=float,
        default=1.0,
        help="Backward gradient scale for non-start active support velocities.",
    )
    parser.add_argument(
        "--flowbp_lagrange_max_active_supports",
        type=int,
        default=0,
        help="Maximum active supports per interval; 0 means no extra limit.",
    )
    parser.add_argument(
        "--debug_flowbp_lagrange_connector",
        action="store_true",
        default=False,
        help="Enable FlowBP-Lagrange connector sanity checks.",
    )
    parser.add_argument(
        "--flowbp_lagrange_weight_scheme",
        type=str,
        default="lagrange",
        choices=("lagrange", "uniform"),
        help="Weight scheme for FlowBP-Lagrange supports.",
    )
    parser.add_argument(
        "--flowbp_lagrange_grad_rescale",
        type=float,
        default=0.0,
        help=(
            "Gradient rescaling strength for active support velocities. "
            "0=off, 1=full rescale (Σ|w_all|/Σ|w_active|), "
            "values in (0,1) interpolate."
        ),
    )
    parser.add_argument(
        "--flowbp_lagrange_anchor_lambda",
        type=float,
        default=1.0,
        help=(
            "Blend factor for anchored FlowBP-Lagrange connector. "
            "1.0 = FlowBP-Lagrange, 0.0 = Euler/LeapAlign jump, "
            "values in (0,1) use Euler-anchored Lagrange correction."
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
            "rollout. Paper default for 25-step SDXL is K=3."
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

    # evaluation
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=500,
        help="evaluation interval",
    )
    parser.add_argument(
        "--evaluation_prompts",
        type=str,
        nargs="+",
        default=[
            "A man lying down on green grass, gazing at the stars during an evening at a countryside villa",
            "A grey tabby cat with yellow eyes rests on a weathered wooden log under bright sunlight.",
            "A photo of four giraffes",
            "A photo of a red rabbit on the left of a white stop sign",
        ],
        help=(
            "A list of prompts to use for generating sample images during training. "
            "To provide multiple prompts, separate them with spaces. "
            "If a prompt contains spaces, you MUST enclose it in quotes. "
            'Example: --evaluation_prompts "a photo of a cat" "an astronaut on a horse"'
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="whether use debug mode",
    )
    parser.add_argument(
        "--eval_prompts_file",
        type=str,
        default="./assets/eval_prompts.txt",
        help="Path to eval prompts file for online HPSv2.1 evaluation.",
    )
    parser.add_argument(
        "--eval_num_imgs_per_prompt",
        type=int,
        default=4,
        help="Number of images to generate per prompt during evaluation.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Random seed for evaluation image generation.",
    )
    parser.add_argument(
        "--eval_reward_fn",
        type=str,
        nargs="+",
        default=["hpsv2", "hpsv3"],
        help=(
            "Reward metrics to compute during evaluation, e.g. "
            "clipscore aesthetic hpsv2 hpsv3 pickscore imagereward."
        ),
    )
    parser.add_argument(
        "--eval_reward_ckpt_path",
        type=str,
        default="",
        help="Directory containing evaluation reward model checkpoints.",
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
    from flowbp.trainers import (
        FlowBPSparseFluxTrainer,
        FlowBPBridgeFluxTrainer,
        LeapAlignDRaFTLVFluxTrainer,
        LeapAlignDRTuneFluxTrainer,
        LeapAlignFluxTrainer,
        FlowBPLagrangeFluxTrainer,
        LeapAlignReFLFluxTrainer,
    )

    trainer_map = {
        "leapalign": LeapAlignFluxTrainer,
        "refl": LeapAlignReFLFluxTrainer,
        "flowbp_lagrange": FlowBPLagrangeFluxTrainer,
        "flowbp_sparse": FlowBPSparseFluxTrainer,
        "flowbp_bridge": FlowBPBridgeFluxTrainer,
        "drtune": LeapAlignDRTuneFluxTrainer,
        "draft_lv": LeapAlignDRaFTLVFluxTrainer,
    }
    trainer_map[normalize_trainer_name(args.trainer)](args).train()


if __name__ == "__main__":
    main()
