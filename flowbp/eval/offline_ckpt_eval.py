from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flowbp.eval.config import EvalConfig, configure_hpsv3_env
from flowbp.eval.metrics import as_float_list, collect_metric_values
from flowbp.eval.rewards.multi_scorer import build_reward_scorer


DEFAULT_PROMPTS_FILE = REPO_ROOT / "flowbp" / "dataset" / "drawbench" / "test.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_evaluation"
DEFAULT_FLUX_BASE = REPO_ROOT / "data" / "flux"
DEFAULT_FLUX2_BASE = REPO_ROOT / "data" / "flux2"
DEFAULT_REWARD_CKPT_PATH = REPO_ROOT / "reward_ckpts"
DEFAULT_REWARD_NAMES = ["hpsv2", "imagereward", "clipscore", "pickscore"]


def _setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size, device


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _all_gather_object(obj: Any, world_size: int) -> list[Any]:
    if not dist.is_initialized():
        return [obj]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def _sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("._-") or "model"


def _default_model_name(model_path: str) -> str:
    path = Path(model_path)
    if path.name:
        return _sanitize_name(path.name)
    return _sanitize_name(path.parent.name)


def _load_prompts(path: str, max_prompts: int | None = None) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _load_pipeline(
    model_type: str,
    base_model_path: str,
    ckpt_path: str,
    torch_dtype: torch.dtype,
    device: torch.device,
):
    use_base_transformer = str(ckpt_path).strip().lower() in {
        "",
        "__base__",
        "base",
        "base_model",
    }
    if model_type == "flux":
        from diffusers import FluxPipeline, FluxTransformer2DModel

        pipe = FluxPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        if not use_base_transformer:
            pipe.transformer = FluxTransformer2DModel.from_pretrained(
                ckpt_path,
                torch_dtype=torch_dtype,
            )
    elif model_type == "flux2":
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

        pipe = Flux2KleinPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        if not use_base_transformer:
            pipe.transformer = Flux2Transformer2DModel.from_pretrained(
                ckpt_path,
                torch_dtype=torch_dtype,
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _score_batch(
    scorer: Any,
    images: list[Image.Image],
    prompts: list[str],
    prompt_indices: list[int],
    img_indices: list[int],
    image_paths: list[str],
) -> list[dict[str, Any]]:
    score_details, _ = scorer(images, prompts, metadata={})
    score_lists = {name: as_float_list(scores) for name, scores in score_details.items()}
    rows: list[dict[str, Any]] = []
    for i, (prompt, prompt_idx, img_idx, image_path) in enumerate(
        zip(prompts, prompt_indices, img_indices, image_paths)
    ):
        row: dict[str, Any] = {
            "prompt_idx": int(prompt_idx),
            "img_idx": int(img_idx),
            "prompt": prompt,
            "image": image_path,
        }
        for reward_name, values in score_lists.items():
            if i < len(values):
                row[reward_name] = float(values[i])
        rows.append(row)
    return rows


@torch.no_grad()
def generate_one_checkpoint(
    args: argparse.Namespace,
    model_path: str,
    model_name: str,
    prompts: list[str],
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    safe_model_name = _sanitize_name(model_name)
    model_output_dir = Path(args.output_dir) / safe_model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"=== Offline eval: {safe_model_name} ===")
        print(f"checkpoint: {model_path}")
        print(f"image_dir: {model_output_dir}")

    pipe = _load_pipeline(
        model_type=args.model_type,
        base_model_path=args.pretrained_model_name_or_path,
        ckpt_path=model_path,
        torch_dtype=_torch_dtype(args.dtype),
        device=device,
    )

    local_meta: list[dict[str, Any]] = []
    local_images: list[Image.Image] = []
    local_prompts: list[str] = []
    local_prompt_indices: list[int] = []
    local_img_indices: list[int] = []
    local_image_paths: list[str] = []

    prompt_indices_iter = range(rank, len(prompts), world_size)
    try:
        for prompt_idx in tqdm(
            prompt_indices_iter,
            total=(len(prompts) + world_size - 1 - rank) // world_size,
            disable=rank != 0,
            desc=f"generate:{safe_model_name}",
        ):
            prompt = prompts[prompt_idx]
            for img_idx in range(args.num_imgs_per_prompt):
                seed = args.seed + prompt_idx + img_idx * len(prompts)
                image_path = model_output_dir / f"{prompt_idx:05d}_{img_idx:02d}.png"

                if args.skip_existing and image_path.exists():
                    image = Image.open(image_path).convert("RGB")
                else:
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.autocast(
                        "cuda",
                        _torch_dtype(args.dtype),
                        enabled=device.type == "cuda" and args.dtype != "fp32",
                    ):
                        image = pipe(
                            prompt=prompt,
                            guidance_scale=args.guidance_scale,
                            height=args.height,
                            width=args.width,
                            num_inference_steps=args.num_steps,
                            max_sequence_length=args.max_sequence_length,
                            generator=generator,
                        ).images[0]
                    image.save(image_path)

                image_path_str = str(image_path)
                local_meta.append(
                    {
                        "prompt_idx": int(prompt_idx),
                        "img_idx": int(img_idx),
                        "seed": int(seed),
                        "prompt": prompt,
                        "image": image_path_str,
                    }
                )
                if args.score:
                    local_images.append(image)
                    local_prompts.append(prompt)
                    local_prompt_indices.append(int(prompt_idx))
                    local_img_indices.append(int(img_idx))
                    local_image_paths.append(image_path_str)
    finally:
        del pipe
        torch.cuda.empty_cache()
        gc.collect()

    local_scores: list[dict[str, Any]] = []
    active_rewards: list[str] = []
    if args.score:
        eval_cfg = EvalConfig(
            prompts_file=args.prompts_file,
            num_imgs_per_prompt=args.num_imgs_per_prompt,
            seed=args.seed,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            reward_fn={name: 1.0 for name in args.reward_names},
            reward_ckpt_path=args.reward_ckpt_path,
            output_dir=args.output_dir,
            height=args.height,
            width=args.width,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            hpsv3_config_path=args.hpsv3_config_path,
            hpsv3_checkpoint_path=args.hpsv3_checkpoint_path,
        )
        hpsv3_paths = configure_hpsv3_env(eval_cfg)
        if rank == 0 and hpsv3_paths:
            print(f"HPSv3 paths: {hpsv3_paths}")

        scorer, active_rewards = build_reward_scorer(
            reward_fn=eval_cfg.reward_fn,
            device=device,
            reward_ckpt_path=eval_cfg.reward_ckpt_path,
            allow_unavailable=not args.fail_on_unavailable_reward,
        )
        try:
            for start in tqdm(
                range(0, len(local_images), args.reward_batch_size),
                disable=rank != 0,
                desc=f"score:{safe_model_name}",
            ):
                end = start + args.reward_batch_size
                local_scores.extend(
                    _score_batch(
                        scorer=scorer,
                        images=local_images[start:end],
                        prompts=local_prompts[start:end],
                        prompt_indices=local_prompt_indices[start:end],
                        img_indices=local_img_indices[start:end],
                        image_paths=local_image_paths[start:end],
                    )
                )
        finally:
            scorer.to(torch.device("cpu"))
            del scorer
            torch.cuda.empty_cache()
            gc.collect()

    gathered_meta = _all_gather_object(local_meta, world_size)
    gathered_scores = _all_gather_object(local_scores, world_size)
    _barrier()

    if rank != 0:
        return None

    all_meta = [item for rank_meta in gathered_meta for item in (rank_meta or [])]
    all_meta.sort(key=lambda item: (item["prompt_idx"], item["img_idx"]))
    _write_json(model_output_dir / "meta.json", all_meta)

    result: dict[str, Any] = {
        "model_name": safe_model_name,
        "model_path": model_path,
        "model_type": args.model_type,
        "base_model": args.pretrained_model_name_or_path,
        "output_dir": str(model_output_dir),
        "num_prompts": len(prompts),
        "num_imgs_per_prompt": args.num_imgs_per_prompt,
        "num_images": len(all_meta),
        "seed_formula": "seed + prompt_idx + img_idx * num_prompts",
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
    }

    if args.score:
        all_scores = [
            item for rank_scores in gathered_scores for item in (rank_scores or [])
        ]
        all_scores.sort(key=lambda item: (item["prompt_idx"], item["img_idx"]))
        metric_names = list(active_rewards) + ["mean"]
        metric_values = collect_metric_values(all_scores, metric_names)
        avg_scores = {
            name: float(np.mean(values))
            for name, values in metric_values.items()
        }
        std_scores = {
            name: float(np.std(values))
            for name, values in metric_values.items()
        }
        result.update(
            {
                "reward_names": list(active_rewards),
                "avg_scores": avg_scores,
                "std_scores": std_scores,
                "scores": all_scores,
            }
        )
        _write_json(model_output_dir / "scores.json", result)
        print(f"=== Summary: {safe_model_name} ===")
        for reward_name, value in avg_scores.items():
            print(f"{reward_name}: {value:.4f}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline FLUX/FLUX.2 checkpoint generation with eval-aligned seeds. "
            "Images are grouped by checkpoint under final_evaluation by default."
        ),
    )
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional names for checkpoint output folders. Defaults to checkpoint folder names.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["flux", "flux2"],
        default="flux",
        help="Pipeline/checkpoint family to load.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Base model directory. Defaults to data/flux for --model_type flux, "
            "and the local FLUX.2 klein-base path for --model_type flux2."
        ),
    )
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS_FILE))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_imgs_per_prompt", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Defaults to 3.5 for FLUX.1 and 4.0 for FLUX.2.",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--score",
        action="store_true",
        help="Also run local reward scoring and write scores.json.",
    )
    parser.add_argument(
        "--reward_ckpt_path",
        type=str,
        default=str(DEFAULT_REWARD_CKPT_PATH),
    )
    parser.add_argument("--reward_names", type=str, nargs="+", default=DEFAULT_REWARD_NAMES)
    parser.add_argument("--reward_batch_size", type=int, default=16)
    parser.add_argument("--fail_on_unavailable_reward", action="store_true", default=False)
    parser.add_argument("--hpsv3_config_path", type=str, default="")
    parser.add_argument("--hpsv3_checkpoint_path", type=str, default="")
    args = parser.parse_args()

    if args.pretrained_model_name_or_path is None:
        if args.model_type == "flux2":
            args.pretrained_model_name_or_path = str(DEFAULT_FLUX2_BASE)
        else:
            args.pretrained_model_name_or_path = str(DEFAULT_FLUX_BASE)
    if args.guidance_scale is None:
        args.guidance_scale = 4.0 if args.model_type == "flux2" else 3.5
    if args.height is None:
        args.height = 512 if args.model_type == "flux2" else 720
    if args.width is None:
        args.width = 512 if args.model_type == "flux2" else 720
    if args.model_names is None:
        args.model_names = [_default_model_name(path) for path in args.model_paths]
    if len(args.model_names) != len(args.model_paths):
        raise ValueError(
            "--model_names and --model_paths must have the same length: "
            f"{len(args.model_names)} vs {len(args.model_paths)}"
        )
    return args


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = _setup_distributed()
    prompts = _load_prompts(args.prompts_file, max_prompts=args.max_prompts)
    if rank == 0:
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        print(f"world_size={world_size}, device={device}")
        print(f"output_dir={args.output_dir}")

    summaries: list[dict[str, Any]] = []
    try:
        for model_name, model_path in zip(args.model_names, args.model_paths):
            result = generate_one_checkpoint(
                args=args,
                model_path=model_path,
                model_name=model_name,
                prompts=prompts,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if rank == 0 and result is not None:
                summary = {
                    "model_name": result["model_name"],
                    "model_path": result["model_path"],
                    "output_dir": result["output_dir"],
                    "num_images": result["num_images"],
                    "seed": result["seed"],
                }
                if args.score:
                    summary["avg_scores"] = result.get("avg_scores", {})
                    summary["std_scores"] = result.get("std_scores", {})
                summaries.append(summary)
            _barrier()

        if rank == 0:
            _write_json(Path(args.output_dir) / "summary.json", summaries)
            print(f"Wrote summary to {Path(args.output_dir) / 'summary.json'}")
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
