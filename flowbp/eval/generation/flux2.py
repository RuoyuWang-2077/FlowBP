
"""Distributed image generation for FLUX.2 online eval.

Mirror of :mod:`flowbp.eval.generation.flux` but built around
``Flux2KleinPipeline``. The trainer-side FSDP-wrapped transformer and the
VAE (already on-device) are spliced into a freshly-loaded pipeline; the
pipeline's own text encoder + tokenizer + scheduler are used for sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
from torch.utils.data import Dataset, DistributedSampler
from tqdm.auto import tqdm

from flowbp.eval.config import EvalConfig
from flowbp.eval.rewards.multi_scorer import normalize_torch_device
from flowbp.utils.logging_ import main_print


class EvalPromptDataset(Dataset):
    def __init__(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> tuple[int, str]:
        return idx, self.prompts[idx]


@dataclass
class GeneratedEvalItem:
    prompt_idx: int
    img_idx: int
    image: Image.Image
    prompt: str


@torch.no_grad()
def sample_flux2_images_distributed(
    eval_cfg: EvalConfig,
    transformer,
    vae,
    rank: int,
    world_size: int,
    device: int | str | torch.device,
) -> list[GeneratedEvalItem]:
    """Generate eval images for rank-local prompts using ``Flux2KleinPipeline``.

    ``transformer`` is the FSDP-wrapped Flux2 transformer being trained; we
    splice it into a freshly-loaded pipeline along with the trainer's VAE
    (already on-device). The pipeline's bundled Qwen3 text encoder and
    tokenizer get loaded onto the rank-local device for the eval window.
    """
    torch_device = normalize_torch_device(device)
    dataset = EvalPromptDataset(eval_cfg.prompts_file)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        eval_cfg.pretrained_model_name_or_path,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.transformer = transformer
    pipe.text_encoder.to(torch_device)
    pipe.set_progress_bar_config(disable=True)

    main_print(
        f"--> Online Evaluation (FLUX.2): {len(dataset)} prompts x "
        f"{eval_cfg.num_imgs_per_prompt} imgs @ {eval_cfg.height}x{eval_cfg.width} "
        f"({eval_cfg.num_steps} steps, guidance={eval_cfg.guidance_scale})"
    )

    local_results: list[GeneratedEvalItem] = []
    try:
        for prompt_idx in tqdm(sampler, postfix=f"Eval Rank {rank}", disable=rank != 0):
            prompt_idx_int = int(prompt_idx)
            _, prompt = dataset[prompt_idx_int]
            for img_idx in range(eval_cfg.num_imgs_per_prompt):
                generator = torch.Generator(device=torch_device).manual_seed(
                    eval_cfg.seed + prompt_idx_int + img_idx * len(dataset)
                )
                with torch.autocast("cuda", torch.bfloat16):
                    image = pipe(
                        prompt=prompt,
                        guidance_scale=eval_cfg.guidance_scale,
                        height=eval_cfg.height,
                        width=eval_cfg.width,
                        num_inference_steps=eval_cfg.num_steps,
                        max_sequence_length=512,
                        generator=generator,
                    ).images[0]
                local_results.append(
                    GeneratedEvalItem(
                        prompt_idx=prompt_idx_int,
                        img_idx=img_idx,
                        image=image,
                        prompt=prompt,
                    )
                )
    finally:
        del pipe

    return local_results
