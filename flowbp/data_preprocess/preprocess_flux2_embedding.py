
"""Precompute Qwen3 prompt embeddings for FLUX.2 LeapAlign training.

Mirrors :mod:`flowbp.data_preprocess.preprocess_flux_embedding` (the
FLUX.1 T5+CLIP preprocessor) but targets the FLUX.2 Qwen3 text encoder.

Output layout (matches ``PrecomputedFlux2Dataset``):

    <output_dir>/
      prompt_embed/<idx>.pt   # (max_seq_len, joint_attention_dim) float
      text_ids/<idx>.pt       # (max_seq_len, 4) long
      captions2embed.json     # [{"prompt_embed_path", "text_ids", "caption"}, ...]

Notes:
* FLUX.2 has no pooled embedding, so we do not produce a
  ``pooled_prompt_embeds`` directory (the FLUX.1 schema is preserved
  otherwise).
* ``text_ids`` only depend on ``max_sequence_length`` so they are identical
  across prompts; we still write per-prompt copies for index-file
  compatibility with the trainer dataset.
* Captions are read from a ``.txt`` (one prompt per line) or a ``.json``
  list / ``{"prompts": [...]}`` / records-with-``caption`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from flowbp.dataset.prompt_flux2_dataset import _load_captions
from flowbp.trainers.flux2.leapalign import (
    encode_prompts_qwen3,
    prepare_text_ids,
)


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


class CaptionDataset(Dataset):
    def __init__(
        self,
        prompt_dir: str,
        caption_key: str = "caption",
        filter_chinese: bool = False,
        max_samples: int | None = None,
    ):
        captions = _load_captions(prompt_dir, caption_key=caption_key)
        if filter_chinese:
            captions = [c for c in captions if not _contains_chinese(c)]
        if max_samples is not None:
            captions = captions[: int(max_samples)]
        if not captions:
            raise ValueError(f"No usable captions in {prompt_dir}")
        self.captions = captions

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {"caption": self.captions[idx], "filename": str(idx)}


def _collate(batch):
    return {
        "caption": [b["caption"] for b in batch],
        "filename": [b["filename"] for b in batch],
    }


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=local_rank,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "text_ids"), exist_ok=True)

    dataset = CaptionDataset(
        prompt_dir=args.prompt_dir,
        caption_key=args.caption_key,
        filter_chinese=args.filter_chinese,
        max_samples=args.max_samples,
    )
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, rank=local_rank, num_replicas=world_size, shuffle=False,
        )
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=_collate,
        shuffle=False,
    )

    print(f"--> loading Qwen3 text encoder from {args.model_path}")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.model_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    ).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.model_path,
        subfolder="tokenizer",
    )

    hidden_layers = tuple(args.text_encoder_out_layers)
    json_data: list[dict] = []
    iterator = enumerate(dataloader)
    if local_rank == 0:
        iterator = tqdm(iterator, total=len(dataloader))

    for _, data in iterator:
        try:
            prompt_embeds = encode_prompts_qwen3(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompts=data["caption"],
                device=device,
                dtype=torch.bfloat16,
                max_sequence_length=args.max_sequence_length,
                hidden_states_layers=hidden_layers,
            )
            text_ids = prepare_text_ids(
                batch_size=prompt_embeds.shape[0],
                seq_len=prompt_embeds.shape[1],
                device=device,
                dtype=torch.int64,
            )

            prompt_embeds_cpu = prompt_embeds.to(args.save_dtype).cpu()
            text_ids_cpu = text_ids.to(torch.int64).cpu()

            for idx, filename in enumerate(data["filename"]):
                prompt_embed_path = os.path.join(
                    args.output_dir, "prompt_embed", filename + ".pt",
                )
                text_ids_path = os.path.join(
                    args.output_dir, "text_ids", filename + ".pt",
                )
                torch.save(prompt_embeds_cpu[idx].clone(), prompt_embed_path)
                torch.save(text_ids_cpu[idx].clone(), text_ids_path)
                json_data.append({
                    "prompt_embed_path": filename + ".pt",
                    "text_ids": filename + ".pt",
                    "caption": data["caption"][idx],
                })
        except Exception as e:
            print(f"Rank {local_rank} Error: {repr(e)}")
            if dist.is_initialized():
                dist.barrier()
            raise

    if dist.is_initialized():
        dist.barrier()
        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, json_data)
        if local_rank == 0:
            all_json = [item for sub in gathered for item in sub]
            with open(os.path.join(args.output_dir, "captions2embed.json"), "w") as f:
                json.dump(all_json, f, indent=2)
    else:
        with open(os.path.join(args.output_dir, "captions2embed.json"), "w") as f:
            json.dump(json_data, f, indent=2)


def _str_to_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
        help="Root directory of the FLUX.2 model with text_encoder/, tokenizer/.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="./assets/prompts.txt",
        help="Captions file (.txt one-per-line, or .json / .jsonl).",
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="caption",
        help="JSON field holding the caption (ignored for .txt inputs).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/rl_embeddings_flux2",
        help="Where to write prompt_embed/, text_ids/, captions2embed.json.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Qwen3 tokenizer max length (FLUX.2 default is 512).",
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[9, 18, 27],
        help="Qwen3 hidden-state layer indices to stack (3 -> joint_attention_dim=7680).",
    )
    parser.add_argument(
        "--save_dtype",
        type=_str_to_dtype,
        default="bf16",
        help="Disk dtype for prompt embeddings (bf16 / fp16 / fp32).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Per-rank batch size for the encoder forward.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optionally cap the number of prompts (debug).",
    )
    parser.add_argument(
        "--filter_chinese",
        action="store_true",
        default=False,
        help="Drop prompts containing CJK characters before encoding.",
    )
    args = parser.parse_args()
    main(args)
