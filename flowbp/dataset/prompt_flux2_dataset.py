
"""Datasets for FLUX.2 LeapAlign training.

Two parallel paths are supported:

1. **Online encoding** — ``PromptFlux2Dataset`` emits raw captions; the
   trainer runs ``Qwen3ForCausalLM`` on the fly each step. No preprocessing
   step required.
2. **Precomputed embeds** — ``PrecomputedFlux2Dataset`` loads ``.pt`` files
   produced by ``flowbp/data_preprocess/preprocess_flux2_embedding.py``.
   This matches the FLUX.1 ``LatentDataset`` schema (an index JSON pointing
   to per-prompt tensors) and lets us skip the ~8 GB Qwen3 text encoder
   during training.

Input formats supported by the online path:

* ``.json`` — a list of caption strings, a list of
  ``{"caption": str, ...}`` records, or a top-level ``{"prompts": [...]}``
  object.
* ``.jsonl`` — one JSON record per line.
* ``.txt`` — one caption per non-empty line. Trailing whitespace is stripped.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import torch
from torch.utils.data import Dataset


def _load_captions(path: str, caption_key: str) -> list[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            captions = [line.strip() for line in f if line.strip()]
        if not captions:
            raise ValueError(f"No captions parsed from {path}")
        return captions

    if ext not in (".json", ".jsonl"):
        raise ValueError(
            f"Unsupported caption file extension {ext!r} for {path}; "
            "expected .txt, .json, or .jsonl"
        )

    if ext == ".jsonl":
        entries: list[Any] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw: Any = json.load(f)
        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, dict) and "prompts" in raw:
            entries = raw["prompts"]
        else:
            raise ValueError(
                f"Unsupported prompts file structure at {path}: "
                "expected a list or an object with key 'prompts'."
            )

    captions: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            captions.append(entry)
        elif isinstance(entry, dict):
            if caption_key not in entry:
                raise KeyError(
                    f"Entry {entry!r} missing caption_key={caption_key!r}"
                )
            captions.append(str(entry[caption_key]))
        else:
            raise TypeError(
                f"Unsupported entry type {type(entry)} in {path}"
            )

    if not captions:
        raise ValueError(f"No captions parsed from {path}")
    return captions


class PromptFlux2Dataset(Dataset):
    """Returns a single caption per index for FLUX.2 RLHF training.

    The ``json_path`` argument is misnamed for backwards compat with the
    FLUX.1 ``LatentDataset`` interface — the file can equally be a
    plain-text caption-per-line listing.
    """

    def __init__(
        self,
        json_path: str,
        cfg_rate: float = 0.0,
        caption_key: str = "caption",
        uncond_caption: str = "",
    ):
        self.json_path = json_path
        self.cfg_rate = float(cfg_rate)
        self.caption_key = caption_key
        self.uncond_caption = uncond_caption
        self.captions: list[str] = _load_captions(json_path, caption_key)

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> str:
        caption = self.captions[idx]
        if self.cfg_rate > 0.0 and random.random() < self.cfg_rate:
            caption = self.uncond_caption
        return caption


def prompt_flux2_collate_function(batch: list[str]) -> dict:
    """Collate raw-caption batches into a dict ``{"caption": list[str]}``."""
    return {"caption": list(batch)}


class PrecomputedFlux2Dataset(Dataset):
    """Loads FLUX.2 Qwen3 embeddings produced by ``preprocess_flux2_embedding``.

    The directory layout mirrors the FLUX.1 ``LatentDataset`` convention:

    ``<root>/captions2embed.json``  -- list of ``{"prompt_embed_path", "text_ids", "caption"}``
    ``<root>/prompt_embed/<name>.pt``
    ``<root>/text_ids/<name>.pt``

    For unconditional dropout (CFG training) we emit a zero prompt-embed of
    the configured shape — the trainer is responsible for wiring the
    classifier-free path; today's distilled klein-4B variant typically uses
    ``cfg_rate=0``.
    """

    def __init__(
        self,
        json_path: str,
        cfg_rate: float = 0.0,
        uncond_shape: tuple[int, int] | None = None,
    ):
        self.json_path = json_path
        self.dataset_dir = os.path.dirname(json_path)
        self.prompt_embed_dir = os.path.join(self.dataset_dir, "prompt_embed")
        self.text_ids_dir = os.path.join(self.dataset_dir, "text_ids")
        self.cfg_rate = float(cfg_rate)

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data_anno = json.load(f)

        if not self.data_anno:
            raise ValueError(f"Empty index at {json_path}")

        if uncond_shape is None:
            sample = torch.load(
                os.path.join(self.prompt_embed_dir, self.data_anno[0]["prompt_embed_path"]),
                map_location="cpu",
                weights_only=True,
            )
            uncond_shape = tuple(sample.shape)
        self.uncond_prompt_embed = torch.zeros(uncond_shape, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data_anno)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.data_anno[idx]
        if self.cfg_rate > 0.0 and random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
        else:
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, entry["prompt_embed_path"]),
                map_location="cpu",
                weights_only=True,
            )
        text_ids = torch.load(
            os.path.join(self.text_ids_dir, entry["text_ids"]),
            map_location="cpu",
            weights_only=True,
        )
        return {
            "prompt_embed": prompt_embed,
            "text_ids": text_ids,
            "caption": entry.get("caption", ""),
        }


def precomputed_flux2_collate_function(batch: list[dict]) -> dict:
    prompt_embeds = torch.stack([item["prompt_embed"] for item in batch], dim=0)
    text_ids = torch.stack([item["text_ids"] for item in batch], dim=0)
    captions = [item["caption"] for item in batch]
    return {
        "prompt_embed": prompt_embeds,
        "text_ids": text_ids,
        "caption": captions,
    }


def flux2_dataloader_wrapper(
    dataloader: torch.utils.data.DataLoader,
    device,
    epoch_idx_auto_increment: bool = True,
):
    """Infinite generator yielding batch dicts.

    The dict always contains ``"caption"`` (a ``list[str]``). When the
    dataset emits precomputed tensors (``PrecomputedFlux2Dataset``), the dict
    also carries ``"prompt_embed"`` and ``"text_ids"`` moved to ``device``.
    Otherwise the trainer is expected to encode the captions on the fly.
    """
    while True:
        for batch in dataloader:
            if "prompt_embed" in batch:
                batch["prompt_embed"] = batch["prompt_embed"].to(device, non_blocking=True)
            if "text_ids" in batch:
                batch["text_ids"] = batch["text_ids"].to(device, non_blocking=True)
            yield batch
        sampler = getattr(dataloader, "sampler", None)
        if epoch_idx_auto_increment and sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(getattr(sampler, "epoch", 0) + 1)
