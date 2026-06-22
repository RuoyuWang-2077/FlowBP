import json
import os
import random

import torch
from torch.utils.data import Dataset


class LatentSD35Dataset(Dataset):
    """Latent dataset for SD3.5 RL training.

    Each sample returns both positive prompt embeddings (text + pooled) and the
    matching negative prompt embeddings (text + pooled, normally encoded from
    the empty string ``""``). Both are required because SD3.5 sampling uses
    real classifier-free guidance, which concatenates ``[neg, pos]`` along the
    batch axis at every transformer call.

    ``cfg_rate`` mirrors the unconditional-dropout knob from the diffusion
    training literature (replace the conditional embedding by the unconditional
    one with probability ``cfg_rate``). For RL fine-tuning you almost always
    want ``cfg_rate=0.0``; otherwise the dropped samples become trivially
    unconditional under CFG (the duplicated ``[neg, neg]`` forward collapses
    the guidance term to zero).
    """

    def __init__(self, json_path, num_latent_t, cfg_rate):
        del num_latent_t  # kept for parity with the Flux dataset signature
        self.json_path = json_path
        self.cfg_rate = float(cfg_rate)
        self.datase_dir_path = os.path.dirname(json_path)
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(
            self.datase_dir_path,
            "pooled_prompt_embeds",
        )
        self.negative_prompt_embed_dir = os.path.join(
            self.datase_dir_path,
            "negative_prompt_embed",
        )
        self.negative_pooled_prompt_embeds_dir = os.path.join(
            self.datase_dir_path,
            "negative_pooled_prompt_embeds",
        )
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data_anno = json.load(f)

    def _load_tensor(self, directory, filename):
        return torch.load(
            os.path.join(directory, filename),
            map_location="cpu",
            weights_only=True,
        )

    def __getitem__(self, idx):
        item = self.data_anno[idx]
        prompt_embed = self._load_tensor(
            self.prompt_embed_dir,
            item["prompt_embed_path"],
        )
        pooled_prompt_embeds = self._load_tensor(
            self.pooled_prompt_embeds_dir,
            item["pooled_prompt_embeds_path"],
        )

        neg_prompt_file = item.get("negative_prompt_embed_path")
        neg_pooled_file = item.get("negative_pooled_prompt_embeds_path")
        if not (neg_prompt_file and neg_pooled_file):
            # Refuse the silent zeros-fallback: SD3.5's "unconditional" is the
            # empty-string embedding, which is decidedly non-zero, so faking it
            # with zeros would corrupt CFG. Re-run preprocess_sd35_embedding.py
            # to populate negative_prompt_embed/ and negative_pooled_prompt_embeds/.
            raise RuntimeError(
                f"Dataset entry {idx} is missing pre-computed negative embeddings. "
                "Re-run flowbp/data_preprocess/preprocess_sd35_embedding.py "
                "to materialise them; SD3.5 RL training requires real CFG embeddings."
            )

        negative_prompt_embed = self._load_tensor(
            self.negative_prompt_embed_dir,
            neg_prompt_file,
        )
        negative_pooled_prompt_embeds = self._load_tensor(
            self.negative_pooled_prompt_embeds_dir,
            neg_pooled_file,
        )

        if self.cfg_rate > 0 and random.random() < self.cfg_rate:
            prompt_embed = negative_prompt_embed
            pooled_prompt_embeds = negative_pooled_prompt_embeds

        return (
            prompt_embed,
            pooled_prompt_embeds,
            negative_prompt_embed,
            negative_pooled_prompt_embeds,
            item["caption"],
        )

    def __len__(self):
        return len(self.data_anno)


def latent_sd35_collate_function(batch):
    (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        caption,
    ) = zip(*batch)
    return (
        torch.stack(prompt_embeds, dim=0),
        torch.stack(pooled_prompt_embeds, dim=0),
        torch.stack(negative_prompt_embeds, dim=0),
        torch.stack(negative_pooled_prompt_embeds, dim=0),
        caption,
    )
