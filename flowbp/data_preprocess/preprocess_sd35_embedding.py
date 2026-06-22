import argparse
import json
import os
import re

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

logger = get_logger(__name__)


def contains_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))


class PromptDataset(Dataset):
    def __init__(self, txt_path, max_items=None):
        self.txt_path = txt_path
        with open(self.txt_path, "r", encoding="utf-8") as f:
            prompts = [line for line in f.read().splitlines() if not contains_chinese(line)]
        self.prompts = prompts[:max_items] if max_items is not None else prompts

    def __getitem__(self, idx):
        return {
            "caption": self.prompts[idx],
            "filename": str(idx),
        }

    def __len__(self):
        return len(self.prompts)


def _setup_distributed():
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", rank))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    return rank, local_rank, world_size


def main(args):
    rank, local_rank, world_size = _setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        for subdir in (
            "prompt_embed",
            "pooled_prompt_embeds",
            "negative_prompt_embed",
            "negative_pooled_prompt_embeds",
        ):
            os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    dataset = PromptDataset(args.prompt_dir, args.max_items)
    # Use explicit rank-stride sharding instead of DistributedSampler. The latter
    # pads samples when len(dataset) is not divisible by world_size, which would
    # duplicate prompts in the generated videos2caption.json.
    rank_indices = list(range(rank, len(dataset), world_size))
    rank_dataset = Subset(dataset, rank_indices)
    dataloader = DataLoader(
        rank_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    json_data = []
    for data in tqdm(dataloader, disable=rank != 0):
        captions = list(data["caption"])
        filenames = list(data["filename"])
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=captions,
                prompt_2=captions,
                prompt_3=captions,
                negative_prompt=[""] * len(captions),
                negative_prompt_2=[""] * len(captions),
                negative_prompt_3=[""] * len(captions),
                do_classifier_free_guidance=True,
                max_sequence_length=args.max_sequence_length,
                device=device,
            )

        for idx, filename in enumerate(filenames):
            tensor_name = f"{filename}.pt"
            torch.save(
                prompt_embeds[idx].detach().cpu(),
                os.path.join(args.output_dir, "prompt_embed", tensor_name),
            )
            torch.save(
                pooled_prompt_embeds[idx].detach().cpu(),
                os.path.join(args.output_dir, "pooled_prompt_embeds", tensor_name),
            )
            torch.save(
                negative_prompt_embeds[idx].detach().cpu(),
                os.path.join(args.output_dir, "negative_prompt_embed", tensor_name),
            )
            torch.save(
                negative_pooled_prompt_embeds[idx].detach().cpu(),
                os.path.join(args.output_dir, "negative_pooled_prompt_embeds", tensor_name),
            )
            json_data.append(
                {
                    "prompt_embed_path": tensor_name,
                    "pooled_prompt_embeds_path": tensor_name,
                    "negative_prompt_embed_path": tensor_name,
                    "negative_pooled_prompt_embeds_path": tensor_name,
                    "caption": captions[idx],
                }
            )

    if dist.is_initialized():
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, json_data)
    else:
        gathered_data = [json_data]

    if rank == 0:
        all_json_data = [item for rank_data in gathered_data for item in rank_data]
        all_json_data.sort(key=lambda item: int(os.path.splitext(item["prompt_embed_path"])[0]))
        with open(os.path.join(args.output_dir, "videos2caption.json"), "w", encoding="utf-8") as f:
            json.dump(all_json_data, f, indent=4)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/sd3.5_medium",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, default="./empty.txt")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()
    main(args)
