
"""FlowBP base and LeapAlign-baseline trainer for FLUX.2.

Mirrors :mod:`flowbp.trainers.flux1.leapalign` but targets the
FLUX.2 architecture:

* Qwen3 text encoder (instead of T5 + CLIP), encoded on the fly per training
  step from raw captions.
* 32-channel VAE latents with a 2x2 patchify stage and BatchNorm-based
  normalization (no scale/shift constants).
* 4D positional ids ``(T, H, W, L)`` for both text and image tokens.
* No pooled projections and no guidance embedding (the distilled klein-4B
  variant disables both).
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from flowbp.config.flowbp import (
    SUPPORTED_INTERNAL_TRAINERS,
    normalize_trainer_name,
)
from accelerate.utils import set_seed
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from flowbp.dataset.prompt_flux2_dataset import (
    PrecomputedFlux2Dataset,
    PromptFlux2Dataset,
    flux2_dataloader_wrapper,
    precomputed_flux2_collate_function,
    prompt_flux2_collate_function,
)
from flowbp.schedulers.fm_euler_discrete_with_pred_x0 import (
    FlowMatchEulerDiscreteWithPredX0Scheduler,
)
from flowbp.trainers.common.jk_sampling import sample_jk_indices
from flowbp.eval.runners.online_flux2 import run_online_flux2_eval
from flowbp.utils.checkpoint import save_checkpoint
from flowbp.utils.debug_utils import setup_debug
from flowbp.utils.ema_utils import save_sharded_ema_checkpoint
from flowbp.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
from flowbp.utils.hpsv2_transforms import HPSV2TransformsWithGrad
from flowbp.utils.logging_ import main_print
from flowbp.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
)

check_min_version("0.37.0")


# -----------------------------------------------------------------------------
# Flux2 latent packing / id helpers
# -----------------------------------------------------------------------------
# Flux2 packs a (B, C=32, H, W) VAE latent into transformer input by:
#   1. patchify: (B, 32, H, W) -> (B, 128, H/2, W/2)        [2x2 channel-pack]
#   2. BN-normalize using vae.bn.running_mean / running_var
#   3. pack:     (B, 128, h, w) -> (B, h*w, 128)            [token sequence]
# The transformer expects ``in_channels=128`` and image_ids with shape
# (B, h*w, 4) carrying coordinates (T=0, H, W, L=0).

FLUX2_LATENT_CHANNELS = 32
FLUX2_PATCH_CHANNELS = FLUX2_LATENT_CHANNELS * 4  # = 128
FLUX2_VAE_DOWNSAMPLE = 8
FLUX2_PATCH_SIZE = 2
FLUX2_TOTAL_DOWNSAMPLE = FLUX2_VAE_DOWNSAMPLE * FLUX2_PATCH_SIZE  # = 16


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, 32, H, W) -> (B, 128, H/2, W/2)."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(b, c * 4, h // 2, w // 2)
    return latents


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, 128, H, W) -> (B, 32, 2H, 2W)."""
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(b, c // 4, h * 2, w * 2)
    return latents


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) -> (B, h*w, C). Used after patchify+BN."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1).contiguous()


def unpack_latents(packed: torch.Tensor, h_tokens: int, w_tokens: int) -> torch.Tensor:
    """(B, h*w, C) -> (B, C, h, w)."""
    b, n, c = packed.shape
    assert n == h_tokens * w_tokens, (n, h_tokens, w_tokens)
    return packed.permute(0, 2, 1).reshape(b, c, h_tokens, w_tokens)


def prepare_latent_ids(
    batch_size: int,
    h_tokens: int,
    w_tokens: int,
    device,
    dtype,
) -> torch.Tensor:
    """Return (B, h*w, 4) ids with (T=0, H, W, L=0)."""
    t = torch.arange(1)
    h = torch.arange(h_tokens)
    w = torch.arange(w_tokens)
    l = torch.arange(1)
    coords = torch.cartesian_prod(t, h, w, l)
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
    return coords.to(device=device, dtype=dtype)


def prepare_text_ids(
    batch_size: int,
    seq_len: int,
    device,
    dtype,
) -> torch.Tensor:
    """Return (B, L, 4) ids with (T=0, H=0, W=0, L=range(L))."""
    t = torch.arange(1)
    h = torch.arange(1)
    w = torch.arange(1)
    l = torch.arange(seq_len)
    coords = torch.cartesian_prod(t, h, w, l)
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
    return coords.to(device=device, dtype=dtype)


def get_latent_bn_normalizers(vae: AutoencoderKLFlux2, dtype, device):
    """Pull (mean, std) tensors from the VAE's running BN statistics."""
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(device=device, dtype=dtype)
    return bn_mean, bn_std


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """The Flux2 dynamic-shift mu schedule. Copied from diffusers."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


# -----------------------------------------------------------------------------
# Qwen3 text encoder helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def encode_prompts_qwen3(
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    prompts: list[str],
    device,
    dtype,
    max_sequence_length: int = 512,
    hidden_states_layers: tuple[int, ...] = (9, 18, 27),
) -> torch.Tensor:
    """Replicates Flux2KleinPipeline._get_qwen3_prompt_embeds.

    Returns ``prompt_embeds`` with shape ``(B, max_sequence_length,
    num_layers * hidden_dim)``.
    """
    input_id_list = []
    attn_mask_list = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        input_id_list.append(inputs["input_ids"])
        attn_mask_list.append(inputs["attention_mask"])

    input_ids = torch.cat(input_id_list, dim=0).to(device)
    attn_mask = torch.cat(attn_mask_list, dim=0).to(device)

    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    stacked = torch.stack(
        [output.hidden_states[k] for k in hidden_states_layers], dim=1
    )  # (B, num_layers, L, D)
    stacked = stacked.to(dtype=dtype, device=device)
    b, num_layers, seq_len, hidden_dim = stacked.shape
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
        b, seq_len, num_layers * hidden_dim
    )
    return prompt_embeds


# -----------------------------------------------------------------------------
# Connector dumping / wandb image logging
# -----------------------------------------------------------------------------

def _maybe_save_connector_dump(args, prefix: str, payload: dict) -> None:
    interval = int(getattr(args, "connector_dump_interval", 0) or 0)
    step = getattr(args, "current_train_step", None)
    if interval <= 0 or step is None or int(step) % interval != 0:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    dump_dir = getattr(args, "connector_dump_dir", None)
    if not dump_dir:
        dump_dir = os.path.join(args.output_dir, "connector_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    saved = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            saved[key] = value.detach().cpu()
        else:
            saved[key] = value
    torch.save(saved, os.path.join(dump_dir, f"{prefix}_step_{int(step):06d}.pt"))


def _decode_packed_latents_for_wandb(args, vae, packed_latents, max_samples: int):
    if packed_latents is None:
        return []
    n = min(int(max_samples), packed_latents.shape[0])
    if n <= 0:
        return []
    h_tokens = args.h // FLUX2_TOTAL_DOWNSAMPLE
    w_tokens = args.w // FLUX2_TOTAL_DOWNSAMPLE
    sample = packed_latents[:n].to(vae.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latents = unpack_latents(sample, h_tokens, w_tokens)
        bn_mean, bn_std = get_latent_bn_normalizers(vae, latents.dtype, latents.device)
        latents = latents * bn_std + bn_mean
        latents = unpatchify_latents(latents)
        images = vae.decode(latents, return_dict=False)[0]
        images = (images * 0.5 + 0.5).clamp(0, 1)
    return [
        image.detach().float().cpu().permute(1, 2, 0).numpy() for image in images
    ]


def _maybe_log_connector_wandb(args, vae, prefix: str, payload: dict | None) -> None:
    interval = int(getattr(args, "connector_wandb_interval", 0) or 0)
    step = getattr(args, "current_train_step", None)
    if interval <= 0 or step is None or int(step) % interval != 0:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    if payload is None or wandb.run is None:
        return

    max_samples = int(getattr(args, "connector_wandb_num_samples", 2) or 2)
    log_payload = {}
    for name in ("xj_pred", "xj", "x0_pred", "x0"):
        images = _decode_packed_latents_for_wandb(
            args, vae, payload.get(name), max_samples,
        )
        if images:
            log_payload[f"connector/{prefix}_{name}"] = [
                wandb.Image(image, caption=f"{prefix} {name} sample {idx}")
                for idx, image in enumerate(images)
            ]
    if log_payload:
        wandb.log(log_payload, step=int(step))


# -----------------------------------------------------------------------------
# Trajectory sampling (LeapAlign baseline on FLUX.2)
# -----------------------------------------------------------------------------

def _flux2_transformer_forward(
    transformer,
    hidden_states,
    timestep,
    encoder_hidden_states,
    txt_ids,
    img_ids,
):
    """One forward pass on the Flux2 transformer with bf16 autocast."""
    with torch.autocast("cuda", torch.bfloat16):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep / 1000,
            guidance=None,
            img_ids=img_ids,
            txt_ids=txt_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    return out


def _cfg_aware_forward(
    args,
    transformer,
    hidden_states,
    timestep,
    encoder_hidden_states,
    txt_ids,
    img_ids,
):
    """Classifier-free-guidance combine for FLUX.2 undistilled checkpoints.

    klein-base-4B and klein-base-9B have ``guidance_embeds: false``, so the
    ``guidance`` argument to the transformer is silently ignored — they need
    explicit two-branch CFG. This helper:

    * runs the conditional forward (with grad if upstream context allows),
    * if ``args.cfg_guidance > 1`` AND a cached negative-prompt embedding is
      available, runs a no-grad detached forward with the empty negative
      prompt and returns ``pred_neg + cfg * (pred_cond - pred_neg)``,
    * otherwise returns the conditional prediction unchanged.

    The negative branch is detached so gradient flows ONLY through the
    conditional pass. Forward-side numerics still match the inference
    pipeline exactly.
    """
    pred_cond = _flux2_transformer_forward(
        transformer,
        hidden_states,
        timestep,
        encoder_hidden_states,
        txt_ids,
        img_ids,
    )

    cfg = float(getattr(args, "cfg_guidance", 1.0))
    neg_embeds = getattr(args, "_neg_prompt_embeds", None)
    neg_text_ids = getattr(args, "_neg_text_ids", None)
    if cfg <= 1.0 or neg_embeds is None or neg_text_ids is None:
        return pred_cond

    batch_size = hidden_states.shape[0]
    if neg_embeds.shape[0] != batch_size:
        neg_embeds = neg_embeds.expand(batch_size, -1, -1)
    if neg_text_ids.shape[0] != batch_size:
        neg_text_ids = neg_text_ids.expand(batch_size, -1, -1)

    # Detach mode (default): pred_neg is computed under no_grad, so the
    # gradient flows only through pred_cond and is amplified by cfg.
    # Non-detach mode: pred_neg also tracks gradient, giving the
    # mathematically standard CFG gradient
    # ``(1-cfg)*d(pred_neg)/dθ + cfg*d(pred_cond)/dθ``. This costs ~1.8x
    # activation memory but cancels the variance amplification of the
    # detached path (cond and neg gradients are highly correlated, so the
    # covariance term reduces the effective variance).
    detach_neg = bool(getattr(args, "cfg_detach_neg", True))
    if detach_neg:
        with torch.no_grad():
            pred_neg = _flux2_transformer_forward(
                transformer,
                hidden_states.detach(),
                timestep,
                neg_embeds,
                neg_text_ids,
                img_ids,
            )
    else:
        pred_neg = _flux2_transformer_forward(
            transformer,
            hidden_states,
            timestep,
            neg_embeds,
            neg_text_ids,
            img_ids,
        )
    return pred_neg + cfg * (pred_cond - pred_neg)


def _encode_neg_prompt_for_cfg(
    args,
    device,
    text_encoder,
    text_tokenizer,
):
    """Pre-encode the empty negative prompt for CFG and stash on ``args``.

    Called once at the top of training. Returns nothing; sets
    ``args._neg_prompt_embeds`` and ``args._neg_text_ids`` if CFG is enabled.

    If the trainer is running in precomputed-embeds mode, Qwen3 is loaded
    briefly here and released after encoding -- only the small negative
    embedding tensor (~7-12 MB per sample) is retained.
    """
    cfg = float(getattr(args, "cfg_guidance", 1.0))
    if cfg <= 1.0:
        args._neg_prompt_embeds = None
        args._neg_text_ids = None
        return

    release_after = False
    if text_encoder is None or text_tokenizer is None:
        main_print(
            "--> CFG enabled (cfg_guidance > 1): loading Qwen3 briefly to encode "
            "the empty negative prompt for classifier-free guidance."
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to(device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        text_tokenizer = Qwen2TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        release_after = True

    neg_embeds = encode_prompts_qwen3(
        text_encoder=text_encoder,
        tokenizer=text_tokenizer,
        prompts=[""],
        device=device,
        dtype=torch.bfloat16,
        max_sequence_length=args.max_sequence_length,
        hidden_states_layers=tuple(args.text_encoder_out_layers),
    )
    # text_ids carry token-position indices (0..max_sequence_length-1) that
    # feed RoPE inside the transformer. bf16 only represents integers exactly
    # up to 256, so values >= 256 (we use max_sequence_length=512) would get
    # silently quantized (e.g. 257 -> 256) and produce a different positional
    # encoding from the inference pipeline (which keeps int64 ids). Keep them
    # in int64 to match `Flux2KleinPipeline._prepare_text_ids`.
    neg_text_ids = prepare_text_ids(
        batch_size=1,
        seq_len=neg_embeds.shape[1],
        device=device,
        dtype=torch.int64,
    )
    args._neg_prompt_embeds = neg_embeds
    args._neg_text_ids = neg_text_ids

    main_print(
        f"--> CFG negative prompt encoded: shape={tuple(neg_embeds.shape)}"
    )

    if release_after:
        del text_encoder
        del text_tokenizer
        torch.cuda.empty_cache()
        main_print("--> Released the temporarily-loaded Qwen3 text encoder.")


def sample_trajectory(
    args,
    device,
    transformer,
    fm_scheduler,
    prompt_embeds,
    text_ids,
    generator,
):
    """LeapAlign double-step rollout for FLUX.2.

    Selects two indices ``select_indices = [k_idx_from_end, j_idx_from_end]``
    (with ``k > j``), performs a Euler "jump" from step ``k`` to step ``j``,
    then continues with gradient until reaching the final pred_x0. The reward
    signal flows through the velocity at the jump-end step.
    """
    h, w = args.h, args.w
    sample_steps = args.sampling_steps

    # Sample the two LeapAlign anchor indices using the configured strategy.
    select_indices, _k_idx, _j_idx = sample_jk_indices(
        args, sample_steps, generator,
    )

    batch_size = prompt_embeds.shape[0]
    h_tokens = h // FLUX2_TOTAL_DOWNSAMPLE
    w_tokens = w // FLUX2_TOTAL_DOWNSAMPLE
    if h % FLUX2_TOTAL_DOWNSAMPLE or w % FLUX2_TOTAL_DOWNSAMPLE:
        raise ValueError(
            f"FLUX.2 requires h, w divisible by {FLUX2_TOTAL_DOWNSAMPLE}; "
            f"got h={h}, w={w}"
        )

    latents_xt = torch.randn(
        (batch_size, FLUX2_PATCH_CHANNELS, h_tokens, w_tokens),
        device=device,
        dtype=torch.bfloat16,
    )
    latents_xt = pack_latents(latents_xt)  # (B, n_tokens, 128)

    # Position ids must stay in an exact integer dtype: `Flux2PosEmbed` casts
    # them to float32 (`pos = ids.float()`) before computing RoPE, and bf16
    # cannot represent integers >= 256 exactly. Image ids only go up to
    # h_tokens-1 (typically <= 64) so bf16 happens to be lossless here, but
    # we use int64 anyway to match the inference pipeline exactly.
    img_ids = prepare_latent_ids(
        batch_size, h_tokens, w_tokens, device, torch.int64
    )

    sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
    image_seq_len = latents_xt.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=sample_steps)
    fm_scheduler.set_timesteps(sample_steps, device=device, sigmas=sigmas, mu=mu)
    timesteps = fm_scheduler.timesteps
    assert timesteps.shape[0] == sample_steps

    grad_on_ind_set = set(
        timesteps.size(0) - select_index.item() for select_index in select_indices
    )

    x_j_reference = None
    pred_xt2 = None
    latents_pred_x0 = None

    for idx, step_t in enumerate(timesteps):
        with nullcontext() if (idx in grad_on_ind_set) else torch.no_grad():
            timestep = step_t.expand(latents_xt.shape[0]).to(latents_xt.dtype)

            if idx == timesteps.size(0) - select_indices[1].item():
                with torch.no_grad():
                    pred_x2_weight_factor = (
                        torch.abs(pred_xt2.double() - latents_xt.double())
                        .mean(dim=tuple(range(1, pred_xt2.ndim)), keepdim=False)
                        .clip(min=args.tau)
                    )
                latents_xt = pred_xt2 + (latents_xt - pred_xt2).detach()
                if args.alpha <= 0.0:
                    dit_latents_xt = latents_xt.detach()
                else:
                    dit_latents_xt = (
                        args.alpha * latents_xt
                        + (1 - args.alpha) * latents_xt.detach()
                    )
            else:
                dit_latents_xt = latents_xt

            pred = _cfg_aware_forward(
                args,
                transformer,
                dit_latents_xt,
                timestep,
                prompt_embeds,
                text_ids,
                img_ids,
            )

            if idx == timesteps.size(0) - select_indices[0].item():
                t2_idx = timesteps.size(0) - select_indices[1].item()
                pred_xt2 = fm_scheduler.jump_to_step(
                    pred, step_t, latents_xt, t2_idx,
                )
                pred = pred.detach()
            if idx == timesteps.size(0) - select_indices[1].item():
                x_j_reference = latents_xt.detach()
                latents_xt, latents_pred_x0 = fm_scheduler.step(
                    pred, step_t, latents_xt, with_pred_x0=True,
                )
                latents_xt = latents_xt.detach()
            else:
                latents_xt = fm_scheduler.step(
                    pred, step_t, latents_xt, with_pred_x0=False,
                )[0]
            if latents_xt.dtype != torch.bfloat16:
                latents_xt = latents_xt.to(torch.bfloat16)

    if latents_pred_x0 is None:
        raise RuntimeError(
            "LeapAlign rollout did not hit the j-index step; check min_idx/max_idx "
            f"vs sampling_steps={sample_steps}"
        )

    with torch.no_grad():
        pred_x0_weight_factor = (
            torch.abs(latents_pred_x0.double() - latents_xt.double())
            .mean(dim=tuple(range(1, latents_pred_x0.ndim)), keepdim=False)
            .clip(min=args.tau)
        )
        traj_sim_weight_factor = pred_x2_weight_factor + pred_x0_weight_factor
        # Log w_sim as the reciprocal similarity weight 1/(d_j+d_0) to stay
        # consistent with flowbp_lagrange.py / sd35. The loss still divides by
        # ``traj_sim_weight_factor`` (the denominator), so training is unchanged.
        traj_metrics = {
            "d0": pred_x0_weight_factor.detach().mean(),
            "dj": pred_x2_weight_factor.detach().mean(),
            "w_sim": (1.0 / traj_sim_weight_factor).detach().mean(),
        }

    latents_pred_x0 = latents_pred_x0 + (latents_xt - latents_pred_x0).detach()

    if x_j_reference is not None and pred_xt2 is not None:
        d_j_dump = torch.abs(pred_xt2.double() - x_j_reference.double()).mean(
            dim=tuple(range(1, pred_xt2.ndim)), keepdim=False,
        )
        d_0_dump = torch.abs(latents_pred_x0.double() - latents_xt.double()).mean(
            dim=tuple(range(1, latents_pred_x0.ndim)), keepdim=False,
        )
        connector_payload = {
            "step": getattr(args, "current_train_step", None),
            "k_idx": timesteps.size(0) - select_indices[0].item(),
            "j_idx": timesteps.size(0) - select_indices[1].item(),
            "select_indices": select_indices.detach(),
            "xj_pred": pred_xt2.detach(),
            "xj": x_j_reference.detach(),
            "x0_pred": latents_pred_x0.detach(),
            "x0": latents_xt.detach(),
            "dj_per_sample": d_j_dump.detach(),
            "d0_per_sample": d_0_dump.detach(),
            "weight_scheme": "euler_jump",
        }
        _maybe_save_connector_dump(args, "leapalign_flux2", connector_payload)
        args._last_connector_payload = connector_payload

    return latents_xt, latents_pred_x0, traj_sim_weight_factor, traj_metrics


# -----------------------------------------------------------------------------
# Reward loss
# -----------------------------------------------------------------------------

def decode_packed_latents(
    args,
    vae,
    packed_latents: torch.Tensor,
) -> torch.Tensor:
    """packed -> unpack -> de-BN -> unpatchify -> VAE decode -> [0, 1] image."""
    h_tokens = args.h // FLUX2_TOTAL_DOWNSAMPLE
    w_tokens = args.w // FLUX2_TOTAL_DOWNSAMPLE
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latents = unpack_latents(packed_latents, h_tokens, w_tokens)
        bn_mean, bn_std = get_latent_bn_normalizers(vae, latents.dtype, latents.device)
        latents = latents * bn_std + bn_mean
        latents = unpatchify_latents(latents)
        image = vae.decode(latents, return_dict=False)[0]
        image = (image * 0.5 + 0.5).clamp(0, 1)
    return image


def get_reward_loss(
    args,
    latents_pred_x0,
    vae,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
    traj_sim_weight_factor,
):
    image_pred_x0 = decode_packed_latents(args, vae, latents_pred_x0)

    if args.use_hpsv2:
        image_pred_x0 = preprocess_val(image_pred_x0)
        text = tokenizer(caption).to(
            device=image_pred_x0.device, non_blocking=True,
        )
        with torch.amp.autocast("cuda"):
            outputs = reward_model(image_pred_x0, text)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            reward_pred_x0 = torch.einsum(
                "bc,bc->b", image_features, text_features,
            )
    else:
        raise NotImplementedError(
            "Only HPSv2 is currently supported for reward computation."
        )

    reward_loss = F.relu(-reward_pred_x0 + args.loss_relu_clip) * args.loss_grad_scale
    if traj_sim_weight_factor is not None:
        reward_loss = reward_loss / traj_sim_weight_factor
    reward_loss = torch.mean(reward_loss)
    return reward_loss, reward_pred_x0.detach().mean()


# -----------------------------------------------------------------------------
# Single training step
# -----------------------------------------------------------------------------

def encode_or_load_prompts(
    args,
    batch: dict,
    device,
    text_encoder,
    text_tokenizer,
):
    """Return ``(prompt_embeds, text_ids, caption_list)``.

    Uses precomputed tensors from the dataset when present; otherwise runs
    Qwen3 on the fly. The trainer's `args.use_precomputed_embeds` only
    governs which dataset is built — at the step level we trust the batch
    contents.
    """
    caption = batch["caption"]
    if "prompt_embed" in batch:
        prompt_embeds = batch["prompt_embed"].to(device=device, dtype=torch.bfloat16)
        # Keep text_ids as integer position indices; do NOT cast to bf16 (see
        # `_encode_neg_prompt_for_cfg` for the precision argument). Precomputed
        # tensors are already int64 on disk (`preprocess_flux2_embedding.py`).
        text_ids = batch["text_ids"].to(device=device, dtype=torch.int64)
        return prompt_embeds, text_ids, caption

    if text_encoder is None or text_tokenizer is None:
        raise RuntimeError(
            "Online prompt encoding requested but text_encoder/tokenizer was not loaded."
        )
    prompt_embeds = encode_prompts_qwen3(
        text_encoder=text_encoder,
        tokenizer=text_tokenizer,
        prompts=list(caption),
        device=device,
        dtype=torch.bfloat16,
        max_sequence_length=args.max_sequence_length,
        hidden_states_layers=tuple(args.text_encoder_out_layers),
    )
    text_ids = prepare_text_ids(
        batch_size=prompt_embeds.shape[0],
        seq_len=prompt_embeds.shape[1],
        device=device,
        dtype=torch.int64,
    )
    return prompt_embeds, text_ids, caption


def train_one_step(
    args,
    inner_step,
    device,
    transformer,
    vae,
    fm_scheduler,
    text_encoder,
    text_tokenizer,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    preprocess_val,
    select_idx_generator,
):
    batch = next(loader)
    prompt_embeds, text_ids, caption = encode_or_load_prompts(
        args, batch, device, text_encoder, text_tokenizer,
    )

    _, latents_pred_x0, traj_sim_weight_factor, traj_metrics = sample_trajectory(
        args,
        device,
        transformer,
        fm_scheduler,
        prompt_embeds,
        text_ids,
        select_idx_generator,
    )
    _maybe_log_connector_wandb(
        args, vae, "leapalign_flux2",
        getattr(args, "_last_connector_payload", None),
    )

    loss, avg_reward_pred_x0 = get_reward_loss(
        args,
        latents_pred_x0,
        vae,
        reward_model,
        tokenizer,
        caption,
        preprocess_val,
        traj_sim_weight_factor,
    )

    # Capture the un-scaled loss for wandb display before applying the CFG
    # gradient compensation below.
    avg_loss = loss.detach()

    # When CFG is active, the detached negative branch makes the effective
    # gradient through theta equal to cfg * grad_no_cfg. Compensating the
    # backward pass by 1/cfg recovers the non-CFG gradient magnitude, so the
    # reported grad_norm and the optimizer step are independent of
    # cfg_guidance. The forward output is unchanged. Toggle off via
    # ``--no-cfg_grad_norm_compensate`` (yaml: flux2.cfg_grad_norm_compensate);
    # with the flag disabled the raw cfg-amplified gradient is
    # used (equivalent to setting LR = base_lr * cfg_guidance).
    cfg_scale = float(getattr(args, "cfg_guidance", 1.0))
    cfg_compensate = bool(getattr(args, "cfg_grad_norm_compensate", True))
    if cfg_scale > 1.0 and cfg_compensate:
        backward_loss = loss / cfg_scale
    else:
        backward_loss = loss
    backward_loss.backward()
    dist.all_reduce(
        avg_loss.div_(args.gradient_accumulation_steps),
        op=dist.ReduceOp.AVG,
    )
    dist.all_reduce(
        avg_reward_pred_x0.div_(args.gradient_accumulation_steps),
        op=dist.ReduceOp.AVG,
    )

    reduced_traj_metrics = {}
    for metric_name, metric_value in traj_metrics.items():
        if metric_value is None:
            reduced_traj_metrics[metric_name] = None
            continue
        metric_tensor = metric_value.detach().float().to(device)
        dist.all_reduce(
            metric_tensor.div_(args.gradient_accumulation_steps),
            op=dist.ReduceOp.AVG,
        )
        reduced_traj_metrics[metric_name] = metric_tensor.item()

    if (inner_step + 1) % args.gradient_accumulation_steps == 0:
        grad_norm = transformer.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        grad_norm = grad_norm.item()
    else:
        grad_norm = None

    return (
        avg_loss.item(),
        avg_reward_pred_x0.item(),
        grad_norm,
        reduced_traj_metrics,
    )


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def _load_text_encoder(args, device):
    """Load and freeze the Qwen3 text encoder for FLUX.2.

    The encoder is kept in bf16 on the rank-local device. We do not wrap it in
    FSDP since it is frozen (no gradients) and re-evaluated each step.
    """
    main_print(f"--> loading Qwen3 text encoder from {args.pretrained_model_name_or_path}")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    text_tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    return text_encoder, text_tokenizer


def run_training(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    trainer_name = normalize_trainer_name(getattr(args, "trainer", "leapalign"))
    args.trainer = trainer_name
    if trainer_name not in {
        "leapalign",
        "refl",
        "flowbp_sparse",
        "flowbp_bridge",
        "flowbp_lagrange",
        "drtune",
        "draft_lv",
    }:
        raise ValueError(
            f"Unsupported trainer {trainer_name!r}. "
            "FLUX.2 currently supports 'leapalign', 'flowbp_sparse', "
            "'flowbp_bridge', 'flowbp_lagrange', 'refl', 'drtune', and "
            "'draft_lv'."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)
    if args.sp_size != 1:
        raise ValueError(
            "FLUX.2 FlowBP trainer only supports sp_size=1 today."
        )

    if args.debug:
        setup_debug()

    main_print("------------------args------------------")
    for arg_k, arg_v in vars(args).items():
        main_print(f"{arg_k}: {arg_v}")
    main_print("------------------args------------------")

    if args.seed is not None:
        set_seed(args.seed + rank)

    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Reward model (HPSv2)
    preprocess_val = None
    if args.use_hpsv2:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        model, _, hps_preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "./hps_ckpt/open_clip_pytorch_model.bin",
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        preprocess_val = HPSV2TransformsWithGrad(hps_preprocess_val)
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"
        checkpoint = torch.load(cp, map_location=f"cuda:{device}")
        model.load_state_dict(checkpoint["state_dict"])
        processor = get_tokenizer("ViT-H-14")
        reward_model = model.to(device)
        reward_model.requires_grad_(False)
        reward_model.eval()
    else:
        raise NotImplementedError(
            "Only HPSv2 is currently supported as the reward model."
        )

    main_print(f"--> loading transformer from {args.pretrained_model_name_or_path}")
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    fm_scheduler = FlowMatchEulerDiscreteWithPredX0Scheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    ema_model = None
    if args.use_ema:
        ema_model = deepcopy(transformer)

    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    transformer = FSDP(transformer, **fsdp_kwargs)

    if args.use_ema and ema_model is not None:
        ema_model = FSDP(ema_model, **fsdp_kwargs)
        ema_model.eval()
        ema_model.requires_grad_(False)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing,
        )

    main_print(f"--> loading VAE from {args.pretrained_model_name_or_path}")
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    use_precomputed = bool(getattr(args, "use_precomputed_embeds", False))
    if use_precomputed:
        main_print(
            "--> use_precomputed_embeds=True: skipping Qwen3 text encoder load"
        )
        text_encoder, text_tokenizer = None, None
    else:
        text_encoder, text_tokenizer = _load_text_encoder(args, device)

    # Encode the (empty) negative prompt once for classifier-free guidance.
    # Required for klein-base-4B / 9B (undistilled, guidance_embeds=false);
    # no-op for klein-4B (distilled, cfg_guidance=1).
    _encode_neg_prompt_for_cfg(args, device, text_encoder, text_tokenizer)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    main_print("--> model loaded")
    transformer.train()

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    if use_precomputed:
        train_dataset = PrecomputedFlux2Dataset(
            json_path=args.data_json_path,
            cfg_rate=args.cfg,
        )
        collate_fn = precomputed_flux2_collate_function
    else:
        train_dataset = PromptFlux2Dataset(
            json_path=args.data_json_path,
            cfg_rate=args.cfg,
            caption_key=getattr(args, "caption_key", "caption"),
        )
        collate_fn = prompt_flux2_collate_function
    sampler = DistributedSampler(
        train_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
        seed=args.sampler_seed if args.sampler_seed is not None else 0,
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    if args.select_idx_seed is not None:
        select_idx_generator = torch.Generator(device="cpu").manual_seed(args.select_idx_seed)
    else:
        select_idx_generator = None

    if rank <= 0:
    # Strip runtime-only attributes so wandb does not try to serialize tensors
    # cached on the argument namespace.
        wandb_config = {
            k: v for k, v in vars(args).items() if not k.startswith("_")
        }
        wandb.init(
            project=args.project,
            config=wandb_config,
            name=args.run_name,
        )

    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = "
        f"{sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    main_print(f"  Master weight dtype: {next(transformer.parameters()).dtype}")

    if args.resume_from_checkpoint:
        raise NotImplementedError("resume_from_checkpoint is not supported.")

    progress_bar = tqdm(
        range(1, args.max_train_steps + 1),
        initial=init_steps,
        desc="Steps",
        disable=local_rank > 0,
    )

    loader = flux2_dataloader_wrapper(
        train_dataloader,
        device=device,
        epoch_idx_auto_increment=True,
    )

    step_times: deque = deque(maxlen=100)

    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for step in range(init_steps + 1, args.max_train_steps + 1):
            args.current_train_step = step
            start_time = time.time()
            total_loss = 0.0
            total_reward_pred_x0 = 0.0
            traj_metrics: dict = {}

            for inner_step in range(args.gradient_accumulation_steps):
                if trainer_name == "flowbp_lagrange":
                    from flowbp.trainers.flux2.flowbp_lagrange import (
                        train_flowbp_lagrange_one_step_flux2,
                    )

                    step_fn = train_flowbp_lagrange_one_step_flux2
                elif trainer_name == "refl":
                    from flowbp.trainers.flux2.refl import (
                        train_refl_one_step_flux2,
                    )

                    step_fn = train_refl_one_step_flux2
                elif trainer_name == "flowbp_sparse":
                    from flowbp.trainers.flux2.flowbp_sparse import (
                        train_flowbp_sparse_one_step_flux2,
                    )

                    step_fn = train_flowbp_sparse_one_step_flux2
                elif trainer_name == "flowbp_bridge":
                    from flowbp.trainers.flux2.flowbp_bridge import (
                        train_flowbp_bridge_one_step_flux2,
                    )

                    step_fn = train_flowbp_bridge_one_step_flux2
                elif trainer_name == "drtune":
                    from flowbp.trainers.flux2.drtune import (
                        train_drtune_one_step_flux2,
                    )

                    step_fn = train_drtune_one_step_flux2
                elif trainer_name == "draft_lv":
                    from flowbp.trainers.flux2.draft_lv import (
                        train_draft_lv_one_step_flux2,
                    )

                    step_fn = train_draft_lv_one_step_flux2
                else:
                    step_fn = train_one_step

                (
                    avg_loss,
                    avg_reward_pred_x0,
                    grad_norm,
                    traj_metrics,
                ) = step_fn(
                    args,
                    inner_step,
                    device,
                    transformer,
                    vae,
                    fm_scheduler,
                    text_encoder,
                    text_tokenizer,
                    reward_model,
                    processor,
                    optimizer,
                    lr_scheduler,
                    loader,
                    preprocess_val,
                    select_idx_generator,
                )
                total_loss += avg_loss
                total_reward_pred_x0 += avg_reward_pred_x0

            log_traj_metrics = {
                f"train/{name}": value
                for name, value in traj_metrics.items()
                if value is not None
            }

            if args.use_ema and ema_model is not None:
                for tgt, src in zip(ema_model.parameters(), transformer.parameters()):
                    tgt.data.lerp_(src.data.to(tgt), 1 - args.ema_decay)

            if step % args.checkpointing_steps == 0:
                save_checkpoint(transformer, rank, args.output_dir, step, epoch)
                if args.use_ema:
                    save_sharded_ema_checkpoint(
                        ema_model, rank, args.output_dir, step, epoch,
                    )
                dist.barrier()

            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            if rank <= 0:
                log_dict = {
                    "train_loss": total_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                    "reward": total_reward_pred_x0,
                }
                log_dict.update(log_traj_metrics)
                wandb.log(log_dict, step=step)
            progress_bar.update(1)

            eval_interval = int(getattr(args, "evaluation_interval", 0) or 0)
            if eval_interval > 0 and (step % eval_interval == 0 or step == 1):
                transformer.eval()
                run_online_flux2_eval(
                    step,
                    args,
                    ema_model if args.use_ema and ema_model is not None else transformer,
                    vae,
                    rank,
                    world_size,
                    device,
                )
                transformer.train()

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


# -----------------------------------------------------------------------------
# Trainer class
# -----------------------------------------------------------------------------

class LeapAlignFlux2Trainer:
    """Trainer wrapper for the FLUX.2 LeapAlign baseline."""

    def __init__(self, config):
        self.args = config

    def train(self):
        self.args.trainer = "leapalign"
        return run_training(self.args)

    def train_one_step(
        self,
        inner_step,
        device,
        transformer,
        vae,
        fm_scheduler,
        text_encoder,
        text_tokenizer,
        reward_model,
        tokenizer,
        optimizer,
        lr_scheduler,
        loader,
        preprocess_val,
        select_idx_generator,
    ):
        return train_one_step(
            self.args,
            inner_step,
            device,
            transformer,
            vae,
            fm_scheduler,
            text_encoder,
            text_tokenizer,
            reward_model,
            tokenizer,
            optimizer,
            lr_scheduler,
            loader,
            preprocess_val,
            select_idx_generator,
        )
