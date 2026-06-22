import argparse
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
from diffusers import AutoencoderKL, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.utils import check_min_version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from flowbp.dataset.latent_flux_rl_datasets import (
    LatentDataset,
    latent_collate_function,
)
from flowbp.schedulers.fm_euler_discrete_with_pred_x0 import (
    FlowMatchEulerDiscreteWithPredX0Scheduler,
)
from flowbp.trainers.common.jk_sampling import sample_jk_indices
from flowbp.utils.checkpoint import save_checkpoint
from flowbp.utils.communications_flux import sp_parallel_dataloader_wrapper
from flowbp.utils.debug_utils import setup_debug
from flowbp.utils.ema_utils import save_sharded_ema_checkpoint
from flowbp.eval.runners.online import run_online_flux_eval
from flowbp.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
from flowbp.utils.hpsv2_transforms import HPSV2TransformsWithGrad
from flowbp.utils.logging_ import main_print
from flowbp.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

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
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latents = unpack_latents(packed_latents[:n].to(vae.device), args.h, args.w, 8)
        latents = (latents / 0.3611) + 0.1159
        images = vae.decode(latents, return_dict=False)[0]
        images = (images * 0.5 + 0.5).clamp(0, 1)
    return [
        image.detach().float().cpu().permute(1, 2, 0).numpy()
        for image in images
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
            args,
            vae,
            payload.get(name),
            max_samples,
        )
        if images:
            log_payload[f"connector/{prefix}_{name}"] = [
                wandb.Image(image, caption=f"{prefix} {name} sample {idx}")
                for idx, image in enumerate(images)
            ]

    if log_payload:
        wandb.log(log_payload, step=int(step))


def sample_trajectory(
    args,
    device, 
    transformer,
    fm_scheduler,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    generator,
):
    w, h = args.w, args.h
    sample_steps = args.sampling_steps

    # Sample the two LeapAlign anchor indices using the configured strategy.
    select_indices, _k_idx, _j_idx = sample_jk_indices(
        args, sample_steps, generator,
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    latents_xt = torch.randn(
        (B, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
        device=device,
        dtype=torch.bfloat16,
    )
    latents_xt = pack_latents(
        latents_xt, 
        B,
        IN_CHANNELS, 
        latent_h, 
        latent_w,
    )
    image_ids = prepare_latent_image_ids(
        B,
        latent_h // 2,
        latent_w // 2, 
        device, 
        torch.bfloat16,
    )
    cfg_guidance = torch.tensor(
        [args.cfg_guidance],
        device=latents_xt.device,
        dtype=torch.bfloat16
    )
    cfg_guidance = cfg_guidance.expand(latents_xt.shape[0])

    sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
    mu = calculate_shift(
        latents_xt.shape[1],
        fm_scheduler.config.base_image_seq_len,
        fm_scheduler.config.max_image_seq_len,
        fm_scheduler.config.base_shift,
        fm_scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        fm_scheduler,
        sample_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    assert num_inference_steps == sample_steps

    grad_on_ind_set = set(
        timesteps.size(0) - select_index.item()
        for select_index in select_indices
    )

    x_j_reference = None

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
                # w/o grad -> w/ grad
                latents_xt = pred_xt2 + (latents_xt - pred_xt2).detach()
                if args.alpha <= 0.0:
                    dit_latents_xt = latents_xt.detach()
                else:
                    dit_latents_xt = (
                        args.alpha * latents_xt +
                        (1 - args.alpha) * latents_xt.detach()
                    )
            else:
                dit_latents_xt = latents_xt

            with torch.autocast("cuda", torch.bfloat16):
                pred= transformer(
                    hidden_states=dit_latents_xt,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep/1000,
                    guidance=cfg_guidance,
                    txt_ids=text_ids,
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            
            if idx == timesteps.size(0) - select_indices[0].item():
                t2_idx = timesteps.size(0) - select_indices[1].item()
                pred_xt2 = fm_scheduler.jump_to_step(
                    pred, 
                    step_t, 
                    latents_xt, 
                    t2_idx,
                )
                pred = pred.detach()
            if idx == timesteps.size(0) - select_indices[1].item():
                x_j_reference = latents_xt.detach()
                latents_xt, latents_pred_x0 = fm_scheduler.step(
                    pred, 
                    step_t, 
                    latents_xt, 
                    with_pred_x0=True,
                )
                latents_xt = latents_xt.detach()
            else:
                latents_xt = fm_scheduler.step(
                    pred, 
                    step_t, 
                    latents_xt, 
                    with_pred_x0=False,
                )[0]
            if latents_xt.dtype != torch.bfloat16:
                latents_xt = latents_xt.to(torch.bfloat16)

    with torch.no_grad():
        pred_x0_weight_factor = (
            torch.abs(latents_pred_x0.double() - latents_xt.double())
            .mean(dim=tuple(range(1, latents_pred_x0.ndim)), keepdim=False)
            .clip(min=args.tau)
        )
        traj_sim_weight_factor = pred_x2_weight_factor + pred_x0_weight_factor
        # Log w_sim as the reciprocal similarity weight 1/(d_j+d_0) to stay
        # consistent with flowbp_lagrange.py. The loss still divides by
        # ``traj_sim_weight_factor`` (the denominator), so training is unchanged.
        traj_metrics = {
            "d0": pred_x0_weight_factor.detach().mean(),
            "dj": pred_x2_weight_factor.detach().mean(),
            "w_sim": (1.0 / traj_sim_weight_factor).detach().mean(),
        }

    latents_pred_x0 = latents_pred_x0 + (latents_xt - latents_pred_x0).detach()

    if x_j_reference is not None:
        d_j_dump = torch.abs(pred_xt2.double() - x_j_reference.double()).mean(
            dim=tuple(range(1, pred_xt2.ndim)),
            keepdim=False,
        )
        d_0_dump = torch.abs(latents_pred_x0.double() - latents_xt.double()).mean(
            dim=tuple(range(1, latents_pred_x0.ndim)),
            keepdim=False,
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
        _maybe_save_connector_dump(args, "leapalign", connector_payload)
        args._last_connector_payload = connector_payload

    return latents_xt, latents_pred_x0, traj_sim_weight_factor, traj_metrics



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
    w, h = args.w, args.h
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latents_pred_x0 = unpack_latents(latents_pred_x0, h, w, 8)
        latents_pred_x0 = (latents_pred_x0 / 0.3611) + 0.1159
        # b,c,h,w
        image_pred_x0 = vae.decode(latents_pred_x0, return_dict=False)[0]
        image_pred_x0 = (image_pred_x0 * 0.5 + 0.5).clamp(0, 1)
    
    if args.use_hpsv2:
        image_pred_x0 = preprocess_val(image_pred_x0)
        # Process the prompt
        text = tokenizer(caption).to(
            device=image_pred_x0.device, non_blocking=True,
        )
        # Calculate the HPS
        with torch.amp.autocast('cuda'):
            outputs = reward_model(image_pred_x0, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            reward_pred_x0 = torch.einsum(
                "bc,bc->b",
                image_features,
                text_features,
            )
    else:
        raise NotImplementedError("Only HPSv2 is currently supported for reward computation.")
    
    reward_loss = F.relu(-reward_pred_x0 + args.loss_relu_clip) * args.loss_grad_scale
    if traj_sim_weight_factor is not None:
        reward_loss = reward_loss / traj_sim_weight_factor
    reward_loss = torch.mean(reward_loss)
    return (
        reward_loss, 
        reward_pred_x0.detach().mean(), 
    )

def train_one_step(
    args,
    inner_step,
    device,
    transformer,
    vae,
    fm_scheduler,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    preprocess_val,
    select_idx_generator,
):
    (
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        caption,
    ) = next(loader)
    #device = latents.device
    text_ids = torch.zeros(
        encoder_hidden_states.shape[1], 3
    ).to(device=device, dtype=text_ids.dtype)

    _, latents_pred_x0, traj_sim_weight_factor, traj_metrics = sample_trajectory(
        args,
        device, 
        transformer,
        fm_scheduler,
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        select_idx_generator,
    )
    _maybe_log_connector_wandb(
        args,
        vae,
        "leapalign",
        getattr(args, "_last_connector_payload", None),
    )
    # loss: 0
    # avg_reward_pred_x0: 0
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

    loss.backward()
    avg_loss = loss.detach()
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
    
    if (inner_step+1)%args.gradient_accumulation_steps==0:
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



def run_training(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    trainer_name = normalize_trainer_name(getattr(args, "trainer", "leapalign"))
    args.trainer = trainer_name
    if trainer_name not in SUPPORTED_INTERNAL_TRAINERS:
        raise ValueError(
            f"Unsupported trainer {trainer_name!r}. "
            "Expected 'leapalign', 'refl', 'flowbp_lagrange', "
            "'flowbp_sparse', 'flowbp_bridge', 'drtune', or 'draft_lv'."
        )
    if int(getattr(args, "sp_size", 1)) != 1:
        raise NotImplementedError(
            "FLUX.1 FlowBP trainer currently supports sp_size=1 only. "
            "The sequence-parallel path is disabled until its DP sampler "
            "and SP group semantics are fixed."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if args.debug:
        setup_debug()
    
    main_print('------------------args------------------')
    for arg_k, arg_v in vars(args).items():
        main_print(f'{arg_k}: {arg_v}')
    main_print('------------------args------------------')

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    preprocess_val = None
    if args.use_hpsv2:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                './hps_ckpt/open_clip_pytorch_model.bin',
                precision='amp',
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
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        preprocess_val = HPSV2TransformsWithGrad(preprocess_val)
        #cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location=f'cuda:{device}')
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.requires_grad_(False)
        reward_model.eval()
    else:
        raise NotImplementedError("Only HPSv2 is currently supported as the reward model.")

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float32
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
            transformer, no_split_modules, args.selective_checkpointing
        )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

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

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    if args.select_idx_seed is not None:
        select_idx_generator = torch.Generator(device='cpu').manual_seed(args.select_idx_seed)
    else:
        select_idx_generator = None

    init_steps = 0

    #vae.enable_tiling()

    if rank <= 0:
        wandb.init(
            project=args.project, 
            config=args,
            name=args.run_name,
        )

    # Train!
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
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {next(transformer.parameters()).dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        raise NotImplementedError("resume_from_checkpoint is not supported.")

    progress_bar = tqdm(
        range(1, args.max_train_steps+1),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
        epoch_idx_auto_increment=True, # auto increment epoch index for each epoch, make sure that the index sequence is different for each epoch
    )

    step_times = deque(maxlen=100)

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch

        for step in range(init_steps+1, args.max_train_steps+1):
            args.current_train_step = step
            start_time = time.time()
            total_loss = 0.0
            total_reward_pred_x0 = 0.0

            for inner_step in range(args.gradient_accumulation_steps):
                if trainer_name == "refl":
                    from flowbp.trainers.flux1.refl import train_refl_one_step

                    step_fn = train_refl_one_step
                elif trainer_name == "flowbp_lagrange":
                    from flowbp.trainers.flux1.flowbp_lagrange import (
                        train_flowbp_lagrange_one_step,
                    )

                    step_fn = train_flowbp_lagrange_one_step
                elif trainer_name == "flowbp_sparse":
                    from flowbp.trainers.flux1.flowbp_sparse import (
                        train_flowbp_sparse_one_step,
                    )

                    step_fn = train_flowbp_sparse_one_step
                elif trainer_name == "flowbp_bridge":
                    from flowbp.trainers.flux1.flowbp_bridge import (
                        train_flowbp_bridge_one_step,
                    )

                    step_fn = train_flowbp_bridge_one_step
                elif trainer_name == "drtune":
                    from flowbp.trainers.flux1.drtune import (
                        train_drtune_one_step,
                    )

                    step_fn = train_drtune_one_step
                elif trainer_name == "draft_lv":
                    from flowbp.trainers.flux1.draft_lv import (
                        train_draft_lv_one_step,
                    )

                    step_fn = train_draft_lv_one_step
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
                save_checkpoint(transformer, rank, args.output_dir,
                                step, epoch)
                if args.use_ema:
                    save_sharded_ema_checkpoint(
                        ema_model, 
                        rank, 
                        args.output_dir, 
                        step, 
                        epoch,
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

            if step % args.evaluation_interval == 0 or step == 1:
                transformer.eval()
                run_online_flux_eval(
                    step,
                    args, 
                    ema_model if args.use_ema else transformer,
                    vae,
                    rank,
                    world_size,
                    device,
                )
                transformer.train()

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()





class LeapAlignFluxTrainer:
    """Trainer wrapper for the FLUX.1 LeapAlign baseline.

    The algorithmic helpers in this module are kept close to the trainer so the
    entrypoint can stay thin while preserving the original training behavior.
    """

    def __init__(self, config):
        self.args = config

    def train(self):
        return run_training(self.args)

    def sample_trajectory(self, device, transformer, fm_scheduler, encoder_hidden_states, pooled_prompt_embeds, text_ids, generator):
        return sample_trajectory(
            self.args,
            device,
            transformer,
            fm_scheduler,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            generator,
        )

    def compute_reward_loss(self, latents_pred_x0, vae, reward_model, tokenizer, caption, preprocess_val, traj_sim_weight_factor):
        return get_reward_loss(
            self.args,
            latents_pred_x0,
            vae,
            reward_model,
            tokenizer,
            caption,
            preprocess_val,
            traj_sim_weight_factor,
        )

    def train_one_step(self, inner_step, device, transformer, vae, fm_scheduler, reward_model, tokenizer, optimizer, lr_scheduler, loader, preprocess_val, select_idx_generator):
        return train_one_step(
            self.args,
            inner_step,
            device,
            transformer,
            vae,
            fm_scheduler,
            reward_model,
            tokenizer,
            optimizer,
            lr_scheduler,
            loader,
            preprocess_val,
            select_idx_generator,
        )
