import os

from safetensors.torch import save_file

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
from flowbp.utils.logging_ import main_print

def save_sharded_ema_checkpoint(ema_transformer, rank, output_dir, step, epoch):
    main_print(f"--> saving ema checkpoint at step {step}")
    with FSDP.state_dict_type(
            ema_transformer,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        cpu_state = ema_transformer.state_dict()
    if rank <= 0:
        ema_save_dir = os.path.join(output_dir, f"checkpoint-ema-{step}-{epoch}")
        os.makedirs(ema_save_dir, exist_ok=True)
        weight_path = os.path.join(ema_save_dir,
                                   "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)
        config_dict = dict(ema_transformer.config)
        if "dtype" in config_dict:
            del config_dict["dtype"]
        config_path = os.path.join(ema_save_dir, "config.json")
        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        main_print(f"--> EMA checkpoint saved at {ema_save_dir}")

    main_print(f"--> checkpoint saved at step {step}")
