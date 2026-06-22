from flowbp.trainers.flux1.leapalign import LeapAlignFluxTrainer
from flowbp.trainers.flux1.flowbp_lagrange import FlowBPLagrangeFluxTrainer
from flowbp.trainers.flux1.refl import LeapAlignReFLFluxTrainer
from flowbp.trainers.flux1.flowbp_sparse import FlowBPSparseFluxTrainer
from flowbp.trainers.flux1.flowbp_bridge import FlowBPBridgeFluxTrainer
from flowbp.trainers.flux1.drtune import LeapAlignDRTuneFluxTrainer
from flowbp.trainers.flux1.draft_lv import LeapAlignDRaFTLVFluxTrainer

# FLUX.2 trainers require diffusers >= 0.37 (Flux2Transformer2DModel /
# AutoencoderKLFlux2 / Qwen3 text encoder). Keep the import lazy so that
# environments still pinned to the FLUX.1 era of diffusers can continue to
# load the FLUX.1 trainers.
try:
    from flowbp.trainers.flux2.leapalign import LeapAlignFlux2Trainer
    from flowbp.trainers.flux2.flowbp_lagrange import FlowBPLagrangeFlux2Trainer
    from flowbp.trainers.flux2.refl import LeapAlignReFLFlux2Trainer
    from flowbp.trainers.flux2.flowbp_sparse import FlowBPSparseFlux2Trainer
    from flowbp.trainers.flux2.flowbp_bridge import FlowBPBridgeFlux2Trainer
    from flowbp.trainers.flux2.drtune import LeapAlignDRTuneFlux2Trainer
    from flowbp.trainers.flux2.draft_lv import LeapAlignDRaFTLVFlux2Trainer

    _FLUX2_AVAILABLE = True
except ImportError:
    LeapAlignFlux2Trainer = None
    LeapAlignReFLFlux2Trainer = None
    LeapAlignDRTuneFlux2Trainer = None
    LeapAlignDRaFTLVFlux2Trainer = None
    FlowBPSparseFlux2Trainer = None
    FlowBPBridgeFlux2Trainer = None
    FlowBPLagrangeFlux2Trainer = None
    _FLUX2_AVAILABLE = False


# SD3.5 trainers require diffusers SD3 components. Keep this optional for
# environments that only run FLUX.1/FLUX.2 code.
try:
    from flowbp.trainers.sd35 import (
        FlowBPBridgeSD35Trainer,
        FlowBPLagrangeSD35Trainer,
        FlowBPSparseSD35Trainer,
        LeapAlignDRTuneSD35Trainer,
        LeapAlignDRaFTLVSD35Trainer,
        LeapAlignReFLSD35Trainer,
        LeapAlignSD35Trainer,
    )

    _SD35_AVAILABLE = True
except ImportError:
    FlowBPBridgeSD35Trainer = None
    FlowBPLagrangeSD35Trainer = None
    FlowBPSparseSD35Trainer = None
    LeapAlignDRTuneSD35Trainer = None
    LeapAlignDRaFTLVSD35Trainer = None
    LeapAlignReFLSD35Trainer = None
    LeapAlignSD35Trainer = None
    _SD35_AVAILABLE = False


__all__ = [
    "LeapAlignFluxTrainer",
    "LeapAlignReFLFluxTrainer",
    "LeapAlignDRTuneFluxTrainer",
    "LeapAlignDRaFTLVFluxTrainer",
    "FlowBPSparseFluxTrainer",
    "FlowBPBridgeFluxTrainer",
    "FlowBPLagrangeFluxTrainer",
    "LeapAlignFlux2Trainer",
    "LeapAlignReFLFlux2Trainer",
    "LeapAlignDRTuneFlux2Trainer",
    "LeapAlignDRaFTLVFlux2Trainer",
    "FlowBPSparseFlux2Trainer",
    "FlowBPBridgeFlux2Trainer",
    "FlowBPLagrangeFlux2Trainer",
    "LeapAlignSD35Trainer",
    "LeapAlignReFLSD35Trainer",
    "LeapAlignDRaFTLVSD35Trainer",
    "LeapAlignDRTuneSD35Trainer",
    "FlowBPSparseSD35Trainer",
    "FlowBPBridgeSD35Trainer",
    "FlowBPLagrangeSD35Trainer",
]
