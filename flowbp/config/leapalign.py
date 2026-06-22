from __future__ import annotations

# Compatibility layer for older scripts that still import the LeapAlign-named
# config module. New code should import flowbp.config.flowbp.
from flowbp.config.flowbp import (
    FLOWBP_TRAINER_CHOICES,
    SUPPORTED_INTERNAL_TRAINERS,
    FlowBPConfig,
    FlowBPEvalConfig,
    load_config_defaults,
    normalize_trainer_name,
)

LeapAlignConfig = FlowBPConfig
LeapAlignEvalConfig = FlowBPEvalConfig
