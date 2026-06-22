from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flowbp.config.flowbp import load_config_defaults
from flowbp.train_flowbp_flux import build_parser


def normalize_sd35_trainer_name(name: str | None) -> str:
    raw_name = str(name or "leapalign").lower().replace("-", "_")
    aliases = {
        "flowbp_sparse": "flowbp_sparse",
        "flowbp_bridge": "flowbp_bridge",
        "flowbp_lagrange": "flowbp_lagrange",
    }
    return aliases.get(raw_name, raw_name)


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    parser.description = "FlowBP SD3.5 fine-tuning entrypoint."
    probe_args, _ = parser.parse_known_args()
    if probe_args.config:
        parser.set_defaults(**load_config_defaults(probe_args.config))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from flowbp.trainers.sd35 import (
        FlowBPBridgeSD35Trainer,
        FlowBPLagrangeSD35Trainer,
        FlowBPSparseSD35Trainer,
        LeapAlignDRTuneSD35Trainer,
        LeapAlignDRaFTLVSD35Trainer,
        LeapAlignReFLSD35Trainer,
        LeapAlignSD35Trainer,
    )

    trainer_map = {
        "leapalign": LeapAlignSD35Trainer,
        "refl": LeapAlignReFLSD35Trainer,
        "flowbp_lagrange": FlowBPLagrangeSD35Trainer,
        "flowbp_sparse": FlowBPSparseSD35Trainer,
        "flowbp_bridge": FlowBPBridgeSD35Trainer,
        "draft_lv": LeapAlignDRaFTLVSD35Trainer,
        "drtune": LeapAlignDRTuneSD35Trainer,
    }
    trainer_name = normalize_sd35_trainer_name(args.trainer)
    if trainer_name not in trainer_map:
        raise ValueError(
            "SD3.5 entrypoint supports flowbp_sparse/flowbp_bridge/"
            "flowbp_lagrange and baselines leapalign/refl/draft_lv/drtune, "
            f"got {args.trainer!r}."
        )
    trainer_map[trainer_name](args).train()


if __name__ == "__main__":
    main()
