#!/usr/bin/env python3
"""Load one released-model checkpoint with the training module and require every key to match."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from Scripts.Sub.train_fixed_lr_scheduler_no_ddp_fsdp_allow_sched_factor_1_augment_eps_edge_dropout import (  # noqa: E402
        build_model,
        get_config,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    config = get_config()
    config.hidden_dim = 640
    config.num_layers = 4
    config.dropout = 0.60
    config.augment_eps = 0.60
    config.pooling = "GAP"
    config.contrast = True
    config.use_residual = False
    config.residual_alpha = 1.0
    config.plddt_pool_weight = False

    device = torch.device("cpu")
    model = build_model(config, 1, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    print(f"OK {checkpoint_path.name} keys={len(state_dict)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
