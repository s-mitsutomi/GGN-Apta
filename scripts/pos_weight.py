#!/usr/bin/env python3
"""Print the released-model pos_weight: auto negatives/positives times PW_MODE (default 0.5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from Scripts.Sub.train_fixed_lr_scheduler_no_ddp_fsdp_allow_sched_factor_1_augment_eps_edge_dropout import (  # noqa: E402
        _filter_multi_fragment_positives,
        _normalize_to_uniprot,
        extract_dataset,
        load_positive_proteins,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fasta_file", required=True)
    parser.add_argument("--aptamer_list_file", required=True)
    parser.add_argument("--pw_mode", type=float, default=0.5)
    args = parser.parse_args()

    train_path = Path(args.train_fasta_file)
    list_path = Path(args.aptamer_list_file)
    if not train_path.is_file():
        print(f"ERROR: train FASTA not found: {train_path}", file=sys.stderr)
        return 1
    if not list_path.is_file():
        print(f"ERROR: aptamer list not found: {list_path}", file=sys.stderr)
        return 1
    if args.pw_mode <= 0:
        print(f"ERROR: pw_mode must be positive (got {args.pw_mode})", file=sys.stderr)
        return 1

    positive_set = load_positive_proteins(str(list_path))
    train_fasta_data = extract_dataset(str(train_path))
    dataset_ids = list(train_fasta_data.keys())
    positive_set, multi_fragment_excluded = _filter_multi_fragment_positives(
        positive_set, dataset_ids
    )
    filtered_ids = [
        protein_id
        for protein_id in dataset_ids
        if _normalize_to_uniprot(protein_id) not in multi_fragment_excluded
    ]
    num_positives = 0
    num_negatives = 0
    for protein_id in filtered_ids:
        if _normalize_to_uniprot(protein_id) in positive_set:
            num_positives += 1
        else:
            num_negatives += 1
    auto_weight = 1.0 if num_positives <= 0 else num_negatives / num_positives
    print(repr(float(auto_weight) * float(args.pw_mode)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
