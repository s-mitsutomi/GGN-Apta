"""Released-model positive weight: auto negatives/positives times pw_mode (default 0.5)."""

from __future__ import annotations

import sys
from pathlib import Path


class CliError(Exception):
    """User-facing failure. The message is printed as-is."""


def require_positive_pw_mode(pw_mode: float) -> None:
    if pw_mode <= 0:
        raise CliError(f"ERROR: --pw-mode must be positive (got {pw_mode})")


def positive_weight_text(
    train_fasta_file: str,
    aptamer_list_file: str,
    pw_mode: float,
) -> str:
    """Return the pos_weight text the trainer receives, matching the old script's repr."""
    require_positive_pw_mode(pw_mode)
    train_path = Path(train_fasta_file)
    list_path = Path(aptamer_list_file)
    if not train_path.is_file():
        raise CliError(f"ERROR: --train-fasta is missing: {train_path}")
    if not list_path.is_file():
        raise CliError(f"ERROR: --aptamer-list is missing: {list_path}")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from Scripts.Sub.train_fixed_lr_scheduler_no_ddp_fsdp_allow_sched_factor_1_augment_eps_edge_dropout import (  # noqa: E402
        _filter_multi_fragment_positives,
        _normalize_to_uniprot,
        extract_dataset,
        load_positive_proteins,
    )

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
    return repr(float(auto_weight) * float(pw_mode))
