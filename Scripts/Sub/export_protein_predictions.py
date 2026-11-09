#!/usr/bin/env python3
"""
Export per-protein prediction probabilities for a FASTA set (all folds).

This script loads saved K-fold models, runs predictions on the given FASTA file,
and writes a TSV with columns:
  protein_id, label, proba_fold1, ..., proba_foldK, ensemble_proba

Uses the same ensemble as test evaluation (evaluate_test_set.py): exactly one
model per fold (prefer early-stopped *_es.pt), ensemble_proba = mean(probas).
So TSV ensemble_proba matches test evaluation ensemble_metrics when the same
model_dir and label logic are used.

Expects unique sequence IDs in the FASTA (one row per ID). Model files are
one per fold, sorted by fold number (fold1, fold2, ...) so column order is consistent.

Uses extract_dataset from train_fixed, which returns a dict mapping
sequence ID -> sequence string (not a list of objects).

Consistency: The nl×cw×lr sweep is run with Scripts/Sub/train_fixed_lr_scheduler.py.
This script imports from train_fixed; the interfaces (extract_dataset, get_data_args,
ProteinGraphDataset, get_config, GGN_GO, labels_from_batch_names,
_filter_multi_fragment_positives) are the same in train_fixed and train_fixed_lr_scheduler,
so loading and inference are consistent with the trained checkpoints.

Usage:
    python export_protein_predictions.py \\
        --model_dir Model/Hyperparameter_Sweep/.../Sampling_strat_no_repl_stop/ \\
        --fasta_file Fasta/.../sequences.fasta.gz \\
        --aptamer_list_file Lists/aptamer_target_proteins_human_2024-08-01.txt \\
        --output_predictions_tsv Results/protein_predictions.tsv
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Add project root to path (same as evaluate_test_set.py)
_script_dir = Path(__file__).resolve().parent
for _parent in [_script_dir, _script_dir.parent, _script_dir.parent.parent]:
    if (_parent / "Scripts").is_dir():
        sys.path.insert(0, str(_parent))
        break
else:
    sys.path.insert(0, str(_script_dir.parent.parent.parent))

from Scripts.Sub.ef10_checkpoint import CheckpointNotFound, resolve_fold_checkpoints
from Scripts.Sub.train_fixed import (
    extract_dataset,
    get_data_args,
    ProteinGraphDataset,
    labels_from_batch_names,
    _filter_multi_fragment_positives,
)
from Scripts.Sub.train_fixed_lr_scheduler_no_ddp_fsdp_allow_sched_factor_1_augment_eps_edge_dropout import (
    GGN_GO,
    build_model,
    get_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_positive_set(aptamer_list_file: str) -> set:
    """Load positive set from aptamer list file."""
    positive_set = set()
    with open(aptamer_list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                positive_set.add(line)
    logger.info(f"Loaded {len(positive_set)} positive UniProt IDs")
    return positive_set


def load_model(model_path: str, device: torch.device, config) -> torch.nn.Module:
    """Load a saved model checkpoint."""
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    output_dim = 1
    model = build_model(config, output_dim, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    positive_set: set,
    device: torch.device,
    config,
) -> List[Tuple[str, int, float]]:
    """Run model on loader; return list of (protein_id, label, proba)."""
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            labels = labels_from_batch_names(batch.name, positive_set, device)
            if config.contrast:
                y_pred, _, _ = model(
                    h_V, batch.edge_index, h_E, batch.seq, batch.batch,
                    node_weight=getattr(batch, "plddt_w", None) if getattr(config, "plddt_pool_weight", False) else None,
                )
            else:
                y_pred = model(
                    h_V, batch.edge_index, h_E, batch.seq, batch.batch,
                    node_weight=getattr(batch, "plddt_w", None) if getattr(config, "plddt_pool_weight", False) else None,
                )
            # Model readout already includes Sigmoid (train_fixed.GGN_GO.readout); do not apply again.
            proba = y_pred.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            for name, label, p in zip(batch.name, labels_np, proba):
                out.append((name, int(label), float(p)))
    return out


def _fold_number(path: Path) -> int:
    """Extract fold index (1-based) for stable sort."""
    fold_part = path.stem.split("fold")[-1]
    return int(fold_part.lstrip("_").split("_")[0])


def _resolve_model_files(model_dir_path: Path, checkpoint_mode: str) -> List[Path]:
    """Resolve one checkpoint per fold (same policy as evaluate_test_set.py).

    ``es`` is only ``*_es.pt``. ``es_ef10`` is only ``*_es_ef10.pt``. ``final``
    excludes both. If ``es`` finds nothing, fall back to final checkpoints so
    older runs without early-stop files still export.
    """
    try:
        return resolve_fold_checkpoints(model_dir_path, checkpoint_mode)
    except CheckpointNotFound:
        if checkpoint_mode != "es":
            raise
        return resolve_fold_checkpoints(model_dir_path, "final")


def build_loader(
    fasta_file: str,
    config,
    prottrans_output_path: str,
    esm2_output_path: str,
    pdb_dir: str,
    proc_pdb_output_path: str,
    dssp_output_path: str,
    num_workers: int,
    batch_size: int,
    disable_prottrans: bool = False,
    disable_esm2: bool = False,
    disable_dssp: bool = False,
    disable_dihedrals: bool = False,
    disable_orientations: bool = False,
    disable_sidechains: bool = False,
    disable_rbf: bool = False,
    disable_pos_embeddings: bool = False,
    disable_edge_direction: bool = False,
    plddt_geometry_downweight: bool = False,
    plddt_dir: str = None,
    plddt_weight_mode: str = "hard",
    plddt_thresh: float = 50.0,
    plddt_soft_lo: float = 40.0,
    plddt_soft_hi: float = 60.0,
    plddt_edge_mode: str = "scale",
    plddt_pool_weight: bool = False,
    plddt_knn_keep_min: float = 0.5,
    fasta_data=None,
) -> DataLoader:
    """Build ProteinGraphDataset and DataLoader for a FASTA file.
    If fasta_data is provided, use it and avoid re-reading the FASTA.
    """
    if fasta_data is None:
        fasta_data = extract_dataset(fasta_file)
    logger.info(f"Loaded {len(fasta_data)} sequences from {fasta_file}")
    args = get_data_args(
        split="train",
        fasta_file=fasta_file,
        prottrans_output_path=prottrans_output_path,
        esm2_output_path=esm2_output_path,
        pdb_dir=pdb_dir,
        proc_pdb_output_path=proc_pdb_output_path,
        dssp_output_path=dssp_output_path,
        num_workers=num_workers,
        disable_prottrans=disable_prottrans,
        disable_esm2=disable_esm2,
        disable_dssp=disable_dssp,
        disable_dihedrals=disable_dihedrals,
        disable_orientations=disable_orientations,
        disable_sidechains=disable_sidechains,
        disable_rbf=disable_rbf,
        disable_pos_embeddings=disable_pos_embeddings,
        disable_edge_direction=disable_edge_direction,
        plddt_geometry_downweight=plddt_geometry_downweight,
        plddt_dir=plddt_dir,
        plddt_weight_mode=plddt_weight_mode,
        plddt_thresh=plddt_thresh,
        plddt_soft_lo=plddt_soft_lo,
        plddt_soft_hi=plddt_soft_hi,
        plddt_edge_mode=plddt_edge_mode,
        plddt_pool_weight=plddt_pool_weight,
        plddt_knn_keep_min=plddt_knn_keep_min,
    )
    dataset = ProteinGraphDataset(
        fasta_data,
        range(len(fasta_data)),
        args,
        task_list="bp",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        prefetch_factor=1,
    )


def export_predictions(
    model_dir: str,
    fasta_file: str,
    aptamer_list_file: str,
    config,
    device: torch.device,
    prottrans_output_path: str,
    esm2_output_path: str,
    pdb_dir: str,
    proc_pdb_output_path: str,
    dssp_output_path: str,
    num_workers: int,
    batch_size: int,
    disable_prottrans: bool,
    disable_esm2: bool,
    disable_dssp: bool,
    output_tsv: str,
    disable_dihedrals: bool = False,
    disable_orientations: bool = False,
    disable_sidechains: bool = False,
    disable_rbf: bool = False,
    disable_pos_embeddings: bool = False,
    disable_edge_direction: bool = False,
    plddt_geometry_downweight: bool = False,
    plddt_dir: str = None,
    plddt_weight_mode: str = "hard",
    plddt_thresh: float = 50.0,
    plddt_soft_lo: float = 40.0,
    plddt_soft_hi: float = 60.0,
    plddt_edge_mode: str = "scale",
    plddt_pool_weight: bool = False,
    plddt_knn_keep_min: float = 0.5,
    checkpoint_mode: str = "es",
) -> None:
    """Run all folds on fasta_file, write TSV with labels."""
    positive_set = load_positive_set(aptamer_list_file)
    fasta_data = extract_dataset(fasta_file)
    all_ids = list(fasta_data.keys())
    positive_set, multi_fragment_positives = _filter_multi_fragment_positives(
        positive_set, all_ids
    )
    if multi_fragment_positives:
        logger.info(
            f"Filtered out {len(multi_fragment_positives)} multi-fragment positives"
        )

    if not fasta_data:
        raise ValueError(f"No sequences in FASTA: {fasta_file}")

    loader = build_loader(
        fasta_file,
        config,
        prottrans_output_path,
        esm2_output_path,
        pdb_dir,
        proc_pdb_output_path,
        dssp_output_path,
        num_workers,
        batch_size,
        disable_prottrans=disable_prottrans,
        disable_esm2=disable_esm2,
        disable_dssp=disable_dssp,
        disable_dihedrals=disable_dihedrals,
        disable_orientations=disable_orientations,
        disable_sidechains=disable_sidechains,
        disable_rbf=disable_rbf,
        disable_pos_embeddings=disable_pos_embeddings,
        disable_edge_direction=disable_edge_direction,
        plddt_geometry_downweight=plddt_geometry_downweight,
        plddt_dir=plddt_dir,
        plddt_weight_mode=plddt_weight_mode,
        plddt_thresh=plddt_thresh,
        plddt_soft_lo=plddt_soft_lo,
        plddt_soft_hi=plddt_soft_hi,
        plddt_edge_mode=plddt_edge_mode,
        plddt_pool_weight=plddt_pool_weight,
        plddt_knn_keep_min=plddt_knn_keep_min,
        fasta_data=fasta_data,
    )

    model_dir_path = Path(model_dir)
    model_files = _resolve_model_files(model_dir_path, checkpoint_mode)
    K = len(model_files)
    logger.info(f"Using ensemble of {K} models (one per fold), same as test evaluation")

    # protein_id -> {"label": int, "probas": [proba_fold1, ...]}
    records = defaultdict(lambda: {"label": None, "probas": []})

    for fold_idx, model_file in enumerate(model_files):
        fold_num = int(model_file.stem.split("fold")[-1].split("_")[0])
        logger.info(f"Fold {fold_num} (file {fold_idx + 1}/{K})")
        model = load_model(str(model_file), device, config)
        preds = run_predictions(model, loader, positive_set, device, config)
        for name, label, proba in preds:
            if records[name]["label"] is None:
                records[name]["label"] = label
            records[name]["probas"].append(proba)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name, data in records.items():
        if len(data["probas"]) != K:
            raise RuntimeError(
                f"Protein {name} has {len(data['probas'])} fold predictions, expected {K}"
            )

    fold_headers = [f"proba_fold{i+1}" for i in range(K)]
    header = ["protein_id", "label"] + fold_headers + ["ensemble_proba"]

    with open(output_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for name in sorted(records.keys()):
            data = records[name]
            probas = data["probas"]
            ensemble = float(np.mean(probas))
            row = (
                [name, str(data["label"])]
                + [f"{p:.6f}" for p in probas]
                + [f"{ensemble:.6f}"]
            )
            f.write("\t".join(row) + "\n")

    logger.info(f"Wrote {len(records)} rows to {output_tsv}")


def main():
    parser = argparse.ArgumentParser(
        description="Export per-protein predictions for all folds"
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--fasta_file", type=str, required=True)
    parser.add_argument("--aptamer_list_file", type=str, required=True)
    parser.add_argument("--output_predictions_tsv", type=str, required=True)
    parser.add_argument(
        "--checkpoint_mode",
        choices=["es", "es_ef10", "final"],
        default="es",
        help=(
            "es uses *_es.pt (default), es_ef10 uses *_es_ef10.pt, "
            "and final excludes both."
        ),
    )

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--augment_eps", type=float, default=0.95)
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("--contrast_weight", type=float, default=0.01)
    parser.add_argument("--contrast_temp", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="MTP")

    parser.add_argument(
        "--prottrans_output_path",
        type=str,
        default="Embeddings/ProtTrans/Models/ProtTrans/prot_t5_xl_uniref50/",
    )
    parser.add_argument(
        "--esm2_output_path",
        type=str,
        default="Embeddings/ESM2/esm2_t33_650M_UR50D/",
    )
    parser.add_argument("--pdb_dir", type=str, default="AlphafoldDB/")
    parser.add_argument("--proc_pdb_output_path", type=str, default="AFDB_coordinates/")
    parser.add_argument("--dssp_output_path", type=str, default="DSSP/AFDB/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--disable_prottrans", action="store_true")
    parser.add_argument("--disable_esm2", action="store_true")
    parser.add_argument("--disable_dssp", action="store_true")
    parser.add_argument("--disable_dihedrals", action="store_true")
    parser.add_argument("--disable_orientations", action="store_true")
    parser.add_argument("--disable_sidechains", action="store_true")
    parser.add_argument("--disable_rbf", action="store_true")
    parser.add_argument("--disable_pos_embeddings", action="store_true")
    parser.add_argument("--disable_edge_direction", action="store_true")
    parser.add_argument("--plddt_geometry_downweight", action="store_true")
    parser.add_argument(
        "--plddt_dir", type=str, default="AFDB_coordinates/per_residue_plddt_tsv"
    )
    parser.add_argument("--plddt_weight_mode", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--plddt_thresh", type=float, default=50.0)
    parser.add_argument("--plddt_soft_lo", type=float, default=40.0)
    parser.add_argument("--plddt_soft_hi", type=float, default=60.0)
    parser.add_argument(
        "--plddt_edge_mode", type=str, default="scale",
        choices=["scale", "knn_highconf", "none"],
    )
    parser.add_argument("--plddt_pool_weight", action="store_true")
    parser.add_argument("--plddt_knn_keep_min", type=float, default=0.5)

    args = parser.parse_args()

    if args.device == "" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    config = get_config()
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    config.dropout = args.dropout
    config.contrast = args.contrast
    config.contrast_weight = args.contrast_weight
    config.contrast_temp = args.contrast_temp
    config.pooling = args.pooling
    config.augment_eps = args.augment_eps
    config.plddt_pool_weight = args.plddt_pool_weight

    export_predictions(
        model_dir=args.model_dir,
        fasta_file=args.fasta_file,
        aptamer_list_file=args.aptamer_list_file,
        config=config,
        device=device,
        prottrans_output_path=args.prottrans_output_path,
        esm2_output_path=args.esm2_output_path,
        pdb_dir=args.pdb_dir,
        proc_pdb_output_path=args.proc_pdb_output_path,
        dssp_output_path=args.dssp_output_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        disable_prottrans=args.disable_prottrans,
        disable_esm2=args.disable_esm2,
        disable_dssp=args.disable_dssp,
        output_tsv=args.output_predictions_tsv,
        disable_dihedrals=args.disable_dihedrals,
        disable_orientations=args.disable_orientations,
        disable_sidechains=args.disable_sidechains,
        disable_rbf=args.disable_rbf,
        disable_pos_embeddings=args.disable_pos_embeddings,
        disable_edge_direction=args.disable_edge_direction,
        plddt_geometry_downweight=args.plddt_geometry_downweight,
        plddt_dir=args.plddt_dir,
        plddt_weight_mode=args.plddt_weight_mode,
        plddt_thresh=args.plddt_thresh,
        plddt_soft_lo=args.plddt_soft_lo,
        plddt_soft_hi=args.plddt_soft_hi,
        plddt_edge_mode=args.plddt_edge_mode,
        plddt_pool_weight=args.plddt_pool_weight,
        plddt_knn_keep_min=args.plddt_knn_keep_min,
        checkpoint_mode=args.checkpoint_mode,
    )


if __name__ == "__main__":
    main()
