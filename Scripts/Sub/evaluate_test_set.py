#!/usr/bin/env python3
"""
Standalone script to evaluate trained models on the test set.

Test evaluation uses an ensemble of K models (one per fold): for each sample,
predictions from all K fold models are averaged (mean of predicted probabilities),
then binary labels are obtained with threshold 0.5. Reported test metrics
(ensemble_metrics) are computed from this ensemble prediction.

Usage:
    python evaluate_test_set.py \
        --model_dir /path/to/checkpoints \
        --test_fasta_file /path/to/test.fasta.gz \
        --aptamer_list_file /path/to/positives.txt \
        --output_file /path/to/metrics.tsv
"""

import sys
import os
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
import json
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from Bio import SeqIO
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# Add parent directory to path to import training modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Scripts.Sub.ef10_checkpoint import CheckpointNotFound, resolve_fold_checkpoints
from Scripts.Sub.train_fixed import (
    extract_dataset,
    get_data_args,
    ProteinGraphDataset,
    labels_from_batch_names,
    _normalize_to_uniprot,
    _filter_multi_fragment_positives,
)
from Scripts.Sub.train_fixed_lr_scheduler_no_ddp_fsdp_allow_sched_factor_1_augment_eps_edge_dropout import (
    GGN_GO,
    build_model,
    get_config,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool = True, warn_only: bool = True) -> None:
    """Configure deterministic behavior across CUDA/cuDNN/PyTorch."""
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def make_worker_init_fn(base_seed: int):
    """Create deterministic worker init function for DataLoader workers."""
    def _seed_worker(worker_id):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _seed_worker


def load_positive_set(aptamer_list_file: str) -> Set[str]:
    """Load positive set from aptamer list file."""
    positive_set = set()
    with open(aptamer_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                positive_set.add(line)
    logger.info(f"Loaded {len(positive_set)} positive UniProt IDs")
    return positive_set


def load_model(model_path: str, device: torch.device, config) -> torch.nn.Module:
    """Load a saved model checkpoint."""
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Create model with same architecture as training
    # GGN_GO parameters: input_dim, hidden_dim, num_layers, augment_eps, dropout, out_dim, pooling, pertub
    output_dim = 1  # Binary classification: 1 logit per protein
    model = build_model(config, output_dim, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    positive_set: Set[str],
    device: torch.device,
    config,
) -> Dict[str, float]:
    """Evaluate a single model on test set."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_names = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            labels = labels_from_batch_names(batch.name, positive_set, device)
            
            # Get predictions
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
            
            # Model already outputs probabilities (readout layer includes Sigmoid)
            # No need to apply sigmoid again - that would cause double sigmoid!
            y_pred_proba = y_pred.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            
            all_predictions.extend(y_pred_proba)
            all_labels.extend(labels_np)
            all_names.extend(batch.name)
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred_proba = np.array(all_predictions)
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    if len(np.unique(y_true)) > 1:  # Both classes present
        metrics['aupr'] = average_precision_score(y_true, y_pred_proba)
        metrics['auroc'] = roc_auc_score(y_true, y_pred_proba)
    else:
        logger.warning("Only one class present in test set, skipping AUPR/AUROC")
        metrics['aupr'] = np.nan
        metrics['auroc'] = np.nan
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Additional statistics
    metrics['num_samples'] = len(y_true)
    metrics['num_positives'] = int(np.sum(y_true))
    metrics['num_negatives'] = int(len(y_true) - np.sum(y_true))
    metrics['positive_rate'] = float(np.mean(y_true))
    
    return metrics, all_names, y_true, y_pred_proba


def evaluate_all_folds(
    model_dir: str,
    test_fasta_file: str,
    aptamer_list_file: str,
    config,
    device: torch.device,
    prottrans_output_path: str,
    esm2_output_path: str,
    pdb_dir: str,
    proc_pdb_output_path: str,
    dssp_output_path: str,
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
    num_workers: int = 8,
    batch_size: int = 16,
    seed_base: int = 1234,
    checkpoint_mode: str = "es",
) -> Dict:
    """Evaluate all K-fold models on test set."""
    
    # Load positive set
    positive_set = load_positive_set(aptamer_list_file)
    
    # Prepare test data
    logger.info(f"Loading test data from {test_fasta_file}")
    test_fasta_data = extract_dataset(test_fasta_file)
    logger.info(f"Loaded {len(test_fasta_data)} test sequences before filtering")
    
    # Filter positive set for multi-fragment proteins AND remove those proteins
    # entirely from the evaluation dataset so they do not contribute as
    # negatives either.
    # extract_dataset returns a dict mapping IDs to sequences, so use keys()
    test_fasta_ids = list(test_fasta_data.keys()) if isinstance(test_fasta_data, dict) else [seq.id for seq in test_fasta_data]
    positive_set, multi_fragment_positives = _filter_multi_fragment_positives(
        positive_set, test_fasta_ids
    )
    if multi_fragment_positives:
        logger.info(f"Filtered out {len(multi_fragment_positives)} multi-fragment positives from label set")

        # Remove all sequences corresponding to multi-fragment positive UniProt IDs
        if isinstance(test_fasta_data, dict):
            before = len(test_fasta_data)
            test_fasta_data = {
                pid: seq
                for pid, seq in test_fasta_data.items()
                if _normalize_to_uniprot(pid) not in multi_fragment_positives
            }
            after = len(test_fasta_data)
        else:
            before = len(test_fasta_data)
            test_fasta_data = [
                seq
                for seq in test_fasta_data
                if _normalize_to_uniprot(seq.id) not in multi_fragment_positives
            ]
            after = len(test_fasta_data)
        logger.info(
            "Removed %d sequences belonging to multi-fragment positive proteins "
            "from test set (before=%d, after=%d)",
            before - after,
            before,
            after,
        )
    
    # Create test dataset and loader from the filtered test_fasta_data
    test_args = get_data_args(
        split="train",  # doesn't matter for test
        fasta_file=test_fasta_file,
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
    # ProteinGraphDataset expects a dict (from extract_dataset) or list of SeqRecords
    # extract_dataset returns a dict, so we can use it directly
    test_data = ProteinGraphDataset(
        test_fasta_data,
        range(len(test_fasta_data)),
        test_args,
        task_list="bp",
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=test_args.num_workers,
        prefetch_factor=1,
        worker_init_fn=make_worker_init_fn(seed_base + 100000),
    )
    
    # One checkpoint per fold. es is only *_es.pt (not *_es_ef10.pt).
    # es_ef10 is only *_es_ef10.pt. final excludes both. Legacy runs with no
    # *_es.pt still fall back to final checkpoints when mode is es.
    model_dir_path = Path(model_dir)
    try:
        model_files = resolve_fold_checkpoints(model_dir_path, checkpoint_mode)
    except CheckpointNotFound:
        if checkpoint_mode != "es":
            raise
        model_files = resolve_fold_checkpoints(model_dir_path, "final")
    num_folds_expected = config.k_folds
    if len(model_files) != num_folds_expected:
        raise ValueError(
            f"Expected exactly {num_folds_expected} fold models for ensemble, "
            f"found {len(model_files)} in {model_dir} (checkpoint_mode={checkpoint_mode})"
        )
    if checkpoint_mode == "es" and not all(p.stem.endswith("_es") for p in model_files):
        raise ValueError(
            "es mode selected a checkpoint that is not *_es.pt. "
            "Refusing a mixed ensemble."
        )
    if checkpoint_mode == "es_ef10" and not all(
        p.stem.endswith("_es_ef10") for p in model_files
    ):
        raise ValueError(
            "es_ef10 mode selected a checkpoint that is not *_es_ef10.pt. "
            "Refusing a mixed ensemble."
        )
    logger.info(f"Using ensemble of {len(model_files)} models (one per fold) for test evaluation")
    
    # Evaluate each fold
    fold_results = []
    all_fold_predictions = defaultdict(list)
    
    for model_file in model_files:
        fold_num = int(model_file.stem.split('fold')[-1].split('_')[0])
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Fold {fold_num}")
        logger.info(f"{'='*60}")
        
        # Load model
        model = load_model(str(model_file), device, config)
        
        # Evaluate
        metrics, names, y_true, y_pred_proba = evaluate_model(
            model, test_loader, positive_set, device, config
        )
        
        metrics['fold'] = fold_num
        fold_results.append(metrics)
        
        # Store predictions for ensemble
        for name, true_label, pred_proba in zip(names, y_true, y_pred_proba):
            all_fold_predictions[name].append((true_label, pred_proba))
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate ensemble predictions (average of 5 fold models per sample)
    logger.info(f"\n{'='*60}")
    logger.info("Calculating ensemble predictions (mean over 5 fold models)")
    logger.info(f"{'='*60}")
    
    ensemble_predictions = []
    ensemble_labels = []
    
    for name, fold_preds in all_fold_predictions.items():
        # All folds share the same label; ensemble = mean probability over folds
        true_label = fold_preds[0][0]
        avg_proba = np.mean([pred for _, pred in fold_preds])
        
        ensemble_predictions.append(avg_proba)
        ensemble_labels.append(true_label)
    
    ensemble_predictions = np.array(ensemble_predictions)
    ensemble_labels = np.array(ensemble_labels)
    ensemble_binary = (ensemble_predictions >= 0.5).astype(int)
    
    # Calculate ensemble metrics
    ensemble_metrics = {}
    if len(np.unique(ensemble_labels)) > 1:
        ensemble_metrics['aupr'] = average_precision_score(ensemble_labels, ensemble_predictions)
        ensemble_metrics['auroc'] = roc_auc_score(ensemble_labels, ensemble_predictions)
    else:
        ensemble_metrics['aupr'] = np.nan
        ensemble_metrics['auroc'] = np.nan
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        ensemble_labels, ensemble_binary, average='binary', zero_division=0
    )
    ensemble_metrics['precision'] = precision
    ensemble_metrics['recall'] = recall
    ensemble_metrics['f1'] = f1
    
    tn, fp, fn, tp = confusion_matrix(ensemble_labels, ensemble_binary).ravel()
    ensemble_metrics['true_positives'] = int(tp)
    ensemble_metrics['true_negatives'] = int(tn)
    ensemble_metrics['false_positives'] = int(fp)
    ensemble_metrics['false_negatives'] = int(fn)
    
    ensemble_metrics['num_samples'] = len(ensemble_labels)
    ensemble_metrics['num_positives'] = int(np.sum(ensemble_labels))
    ensemble_metrics['num_negatives'] = int(len(ensemble_labels) - np.sum(ensemble_labels))
    
    # Calculate statistics across folds
    fold_metrics_summary = {}
    metric_names = ['aupr', 'auroc', 'precision', 'recall', 'f1']
    for metric_name in metric_names:
        values = [r[metric_name] for r in fold_results if not np.isnan(r[metric_name])]
        if values:
            fold_metrics_summary[f'{metric_name}_mean'] = float(np.mean(values))
            fold_metrics_summary[f'{metric_name}_std'] = float(np.std(values))
            fold_metrics_summary[f'{metric_name}_min'] = float(np.min(values))
            fold_metrics_summary[f'{metric_name}_max'] = float(np.max(values))
    
    return {
        'fold_results': fold_results,
        'ensemble_metrics': ensemble_metrics,
        'fold_metrics_summary': fold_metrics_summary,
        'num_folds': len(fold_results),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on test set')
    
    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model checkpoint files')
    parser.add_argument('--test_fasta_file', type=str, required=True,
                        help='Path to test FASTA file')
    parser.add_argument('--aptamer_list_file', type=str, required=True,
                        help='Path to aptamer list file (positive set)')
    
    # Model configuration (must match training)
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds (ensemble size); must match training (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--augment_eps', type=float, default=0.95,
                        help='Augment epsilon / edge dropout keep probability (default: 0.95)')
    parser.add_argument('--contrast', action='store_true',
                        help='Use contrastive learning')
    parser.add_argument('--contrast_weight', type=float, default=0.01,
                        help='Contrastive weight (default: 0.01)')
    parser.add_argument('--contrast_temp', type=float, default=0.1,
                        help='Contrastive temperature (default: 0.1)')
    parser.add_argument('--pooling', type=str, default='MTP',
                        choices=['MTP', 'GMP', 'GAP'],
                        help='Pooling method: MTP, GMP, or GAP (default: MTP)')
    
    # Data paths
    parser.add_argument('--prottrans_output_path', type=str,
                        default='Embeddings/ProtTrans/Models/ProtTrans/prot_t5_xl_uniref50/',
                        help='ProtTrans embeddings path')
    parser.add_argument('--esm2_output_path', type=str,
                        default='Embeddings/ESM2/esm2_t33_650M_UR50D/',
                        help='ESM2 embeddings path')
    parser.add_argument('--pdb_dir', type=str, default='AlphafoldDB/',
                        help='PDB directory')
    parser.add_argument('--proc_pdb_output_path', type=str, default='AFDB_coordinates/',
                        help='Processed PDB coordinates path')
    parser.add_argument('--dssp_output_path', type=str, default='DSSP/AFDB/',
                        help='DSSP output path')
    parser.add_argument('--disable_prottrans', action='store_true',
                        help='Use zero ProtTrans features instead of loading embeddings')
    parser.add_argument('--disable_esm2', action='store_true',
                        help='Use zero ESM2 features instead of loading embeddings')
    parser.add_argument('--disable_dssp', action='store_true',
                        help='Use zero DSSP features instead of loading DSSP arrays')
    parser.add_argument('--disable_dihedrals', action='store_true',
                        help='Zero backbone dihedral node_s features (6 dims)')
    parser.add_argument('--disable_orientations', action='store_true',
                        help='Zero backbone orientation node_v vectors (2 channels)')
    parser.add_argument('--disable_sidechains', action='store_true',
                        help='Zero sidechain proxy node_v vector (1 channel)')
    parser.add_argument('--disable_rbf', action='store_true',
                        help='Zero RBF distance edge_s features (16 dims)')
    parser.add_argument('--disable_pos_embeddings', action='store_true',
                        help='Zero sinusoidal positional edge_s embeddings (16 dims)')
    parser.add_argument('--disable_edge_direction', action='store_true',
                        help='Zero unit Cα–Cα edge_v direction vectors')
    parser.add_argument('--plddt_geometry_downweight', action='store_true',
                        help='Downweight low-pLDDT geometry (keep sequence/LM nodes)')
    parser.add_argument('--plddt_dir', type=str, default='AFDB_coordinates/per_residue_plddt_tsv',
                        help='Per-residue pLDDT TSV directory')
    parser.add_argument('--plddt_weight_mode', type=str, default='hard', choices=['hard', 'soft'])
    parser.add_argument('--plddt_thresh', type=float, default=50.0)
    parser.add_argument('--plddt_soft_lo', type=float, default=40.0)
    parser.add_argument('--plddt_soft_hi', type=float, default=60.0)
    parser.add_argument('--plddt_edge_mode', type=str, default='scale',
                        choices=['scale', 'knn_highconf', 'none'])
    parser.add_argument('--plddt_pool_weight', action='store_true',
                        help='Weight GAP pooling by pLDDT')
    parser.add_argument('--plddt_knn_keep_min', type=float, default=0.5)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers (default: 8)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID (default: 0)')
    parser.add_argument('--seed_base', type=int, default=1234,
                        help='Base seed for reproducibility (default: 1234)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Enable deterministic CUDA/cuDNN/PyTorch behavior (default: True)')
    parser.add_argument('--non_deterministic', dest='deterministic', action='store_false',
                        help='Disable deterministic behavior')
    parser.add_argument('--deterministic_warn_only', action='store_true', default=True,
                        help='Warn (instead of error) on nondeterministic ops (default: True)')
    parser.add_argument('--deterministic_fail_fast', dest='deterministic_warn_only', action='store_false',
                        help='Raise errors on nondeterministic ops')
    
    # Output
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for results (JSON format)')
    parser.add_argument('--output_tsv', type=str, default=None,
                        help='Output file for results (TSV format)')
    parser.add_argument(
        '--checkpoint_mode',
        type=str,
        default='es',
        choices=['es', 'es_ef10', 'final'],
        help='es uses *_es.pt (default), es_ef10 uses *_es_ef10.pt, final excludes both',
    )
    
    args = parser.parse_args()

    configure_determinism(
        enabled=args.deterministic,
        warn_only=args.deterministic_warn_only,
    )
    set_all_seeds(args.seed_base)
    logger.info(
        "Determinism settings: enabled=%s, warn_only=%s, seed_base=%d, CUBLAS_WORKSPACE_CONFIG=%s",
        args.deterministic,
        args.deterministic_warn_only,
        args.seed_base,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
    )
    
    # Set up device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
        logger.info(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU")
    
    # Create config
    config = get_config()
    config.k_folds = args.k_folds
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    config.dropout = args.dropout
    config.contrast = args.contrast
    config.contrast_weight = args.contrast_weight
    config.contrast_temp = args.contrast_temp
    config.pooling = args.pooling
    config.batch_size = args.batch_size
    config.augment_eps = args.augment_eps
    config.plddt_pool_weight = args.plddt_pool_weight
    config.device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "" else "cpu"
    
    # Evaluate
    results = evaluate_all_folds(
        model_dir=args.model_dir,
        test_fasta_file=args.test_fasta_file,
        aptamer_list_file=args.aptamer_list_file,
        config=config,
        device=device,
        prottrans_output_path=args.prottrans_output_path,
        esm2_output_path=args.esm2_output_path,
        pdb_dir=args.pdb_dir,
        proc_pdb_output_path=args.proc_pdb_output_path,
        dssp_output_path=args.dssp_output_path,
        disable_prottrans=args.disable_prottrans,
        disable_esm2=args.disable_esm2,
        disable_dssp=args.disable_dssp,
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
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed_base=args.seed_base,
        checkpoint_mode=args.checkpoint_mode,
    )
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    logger.info("\nPer-Fold Results:")
    for fold_result in results['fold_results']:
        logger.info(f"\nFold {fold_result['fold']}:")
        logger.info(f"  AUPR: {fold_result['aupr']:.4f}")
        logger.info(f"  AUROC: {fold_result['auroc']:.4f}")
        logger.info(f"  Precision: {fold_result['precision']:.4f}")
        logger.info(f"  Recall: {fold_result['recall']:.4f}")
        logger.info(f"  F1: {fold_result['f1']:.4f}")
    
    logger.info("\nEnsemble Results (Average across folds):")
    em = results['ensemble_metrics']
    logger.info(f"  AUPR: {em['aupr']:.4f}")
    logger.info(f"  AUROC: {em['auroc']:.4f}")
    logger.info(f"  Precision: {em['precision']:.4f}")
    logger.info(f"  Recall: {em['recall']:.4f}")
    logger.info(f"  F1: {em['f1']:.4f}")
    logger.info(f"  TP: {em['true_positives']}, TN: {em['true_negatives']}")
    logger.info(f"  FP: {em['false_positives']}, FN: {em['false_negatives']}")
    
    logger.info("\nFold Statistics:")
    for key, value in results['fold_metrics_summary'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {args.output_file}")
    
    if args.output_tsv:
        import csv
        with open(args.output_tsv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['fold', 'aupr', 'auroc', 'precision', 'recall', 'f1'])
            writer.writeheader()
            for fold_result in results['fold_results']:
                writer.writerow({
                    'fold': fold_result['fold'],
                    'aupr': fold_result['aupr'],
                    'auroc': fold_result['auroc'],
                    'precision': fold_result['precision'],
                    'recall': fold_result['recall'],
                    'f1': fold_result['f1'],
                })
            # Add ensemble row
            writer.writerow({
                'fold': 'ensemble',
                'aupr': em['aupr'],
                'auroc': em['auroc'],
                'precision': em['precision'],
                'recall': em['recall'],
                'f1': em['f1'],
            })
        logger.info(f"TSV results saved to {args.output_tsv}")


if __name__ == '__main__':
    main()
