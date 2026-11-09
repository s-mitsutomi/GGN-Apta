"""Pinned arguments copied from the released-model shell wrappers."""

from __future__ import annotations

TRAIN_SUFFIX = (
    "_baseline_fa3p7_fg1p5_pw0p5x_lr5e-5_wd1e-4_drop0p60_bs6_kpos1_norep_"
    "homology_id30_seed42_strat_no_repl_stop"
)

TRAIN_DESCRIPTION = """\
Train the released model (homology identity 30%, seed 42, positional embeddings off).

Pinned settings:
  focal alpha 3.7, focal gamma 1.5
  positive weight 0.5 times the training-set negative/positive ratio, unless --pos-weight is set
  learning rate 5e-5, weight decay 1e-4
  dropout 0.60, batch size 6, k_pos 1
  hidden size 640, layers 4, pooling GAP
  contrastive weight 1.0, temperature 0.1
  edge keep probability 0.60
  early stopping on validation AUPRC, patience 5, min delta 0.001
  exponential schedule, gamma 0.950
  5 folds, positional embeddings off
"""

EVALUATE_DESCRIPTION = """\
Score a FASTA with the released five-fold ensemble.

Pinned settings:
  hidden size 640, layers 4, dropout 0.60
  edge keep probability 0.60
  contrastive weight 1.0, temperature 0.1
  pooling GAP, positional embeddings off
"""

PREDICT_DESCRIPTION = """\
Write per-protein ensemble probabilities for a FASTA file.

Pinned settings:
  hidden size 640, layers 4, dropout 0.60
  edge keep probability 0.60
  contrastive weight 1.0, temperature 0.1
  pooling GAP, positional embeddings off
"""

CHECK_DESCRIPTION = (
    "Load one released-model checkpoint with the training module and require every key to match."
)


def train_flags(
    *,
    aptamer_list: str,
    train_fasta: str,
    output_dir: str,
    prottrans: str,
    esm2: str,
    pdb_dir: str,
    coordinates: str,
    dssp: str,
    pos_weight: str,
    num_workers: int,
    device: str,
) -> list[str]:
    return [
        "--aptamer_list_file",
        aptamer_list,
        "--train_fasta_file",
        train_fasta,
        "--output_dir",
        output_dir,
        "--prottrans_output_path",
        prottrans,
        "--esm2_output_path",
        esm2,
        "--pdb_dir",
        pdb_dir,
        "--proc_pdb_output_path",
        coordinates,
        "--dssp_output_path",
        dssp,
        "--learning_rate",
        "5e-5",
        "--weight_decay",
        "1e-4",
        "--batch_size",
        "6",
        "--seed_base",
        "1234",
        "--k_folds",
        "5",
        "--max_epochs",
        "100",
        "--focal_alpha",
        "3.7",
        "--focal_gamma",
        "1.5",
        "--pos_weight",
        pos_weight,
        "--dropout",
        "0.60",
        "--k_pos_override",
        "1",
        "--lr_schedule_type",
        "exponential",
        "--exp_gamma",
        "0.950",
        "--scheduler_patience",
        "5",
        "--gradient_clip_max_norm",
        "5.0",
        "--hidden_dim",
        "640",
        "--num_layers",
        "4",
        "--pooling",
        "GAP",
        "--contrast",
        "true",
        "--contrast_weight",
        "1.0",
        "--contrast_temp",
        "0.1",
        "--augment_eps",
        "0.60",
        "--use_stratified_sampling",
        "true",
        "--balanced_epoch_sampling",
        "false",
        "--stratified_without_replacement",
        "true",
        "--stop_epoch_when_exhausted",
        "true",
        "--use_mixed_precision",
        "true",
        "--early_stopping_metric",
        "aupr",
        "--early_stopping_min_delta",
        "0.001",
        "--early_stopping_patience",
        "5",
        "--swa_enabled",
        "false",
        "--eval_use_swa",
        "false",
        "--ema_enabled",
        "false",
        "--ema_decay",
        "0.999",
        "--eval_use_ema",
        "true",
        "--loss_type",
        "focal",
        "--loss_use_pos_weight_scale",
        "true",
        "--negative_rounds",
        "1",
        "--negative_rotation_seed",
        "20260718",
        "--num_workers",
        str(num_workers),
        "--device",
        device,
        "--gpu_ids",
        device,
        "--disable_prottrans",
        "false",
        "--disable_esm2",
        "false",
        "--disable_dssp",
        "false",
        "--disable_dihedrals",
        "false",
        "--disable_orientations",
        "false",
        "--disable_sidechains",
        "false",
        "--disable_rbf",
        "false",
        "--disable_pos_embeddings",
        "true",
        "--disable_edge_direction",
        "false",
        "--deterministic",
        "true",
        "--deterministic_warn_only",
        "true",
        "--suffix",
        TRAIN_SUFFIX,
    ]


def _score_flags(
    *,
    model_dir: str,
    fasta_flag: str,
    fasta: str,
    aptamer_list: str,
    output_flag: str,
    output: str,
    checkpoint_mode: str,
    prottrans: str,
    esm2: str,
    pdb_dir: str,
    coordinates: str,
    dssp: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> list[str]:
    return [
        "--model_dir",
        model_dir,
        fasta_flag,
        fasta,
        "--aptamer_list_file",
        aptamer_list,
        output_flag,
        output,
        "--checkpoint_mode",
        checkpoint_mode,
        "--hidden_dim",
        "640",
        "--num_layers",
        "4",
        "--dropout",
        "0.60",
        "--augment_eps",
        "0.60",
        "--contrast",
        "--contrast_weight",
        "1.0",
        "--contrast_temp",
        "0.1",
        "--pooling",
        "GAP",
        "--prottrans_output_path",
        prottrans,
        "--esm2_output_path",
        esm2,
        "--pdb_dir",
        pdb_dir,
        "--proc_pdb_output_path",
        coordinates,
        "--dssp_output_path",
        dssp,
        "--disable_pos_embeddings",
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--device",
        device,
    ]


def evaluate_flags(
    *,
    model_dir: str,
    test_fasta: str,
    aptamer_list: str,
    output: str,
    checkpoint_mode: str,
    prottrans: str,
    esm2: str,
    pdb_dir: str,
    coordinates: str,
    dssp: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> list[str]:
    return _score_flags(
        model_dir=model_dir,
        fasta_flag="--test_fasta_file",
        fasta=test_fasta,
        aptamer_list=aptamer_list,
        output_flag="--output_file",
        output=output,
        checkpoint_mode=checkpoint_mode,
        prottrans=prottrans,
        esm2=esm2,
        pdb_dir=pdb_dir,
        coordinates=coordinates,
        dssp=dssp,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def predict_flags(
    *,
    model_dir: str,
    fasta: str,
    aptamer_list: str,
    output: str,
    checkpoint_mode: str,
    prottrans: str,
    esm2: str,
    pdb_dir: str,
    coordinates: str,
    dssp: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> list[str]:
    return _score_flags(
        model_dir=model_dir,
        fasta_flag="--fasta_file",
        fasta=fasta,
        aptamer_list=aptamer_list,
        output_flag="--output_predictions_tsv",
        output=output,
        checkpoint_mode=checkpoint_mode,
        prottrans=prottrans,
        esm2=esm2,
        pdb_dir=pdb_dir,
        coordinates=coordinates,
        dssp=dssp,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
