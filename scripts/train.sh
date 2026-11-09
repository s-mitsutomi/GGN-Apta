#!/usr/bin/env bash
# Train the released model (homology id30, seed 42, positional embeddings off).
#
# Required environment variables:
#   APTAMER_LIST_FILE, TRAIN_FASTA_FILE, OUTPUT_DIR
#   PROTTRANS_OUTPUT_PATH, ESM2_OUTPUT_PATH, PDB_DIR, PROC_PDB_OUTPUT_PATH, DSSP_OUTPUT_PATH
# Optional:
#   PROJECT_ROOT, DATA_ROOT, PYTHON_BIN, POS_WEIGHT, PW_MODE (default 0.5)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${APTAMER_LIST_FILE:?Set APTAMER_LIST_FILE}"
: "${TRAIN_FASTA_FILE:?Set TRAIN_FASTA_FILE}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${PROTTRANS_OUTPUT_PATH:?Set PROTTRANS_OUTPUT_PATH}"
: "${ESM2_OUTPUT_PATH:?Set ESM2_OUTPUT_PATH}"
: "${PDB_DIR:?Set PDB_DIR}"
: "${PROC_PDB_OUTPUT_PATH:?Set PROC_PDB_OUTPUT_PATH}"
: "${DSSP_OUTPUT_PATH:?Set DSSP_OUTPUT_PATH}"

require_file "${APTAMER_LIST_FILE}" "APTAMER_LIST_FILE"
require_file "${TRAIN_FASTA_FILE}" "TRAIN_FASTA_FILE"
require_dir "${PROTTRANS_OUTPUT_PATH}" "PROTTRANS_OUTPUT_PATH"
require_dir "${ESM2_OUTPUT_PATH}" "ESM2_OUTPUT_PATH"
require_dir "${PDB_DIR}" "PDB_DIR"
require_dir "${PROC_PDB_OUTPUT_PATH}" "PROC_PDB_OUTPUT_PATH"
require_dir "${DSSP_OUTPUT_PATH}" "DSSP_OUTPUT_PATH"
mkdir -p "${OUTPUT_DIR}"

PW_MODE="${PW_MODE:-0.5}"
if [[ -z "${POS_WEIGHT:-}" ]]; then
  POS_WEIGHT="$("${PYTHON_BIN}" "${SCRIPT_DIR}/pos_weight.py" \
    --train_fasta_file "${TRAIN_FASTA_FILE}" \
    --aptamer_list_file "${APTAMER_LIST_FILE}" \
    --pw_mode "${PW_MODE}")"
fi

TRAINER="${PROJECT_ROOT}/Scripts/Sub/train_fixed_lr_scheduler_no_ddp_fsdp_allow_sched_factor_1_augment_eps_edge_dropout.py"

exec "${PYTHON_BIN}" "${TRAINER}" \
  --aptamer_list_file "${APTAMER_LIST_FILE}" \
  --train_fasta_file "${TRAIN_FASTA_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --prottrans_output_path "${PROTTRANS_OUTPUT_PATH}" \
  --esm2_output_path "${ESM2_OUTPUT_PATH}" \
  --pdb_dir "${PDB_DIR}" \
  --proc_pdb_output_path "${PROC_PDB_OUTPUT_PATH}" \
  --dssp_output_path "${DSSP_OUTPUT_PATH}" \
  --learning_rate 5e-5 \
  --weight_decay 1e-4 \
  --batch_size 6 \
  --seed_base 1234 \
  --k_folds 5 \
  --max_epochs 100 \
  --focal_alpha 3.7 \
  --focal_gamma 1.5 \
  --pos_weight "${POS_WEIGHT}" \
  --dropout 0.60 \
  --k_pos_override 1 \
  --lr_schedule_type exponential \
  --exp_gamma 0.950 \
  --scheduler_patience 5 \
  --gradient_clip_max_norm 5.0 \
  --hidden_dim 640 \
  --num_layers 4 \
  --pooling GAP \
  --contrast true \
  --contrast_weight 1.0 \
  --contrast_temp 0.1 \
  --augment_eps 0.60 \
  --use_stratified_sampling true \
  --balanced_epoch_sampling false \
  --stratified_without_replacement true \
  --stop_epoch_when_exhausted true \
  --use_mixed_precision true \
  --early_stopping_metric aupr \
  --early_stopping_min_delta 0.001 \
  --early_stopping_patience 5 \
  --swa_enabled false \
  --eval_use_swa false \
  --ema_enabled false \
  --ema_decay 0.999 \
  --eval_use_ema true \
  --loss_type focal \
  --loss_use_pos_weight_scale true \
  --negative_rounds 1 \
  --negative_rotation_seed 20260718 \
  --num_workers "${NUM_WORKERS:-4}" \
  --device 0 \
  --gpu_ids 0 \
  --disable_prottrans false \
  --disable_esm2 false \
  --disable_dssp false \
  --disable_dihedrals false \
  --disable_orientations false \
  --disable_sidechains false \
  --disable_rbf false \
  --disable_pos_embeddings true \
  --disable_edge_direction false \
  --deterministic true \
  --deterministic_warn_only true \
  --suffix "_baseline_fa3p7_fg1p5_pw0p5x_lr5e-5_wd1e-4_drop0p60_bs6_kpos1_norep_homology_id30_seed42_strat_no_repl_stop"
