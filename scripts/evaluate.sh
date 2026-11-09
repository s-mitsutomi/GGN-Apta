#!/usr/bin/env bash
# Score a FASTA with the released five-fold ensemble.
#
# Required:
#   MODEL_DIR, TEST_FASTA_FILE, APTAMER_LIST_FILE, OUTPUT_FILE
#   PROTTRANS_OUTPUT_PATH, ESM2_OUTPUT_PATH, PDB_DIR, PROC_PDB_OUTPUT_PATH, DSSP_OUTPUT_PATH
# Optional: PROJECT_ROOT, DATA_ROOT, PYTHON_BIN, CHECKPOINT_MODE (default es)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${MODEL_DIR:?Set MODEL_DIR}"
: "${TEST_FASTA_FILE:?Set TEST_FASTA_FILE}"
: "${APTAMER_LIST_FILE:?Set APTAMER_LIST_FILE}"
: "${OUTPUT_FILE:?Set OUTPUT_FILE}"
: "${PROTTRANS_OUTPUT_PATH:?Set PROTTRANS_OUTPUT_PATH}"
: "${ESM2_OUTPUT_PATH:?Set ESM2_OUTPUT_PATH}"
: "${PDB_DIR:?Set PDB_DIR}"
: "${PROC_PDB_OUTPUT_PATH:?Set PROC_PDB_OUTPUT_PATH}"
: "${DSSP_OUTPUT_PATH:?Set DSSP_OUTPUT_PATH}"

require_dir "${MODEL_DIR}" "MODEL_DIR"
require_file "${TEST_FASTA_FILE}" "TEST_FASTA_FILE"
require_file "${APTAMER_LIST_FILE}" "APTAMER_LIST_FILE"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/Scripts/Sub/evaluate_test_set.py" \
  --model_dir "${MODEL_DIR}" \
  --test_fasta_file "${TEST_FASTA_FILE}" \
  --aptamer_list_file "${APTAMER_LIST_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --checkpoint_mode "${CHECKPOINT_MODE:-es}" \
  --hidden_dim 640 \
  --num_layers 4 \
  --dropout 0.60 \
  --augment_eps 0.60 \
  --contrast \
  --contrast_weight 1.0 \
  --contrast_temp 0.1 \
  --pooling GAP \
  --prottrans_output_path "${PROTTRANS_OUTPUT_PATH}" \
  --esm2_output_path "${ESM2_OUTPUT_PATH}" \
  --pdb_dir "${PDB_DIR}" \
  --proc_pdb_output_path "${PROC_PDB_OUTPUT_PATH}" \
  --dssp_output_path "${DSSP_OUTPUT_PATH}" \
  --disable_pos_embeddings \
  --batch_size "${BATCH_SIZE:-6}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --device "${DEVICE:-0}"
