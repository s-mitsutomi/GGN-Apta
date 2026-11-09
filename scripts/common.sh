#!/usr/bin/env bash
# Shared paths for the train, evaluate, and predict wrappers. Source this file; do not execute it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

require_file() {
  local path="${1:?path required}"
  local label="${2:?label required}"
  if [[ -z "${path}" || ! -f "${path}" ]]; then
    echo "ERROR: ${label} is missing: ${path:-<unset>}" >&2
    exit 1
  fi
}

require_dir() {
  local path="${1:?path required}"
  local label="${2:?label required}"
  if [[ -z "${path}" || ! -d "${path}" ]]; then
    echo "ERROR: ${label} is missing: ${path:-<unset>}" >&2
    exit 1
  fi
}
