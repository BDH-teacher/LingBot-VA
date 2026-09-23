#!/usr/bin/env bash
set -Eeuo pipefail
export LINGBOT_UR5_ROOT="${LINGBOT_UR5_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
export PROJECT_ROOT="$LINGBOT_UR5_ROOT"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/local/environment"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_EXE" ]]; then
  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then CONDA_EXE="$candidate"; break; fi
  done
fi
run() { python3 "$PROJECT_ROOT/scripts/log_command.py" "$@"; }
in_env() { "$CONDA_EXE" run --no-capture-output -n "$1" "${@:2}"; }
require_conda() { [[ -x "$CONDA_EXE" ]] || { echo 'Conda missing: see README.md#installation' >&2; exit 1; }; }
