#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cd "$PROJECT_ROOT"
run "$CONDA_EXE" run --no-capture-output -n lingbot python "$PROJECT_ROOT/scripts/validate_dataset.py" "$@"
