#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
run "$CONDA_EXE" run --no-capture-output -n lingbot python "$PROJECT_ROOT/scripts/real_run.py" --dry-run "$@"
