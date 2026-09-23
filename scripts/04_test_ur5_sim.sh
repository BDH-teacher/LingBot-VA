#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
run "$CONDA_EXE" run --no-capture-output -n robotwin python "$PROJECT_ROOT/scripts/run_sim.py" --episodes 1 "$@"
