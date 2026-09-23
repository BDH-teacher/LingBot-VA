#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
run "$CONDA_EXE" run --no-capture-output -n robotwin python "$PROJECT_ROOT/scripts/audit_embodiment.py"
run "$CONDA_EXE" run --no-capture-output -n robotwin python "$PROJECT_ROOT/scripts/prepare_robotwin_runtime.py"
run "$CONDA_EXE" run --no-capture-output -n robotwin python "$PROJECT_ROOT/scripts/test_renderer.py"
