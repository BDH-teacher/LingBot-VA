#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
export CONDA_EXE
run python3 "$PROJECT_ROOT/scripts/run_evaluation.py" --official --checkpoint "$PROJECT_ROOT/checkpoints/pretrained/lingbot-va-posttrain-robotwin" --episodes 1 "$@"
