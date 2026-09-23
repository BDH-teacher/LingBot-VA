#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
run "$CONDA_EXE" run --no-capture-output -n lingbot python -m lingbot_ur5.train "$@"
