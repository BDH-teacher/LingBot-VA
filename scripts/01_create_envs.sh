#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_conda
for name in robotwin lingbot; do
  if ! "$CONDA_EXE" run -n "$name" python --version >/dev/null 2>&1; then
    run "$CONDA_EXE" create -y -n "$name" --override-channels -c conda-forge python=3.10.16 pip=25.0.1
  fi
  "$CONDA_EXE" env export -n "$name" --from-history | sed '/^prefix:/d' > "$PROJECT_ROOT/local/environment/$name.yml"
done
