#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
for name in robotwin lingbot; do
  "$CONDA_EXE" env export -n "$name" --from-history | sed '/^prefix:/d' > "$PROJECT_ROOT/local/environment/$name.yml"
  in_env "$name" python -m pip freeze > "$PROJECT_ROOT/local/environment/${name}_pip_freeze.txt"
done
run python3 "$PROJECT_ROOT/scripts/export_versions.py"
