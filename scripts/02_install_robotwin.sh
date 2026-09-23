#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_conda
run "$CONDA_EXE" run --no-capture-output -n robotwin python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
run "$CONDA_EXE" run --no-capture-output -n robotwin python -m pip install -r "$PROJECT_ROOT/envs/robotwin-requirements.txt"
run "$CONDA_EXE" install -y -n robotwin --override-channels -c conda-forge -c nvidia/label/cuda-12.1.1 cuda-toolkit=12.1 gcc_linux-64=12 gxx_linux-64=12 cmake ninja ffmpeg git-lfs
run "$CONDA_EXE" run --no-capture-output -n robotwin python "$PROJECT_ROOT/scripts/install_robotwin_extras.py"
in_env robotwin python -m pip freeze > "$PROJECT_ROOT/local/environment/robotwin_pip_freeze.txt"
"$CONDA_EXE" env export -n robotwin --from-history | sed '/^prefix:/d' > "$PROJECT_ROOT/local/environment/robotwin.yml"
