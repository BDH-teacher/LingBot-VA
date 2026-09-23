#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require_conda
run "$CONDA_EXE" run --no-capture-output -n lingbot python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
run "$CONDA_EXE" run --no-capture-output -n lingbot python -m pip install -r "$PROJECT_ROOT/envs/lingbot-requirements.txt"
run "$CONDA_EXE" run --no-capture-output -n lingbot python -m pip install lerobot==0.3.3 --no-deps
run "$CONDA_EXE" install -y -n lingbot --override-channels -c conda-forge ffmpeg=7.1
# Official torch SDPA inference and flex training do not require flash-attn.
run "$CONDA_EXE" run --no-capture-output -n lingbot python "$PROJECT_ROOT/scripts/prepare_lingbot_runtime.py"
PYTHONPATH="$PROJECT_ROOT/outputs/runtime/lingbot-va:$PYTHONPATH" run "$CONDA_EXE" run --no-capture-output -n lingbot python -c 'import torch, diffusers, transformers, lerobot; from wan_va.modules.model import WanTransformer3DModel; print("LINGBOT IMPORT OK", torch.__version__, torch.cuda.is_available())'
in_env lingbot python -m pip freeze > "$PROJECT_ROOT/local/environment/lingbot_pip_freeze.txt"
"$CONDA_EXE" env export -n lingbot --from-history | sed '/^prefix:/d' > "$PROJECT_ROOT/local/environment/lingbot.yml"
