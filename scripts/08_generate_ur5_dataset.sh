#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
episodes=1; task=move_can_pot; output="$PROJECT_ROOT/datasets/ur5_move_can_pot"; seed=0; extras=()
while (($#)); do
  case "$1" in
    --episodes) episodes="$2"; shift 2;;
    --task) task="$2"; shift 2;;
    --output-dir) output="$2"; shift 2;;
    --seed) seed="$2"; shift 2;;
    --randomization) extras+=(--randomization); shift;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
raw="${output}_raw"
run "$CONDA_EXE" run --no-capture-output -n robotwin python "$PROJECT_ROOT/scripts/run_sim.py" --headless --task "$task" --episodes "$episodes" --seed "$seed" --output-dir "$raw" "${extras[@]}"
run "$CONDA_EXE" run --no-capture-output -n lingbot python "$PROJECT_ROOT/scripts/convert_dataset.py" --raw "$raw" --output "$output"
run "$CONDA_EXE" run --no-capture-output -n lingbot python "$PROJECT_ROOT/scripts/validate_dataset.py" --dataset "$output"
