#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ " ${*} " != *" --execute-real "* ]]; then
  exec bash "$PROJECT_ROOT/scripts/13_real_dry_run.sh" "$@"
fi
# Deliberately not executed during repository setup. Requires system ROS Jazzy.
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then echo 'ROS 2 Jazzy missing: README.md#real-robot' >&2; exit 1; fi
set +u
source /opt/ros/jazzy/setup.bash
set -u
[[ -x "$PROJECT_ROOT/.venv-ros/bin/python" ]] || { echo 'Run scripts/prepare_ros_env.sh first'; exit 1; }
run "$PROJECT_ROOT/.venv-ros/bin/python" "$PROJECT_ROOT/scripts/real_run.py" "$@"
