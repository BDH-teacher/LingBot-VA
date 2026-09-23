#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo 'Install ROS 2 Jazzy first using README.md#real-robot. No robot connection attempted.' >&2
  exit 1
fi
run /usr/bin/python3 -m venv --system-site-packages "$PROJECT_ROOT/.venv-ros"
run "$PROJECT_ROOT/.venv-ros/bin/python" -m pip install -r "$PROJECT_ROOT/envs/ros-requirements.txt"
set +u
source /opt/ros/jazzy/setup.bash
set -u
run "$PROJECT_ROOT/.venv-ros/bin/python" -c 'import rclpy, control_msgs, ur_dashboard_msgs, ur_msgs, moveit_msgs; print("ROS IMPORT OK; no node initialized")'
