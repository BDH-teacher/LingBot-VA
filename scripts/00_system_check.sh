#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
{
  date --iso-8601=seconds
  cat /etc/os-release
  uname -a
  lscpu | head -25
  nvidia-smi || true
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv || true
  free -h
  df -h "$PROJECT_ROOT"
  if command -v vulkaninfo >/dev/null; then vulkaninfo --summary; else
    echo 'vulkaninfo unavailable; loader/ICD inventory only (NOT renderer validation)'
    ldconfig -p | grep vulkan || true
    ls /usr/share/vulkan/icd.d || true
  fi
  echo 'nvidia-smi CUDA = driver compatibility ceiling; nvcc = installed toolkit'
  if command -v nvcc >/dev/null; then nvcc --version; else echo 'nvcc not on PATH'; fi
  if [[ -n "$CONDA_EXE" ]]; then "$CONDA_EXE" --version; "$CONDA_EXE" info --envs; fi
  python3 --version
  git --version
  git lfs version || true
} 2>&1 | tee "$PROJECT_ROOT/logs/system_check.txt"
