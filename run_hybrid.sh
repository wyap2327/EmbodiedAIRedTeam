#!/bin/bash
# Run hybrid red agent strategy — 5 unsafe scenarios, max 15 turns
set -e
source ~/miniconda3/etc/profile.d/conda.sh
cd "$(dirname "$0")"

export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/1000
export PYTHONUNBUFFERED=1

conda run -n safeagentbench python -u red_blue_main.py \
  --scenes 5 \
  --max-turns 15 \
  --strategy hybrid \
  --category unsafe \
  --red-model dolphin-mistral \
  --model llama3.2 \
  2>&1 | tee outputs/hybrid_run_live.log
