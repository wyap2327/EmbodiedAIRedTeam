#!/bin/bash
# Run any command inside the safeagentbench conda env from WSL.
# Usage: bash wsl_run.sh <python-script> [args...]
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safeagentbench
cd "$(dirname "$0")"
exec python "$@"
