#!/usr/bin/env bash
# Invoke with bash bootstrap.sh; never source this script.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=default
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
python3 scripts/bootstrap.py
