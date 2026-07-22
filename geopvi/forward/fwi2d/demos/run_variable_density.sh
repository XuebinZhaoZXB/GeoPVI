#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
python3 "$(dirname "$0")/variable_density_demo.py" "$@"
