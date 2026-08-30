#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

cd "$SOURCE"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_dinov2_final_conv.py' -v
