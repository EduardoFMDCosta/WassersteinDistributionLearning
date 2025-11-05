#!/usr/bin/env bash
set -euo pipefail

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "START"
python -u experiments/partition --distribution Gaussian --num_dims 10 --setting 0
echo "FINNISHED"