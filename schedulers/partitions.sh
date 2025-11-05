#!/usr/bin/env bash

set -euo pipefail

# go to the repo root first
cd /path/to/your/repo

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"


echo "START"
python -m experiments.partition --distribution Gaussian --num_dims 10 --setting 0
python -m experiments.partition --distribution Gaussian --num_dims 100 --setting 0
echo "FINNISHED"