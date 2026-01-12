#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "START"

for seed in 0 1 2 3 4 5 6 7 8 9; do
echo "---------------- Random seed = ${seed} -------------------"

python -m experiments.partitions --distribution UCI-Turbine --num_dims 11 --setting 0 --random_seed "$seed"

done
echo "FINISHED"