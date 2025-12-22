#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "START"


echo "------- num_dims = 10---------"

for seed in 0; do
echo "---------------- Random seed = ${seed} -------------------"

python -m experiments.partitions --distribution Gaussian --num_dims 10 --setting 0 --random_seed "$seed"

done

echo "FINISHED"