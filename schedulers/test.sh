#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "dim = 1, setting = -1"
python -m experiments.generate_results --distribution Gaussian --num_dims 2 --setting -1 --wasserstein_order 1 --method joint_diagonal_milp --num_samples_training 5000 --random_seed 0


echo "Finished"


