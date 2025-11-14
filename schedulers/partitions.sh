#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "START"

# echo "Uniform"
# python -m experiments.partitions --distribution Uniform --num_dims 2 --setting 0
# python -m experiments.partitions --distribution Uniform --num_dims 2 --setting 1
# python -m experiments.partitions --distribution Uniform --num_dims 2 --setting 2
# python -m experiments.partitions --distribution Uniform --num_dims 3 --setting 0

# python -m experiments.partitions --distribution Uniform --num_dims 100 --setting 0
# python -m experiments.partitions --distribution Uniform --num_dims 100 --setting 1
python -m experiments.partitions --distribution Uniform --num_dims 100 --setting 2

# echo "Gaussians"
# python -m experiments.partitions --distribution Gaussian --num_dims 2 --setting 0
# python -m experiments.partitions --distribution Gaussian --num_dims 3 --setting 0
# python -m experiments.partitions --distribution Gaussian --num_dims 10 --setting 0
# python -m experiments.partitions --distribution Gaussian --num_dims 100 --setting 0

# echo "GaussianMixture"
# python -m experiments.partitions --distribution GaussianMixture --num_dims 2 --setting 0
# python -m experiments.partitions --distribution GaussianMixture --num_dims 3 --setting 0

echo "FINNISHED"