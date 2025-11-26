#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "START"

for seed in 1 2 3 4 5 6 7 8 9; do
echo "---------------- Random seed = ${seed} -------------------"

echo "Uniform"
python -m experiments.partitions --distribution Uniform --num_dims 2 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 2 --setting 1 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 2 --setting 2 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 2 --setting 3 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 2 --setting 4 --random_seed "$seed"

python -m experiments.partitions --distribution Uniform --num_dims 3 --setting 0 --random_seed "$seed"

python -m experiments.partitions --distribution Uniform --num_dims 10 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 10 --setting 1 --random_seed "$seed"

python -m experiments.partitions --distribution Uniform --num_dims 25 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 25 --setting 1 --random_seed "$seed"

python -m experiments.partitions --distribution Uniform --num_dims 50 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 50 --setting 1 --random_seed "$seed"

python -m experiments.partitions --distribution Uniform --num_dims 75 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 75 --setting 1 --random_seed "$seed"

python -m experiments.partitions --distribution Uniform --num_dims 100 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 100 --setting 1 --random_seed "$seed"
python -m experiments.partitions --distribution Uniform --num_dims 100 --setting 2 --random_seed "$seed"

echo "Gaussians"
python -m experiments.partitions --distribution Gaussian --num_dims 2 --setting -1 --random_seed "$seed"
python -m experiments.partitions --distribution Gaussian --num_dims 2 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution Gaussian --num_dims 2 --setting 1 --random_seed "$seed"
python -m experiments.partitions --distribution Gaussian --num_dims 2 --setting 2 --random_seed "$seed"
python -m experiments.partitions --distribution Gaussian --num_dims 2 --setting 3 --random_seed "$seed"
python -m experiments.partitions --distribution Gaussian --num_dims 2 --setting 4 --random_seed "$seed"
python -m experiments.partitions --distribution Gaussian --num_dims 2 --setting 5 --random_seed "$seed"

python -m experiments.partitions --distribution Gaussian --num_dims 3 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution Gaussian --num_dims 10 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution Gaussian --num_dims 100 --setting 0 --random_seed "$seed"

echo "GaussianMixture"
python -m experiments.partitions --distribution GaussianMixture --num_dims 2 --setting 0 --random_seed "$seed"
python -m experiments.partitions --distribution GaussianMixture --num_dims 3 --setting 0 --random_seed "$seed"

done
echo "FINNISHED"