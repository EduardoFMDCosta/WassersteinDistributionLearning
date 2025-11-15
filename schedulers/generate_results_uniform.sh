#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"


for num_samples_training in 1000; do
echo "---------------- num training samples: ${num_samples_training} -------------------"

for rho in 1 2; do
echo "---------------- W${rho} -------------------"
for method in joint_optimization_milp joint_full_expansion_milp diagonal_constrained_tp triangle_inequality_vertex; do

echo "------- method = ${method} ---------"

echo "-- Uniform --"
    echo "dim = 2, setting = 0"
    python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 0 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training"

    echo "dim = 2, setting = 1"
    python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 1 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training"

    echo "dim = 2, setting = 2"
    python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 2 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training"

    echo "dim = 3, setting = 0"
    python -m experiments.generate_results --distribution Uniform --num_dims 3 --setting 0 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training"

    echo "dim = 100, setting = 0"
    python -m experiments.generate_results --distribution Uniform --num_dims 100 --setting 0 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training"

    echo "dim = 100, setting = 1"
    python -m experiments.generate_results --distribution Uniform --num_dims 100 --setting 1 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training"

    echo "dim = 100, setting = 2"
    python -m experiments.generate_results --distribution Uniform --num_dims 100 --setting 2 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training"


done        
done
done

echo "Finnished"


