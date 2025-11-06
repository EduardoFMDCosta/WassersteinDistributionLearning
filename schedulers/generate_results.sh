#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"



for rho in 1; do
echo "---------------- W${rho} -------------------"
for method in stochastic_vertice_ascent \
    joint_optimization_milp \
    diagonal_constrained_tp \
    triangle_inequality_vertex \
    no_triangle_inequality \
    scalar_strategy; do

echo "------- method = ${method} ---------"

echo "-- Uniform --"
    echo "dim = 2, setting = 0"
    python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 0 --wasserstein_order "$rho" --method "$method"

#     echo "dim = 2, setting = 1"
#     python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 1 --wasserstein_order "$rho" --method "$method"

#     echo "dim = 2, setting = 2"
#     python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 2 --wasserstein_order "$rho" --method "$method"

#     echo "dim = 3, setting = 0"
#     python -m experiments.generate_results --distribution Uniform --num_dims 3 --setting 0 --wasserstein_order "$rho" --method "$method"


# echo "-- Gaussian --"
#     echo "dim = 2, setting = 0"
#     python -m experiments.generate_results --distribution Gaussian --num_dims 2 --setting 0 --wasserstein_order "$rho" --method "$method"

#     echo "dim = 3, setting = 0"
#     python -m experiments.generate_results --distribution Gaussian --num_dims 3 --setting 0 --wasserstein_order "$rho" --method "$method"

#     echo "dim = 10, setting = 0"
#     python -m experiments.generate_results --distribution Gaussian --num_dims 10 --setting 0 --wasserstein_order "$rho" --method "$method"

#     echo "dim = 100, setting = 0"
#     python -m experiments.generate_results --distribution Gaussian --num_dims 100 --setting 0 --wasserstein_order "$rho" --method "$method"


# echo "-- GaussianMixture --"
#     echo "dim = 2, setting = 0"
#     python -m experiments.generate_results --distribution GaussianMixture --num_dims 2 --setting 0 --wasserstein_order "$rho" --method "$method"

#     echo "dim = 3, setting = 0"
#     python -m experiments.generate_results --distribution GaussianMixture --num_dims 3 --setting 0 --wasserstein_order "$rho" --method "$method"

done        
done


