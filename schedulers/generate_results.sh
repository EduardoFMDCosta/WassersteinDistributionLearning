#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

## Solvers to test:
# stochastic_vertice_ascent
# joint_optimization_milp
# diagonal_constrained_tp
# triangle_inequality_vertex
# no_triangle_inequality
# scalar_strategy

echo "--W2--"

echo "-- Uniform --"
echo "dim = 2, setting = 0"
python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 0 --wasserstein_order 2 --method stochastic_vertice_ascent
python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 0 --wasserstein_order 2 --method joint_optimization_milp
python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 0 --wasserstein_order 2 --method diagonal_constrained_tp 
python -m experiments.generate_results --distribution Uniform --num_dims 2 --setting 0 --wasserstein_order 2 --method triangle_inequality_vertex 
python -m experiments.generate_results  --distribution Uniform --num_dims 2 --setting 0 --wasserstein_order 2 --method scalar_strategy

echo "dim = 2, setting = 1"
echo "dim = 2, setting = 2"
echo "dim = 3, setting = 0"


echo "-- Gaussian --"
echo "dim = 2, setting = 0"
echo "dim = 3, setting = 0"
echo "dim = 10, setting = 0"
echo "dim = 100, setting = 0"


echo "-- GaussianMixture --"
echo "dim = 2, setting = 0"
echo "dim = 3, setting = 0"


