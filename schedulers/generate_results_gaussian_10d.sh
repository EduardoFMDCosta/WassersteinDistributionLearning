#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"


for seed in 0 1 2 3 4 5 6 7 8 9; do
echo "-------------------------------------------- Random seed = ${seed} -----------------------------------------------"

for method in triangle_inequality_vertex joint_diagonal_milp; do
echo "------------------------------------------------- method = ${method} -----------------------------------------------------"

for num_samples_training in 5000; do
echo "---------------- num training samples: ${num_samples_training} -------------------"

for rho in 2; do
echo "---------------- W${rho} -------------------"

echo "-- Gaussian --"
    echo "dim = 10, setting = 1"
    python -m experiments.generate_results --distribution Gaussian --num_dims 10 --setting 1 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 2"
    python -m experiments.generate_results --distribution Gaussian --num_dims 10 --setting 2 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 3"
    python -m experiments.generate_results --distribution Gaussian --num_dims 10 --setting 3 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 4"
    python -m experiments.generate_results --distribution Gaussian --num_dims 10 --setting 4 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 5"
    python -m experiments.generate_results --distribution Gaussian --num_dims 10 --setting 5 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 6"
    python -m experiments.generate_results --distribution Gaussian --num_dims 10 --setting 6 --wasserstein_order "$rho" --method "$method" --num_samples_training "$num_samples_training" --random_seed "$seed"  

done        
done
done
done

echo "Finished"


