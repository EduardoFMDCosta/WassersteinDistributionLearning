#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"


for method in triangle_inequality_vertex joint_diagonal_milp; do
echo "------------------------------------------------- method = ${method} -----------------------------------------------------"

for num_samples_training in 5000 1000; do
echo "---------------- num training samples: ${num_samples_training} -------------------"

for rho in 1 2; do
echo "---------------- W${rho} -------------------"

echo "-- GaussianMixture --"
    echo "dim = 2, setting = 0"
    python -m experiments.generate_results --distribution GaussianMixture --num_dims 2 --setting 0 --wasserstein_order "$rho" --method "$method"  --num_samples_training "$num_samples_training"

    echo "dim = 3, setting = 0"
    python -m experiments.generate_results --distribution GaussianMixture --num_dims 3 --setting 0 --wasserstein_order "$rho" --method "$method"  --num_samples_training "$num_samples_training"

done        
done
done

echo "Finnished"

