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

for method in joint_diagonal_milp; do
echo "------------------------------------------------- method = ${method} -----------------------------------------------------"

for rho in 2; do
echo "---------------- W${rho} -------------------"

python -m experiments.generate_results_datasets --distribution OCTMNIST --num_dims 784 --setting 0 --wasserstein_order "$rho" --method "$method"  --random_seed "$seed"

done
done
done

echo "Finished"

