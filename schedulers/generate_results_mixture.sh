#!/usr/bin/env bash
set -euo pipefail

# Initialize conda
source /home/sjladams/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate concentration_inequalities

# optional but explicit
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"


for rho in 1 2; do
echo "---------------- W${rho} -------------------"
for method in joint_full_expansion_milp; do

echo "------- method = ${method} ---------"

echo "-- GaussianMixture --"
    echo "dim = 2, setting = 0"
    python -m experiments.generate_results --distribution GaussianMixture --num_dims 2 --setting 0 --wasserstein_order "$rho" --method "$method"

    echo "dim = 3, setting = 0"
    python -m experiments.generate_results --distribution GaussianMixture --num_dims 3 --setting 0 --wasserstein_order "$rho" --method "$method"

done        
done

echo "Finnished"

