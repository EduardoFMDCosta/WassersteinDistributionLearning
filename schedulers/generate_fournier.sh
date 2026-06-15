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

for num_samples_training in 1000 5000; do
echo "---------------- num training samples: ${num_samples_training} -------------------"

for rho in 2; do
echo "---------------- W${rho} -------------------"

echo "-- TruncatedGaussian --"
    echo "dim = 1, setting = -1"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 2 --setting -1 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 1, setting = 1"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 2 --setting 1 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 1, setting = 2"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 2 --setting 2 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 1, setting = 3"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 2 --setting 3 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 1, setting = 4"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 2 --setting 4 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 1"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 10 --setting 1 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 2"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 10 --setting 2 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 3"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 10 --setting 3 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 4"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 10 --setting 4 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 5"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 10 --setting 5 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 6"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 10 --setting 6 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 10, setting = 0"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 10 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 25, setting = 0"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 25 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 50, setting = 0"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 50 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 75, setting = 0"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 75 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 100, setting = 0"
    python -m experiments.generate_fournier --distribution TruncatedGaussian --num_dims 100 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed" 

echo "-- TruncatedGaussianMixture --"
    echo "dim = 2, setting = 0"
    python -m experiments.generate_fournier --distribution TruncatedGaussianMixture --num_dims 2 --setting 0 --wasserstein_order "$rho"  --num_samples_training "$num_samples_training" --random_seed "$seed"

    echo "dim = 3, setting = 0"
    python -m experiments.generate_fournier --distribution TruncatedGaussianMixture --num_dims 3 --setting 0 --wasserstein_order "$rho"  --num_samples_training "$num_samples_training" --random_seed "$seed"

done        


for rho in 1; do
echo "---------------- W${rho} -------------------"

echo "-- Uniform --"
    echo "dim = 2, setting = 0"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 2 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 2, setting = 1"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 2 --setting 1 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 2, setting = 2"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 2 --setting 2 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 2, setting = 3"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 2 --setting 3 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"  

    echo "dim = 2, setting = 4"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 2 --setting 4 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"

    echo "dim = 10, setting = 0"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 10 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"

    echo "dim = 25, setting = 0"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 25 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"

    echo "dim = 50, setting = 0"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 50 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"

    echo "dim = 75, setting = 0"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 75 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"

    echo "dim = 100, setting = 0"
    python -m experiments.generate_fournier --distribution Uniform --num_dims 100 --setting 0 --wasserstein_order "$rho" --num_samples_training "$num_samples_training" --random_seed "$seed"

done
done
done

echo "Finished"


