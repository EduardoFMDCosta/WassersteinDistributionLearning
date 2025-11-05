import torch
import time
import os
import csv
from argparse import Namespace
from typing import List, Dict, Tuple, Optional, Any, Generic, TypeVar, Union

from quantization import UncertainQuantization
from bound import DataDrivenRadius, fournier_radius as compute_fournier_radius, EmpiricalRadius
from solvers import get_solver

from configs.construct import get_support_assumption, get_distribution
from experiments.partitions import get_dict_of_partitions
from experiments.datastructures import TimeLogger, DataDrivenRadii, FournierRadii, EmpiricalRadii, Quantizations


def estimate_memory_usage(N, M, num_dims=2):
    """Estimate memory usage for given N, M parameters"""
    # Samples tensor: N x num_dims
    samples_memory = N * num_dims * 8 / (1024 * 1024)  # 8 bytes for float64
    
    # Distance matrix for K-means: N x M (potentially)
    distance_matrix_memory = N * M * 8 / (1024 * 1024)
    
    # Cluster assignments: N elements
    assignments_memory = N * 8 / (1024 * 1024)
    
    total_estimated_mb = samples_memory + distance_matrix_memory + assignments_memory
    return total_estimated_mb

def estimate_kmeans_complexity(N, k, num_dims=2, max_iterations=100):
    """Estimate computational complexity of K-means"""
    # K-means complexity is roughly O(n * k * d * iterations)
    # where n=samples, k=clusters, d=num_dimss, iterations=convergence steps
    operations = N * k * num_dims * max_iterations
    return operations

def assess_kmeans_feasibility(N, k, num_dims=2, max_memory_mb=2000, max_operations=1e9):
    """Pre-assess if K-means is likely to be feasible"""
    reasons = []
    
    # Memory check
    memory_mb = estimate_memory_usage(N, k, num_dims)
    if memory_mb > max_memory_mb:
        reasons.append(f"Memory: {memory_mb:.1f}MB > {max_memory_mb}MB limit")
    
    # Computational complexity check
    operations = estimate_kmeans_complexity(N, k, num_dims)
    if operations > max_operations:
        reasons.append(f"Complexity: {operations:.1e} > {max_operations:.1e} operations")

    
    estimated_time_seconds = operations / 1e8  # Rough estimate: 100M operations per second

    is_feasible = len(reasons) == 0
    return is_feasible, reasons, {
        'memory_mb': memory_mb,
        'operations': operations,
        'estimated_time_s': estimated_time_seconds
    }

def check_memory_feasibility(N, M, max_memory_mb=8000):  # 8GB default limit
    """Check if the operation is likely to fit in memory"""
    estimated_memory = estimate_memory_usage(N, M)
    if estimated_memory > max_memory_mb:
        print(f"Estimated memory usage: {estimated_memory:.1f} MB (exceeds limit of {max_memory_mb} MB)")
        return False
    print(f"Estimated memory usage: {estimated_memory:.1f} MB")
    return True


def run_combinations(
    args: Namespace, 
    M_options: List[int], 
    N_options: List[int], 
    max_memory_mb: int = 2000, 
    print_timings: bool = False,
    compute_empirical_radii: bool = False,
):
    solver = get_solver(method=args.method, compute_discrete_bound=args.compute_discrete_bound, compute_moment_bound=args.compute_moment_bound)

    distribution = get_distribution(**vars(args))

    partitions = get_dict_of_partitions(args, num_samples_options=N_options, num_clusters_options=M_options)
    
    quantization_times, radius_computation_times, computation_times = TimeLogger(), TimeLogger(), TimeLogger()
    quantizations, data_driven_radii, fournier_radii, empirical_radii = Quantizations(), DataDrivenRadii(), FournierRadii(), EmpiricalRadii()
    for N in N_options:
        samples_quantization = distribution.sample((N,))

        for M in M_options:
            is_feasible, reasons, stats = assess_kmeans_feasibility(
                N=N, k=M, num_dims=2, 
                max_memory_mb=max_memory_mb, 
                max_operations=1e9
            )
            if not is_feasible:
                print(f"Skipping M={M}, N={N}. Reasons: {'; '.join(reasons)}")
                print(f"  Stats: {stats['memory_mb']:.1f}MB, {stats['operations']:.1e} ops, ~{stats['estimated_time_s']:.1f}s")
                continue

            print(f"Proceeding with M={M}, N={N} (estimated: {stats['memory_mb']:.1f}MB, ~{stats['estimated_time_s']:.1f}s)")
            
            try:
                start = time.time()
                partition = partitions.at((N, M))
                quantizations.append((N, M),  UncertainQuantization(
                    partition=partition, 
                    samples=samples_quantization, 
                    beta=args.beta
                ))
                quantization_times.append((N, M), torch.as_tensor(time.time() - start))
                if print_timings:
                    print(f"K-means completed in {quantization_times.at((N, M)):.2f} seconds")

                start = time.time()
                data_driven_radii.append((N, M), DataDrivenRadius(
                    quantization=quantizations.at((N, M)),
                    solver=solver
                ))
                radius_computation_times.append((N, M), torch.as_tensor(time.time() - start))
                if print_timings:
                    print(f"Data-driven bounding completed in {radius_computation_times.at((N, M)):.2f} seconds")

                fournier_radii.append((N, M), compute_fournier_radius(support=partition.support, nsamples=2*N, beta=args.beta))

                computation_times.append((N, M), quantization_times.at((N, M)) + radius_computation_times.at((N, M)))

            except Exception as e:
                print(f"Unexpected error for M={M}, N={N}: {e}. Skipping this configuration.")
                continue

            if compute_empirical_radii:
                empirical_radii.append((N, M), EmpiricalRadius(quantization=quantizations.at((N, M)), dist=distribution))

    return (quantizations, data_driven_radii, fournier_radii, empirical_radii), (quantization_times, radius_computation_times, computation_times)

def generate_table(
    data_driven_radii: DataDrivenRadii,
    fournier_radii: FournierRadii,
    empirical_radii: EmpiricalRadii,
    args
): 
    # Prepare CSV file name
    csv_path = os.path.join(args.results_dir, f"ndims={args.num_dims}_set={args.setting}_radii.csv")

    # Prepare rows
    rows = []
    for (N, M) in data_driven_radii.keys():
        rows.append({
            "Distribution": args.distribution,
            "Dimension": args.num_dims,
            "Support radius": args.support_linf_radius,
            "N": N,
            "M": M,
            "Moment bound": f"{data_driven_radii.moment_bound_at((N, M))}",
            "Discrete bound": f"{data_driven_radii.discrete_bound_at((N, M))}",
            "Ours": f"{data_driven_radii.radius_at((N, M))}",
            "Fournier": f"{fournier_radii.radius_at((N, M))}",
            "Empirical (samples)": f"{empirical_radii.radius_samples_at((N, M))}",
            "Empirical (locs)": f"{empirical_radii.radius_quantization_at((N, M))}",
        })

    # Write to CSV
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Distribution",
                "Dimension",
                "Support radius",
                "N",
                "M",
                "Moment bound",
                "Discrete bound",
                "Ours",
                "Fournier",
                "Empirical (samples)",
                "Empirical (locs)",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)