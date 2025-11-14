import torch
import time
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
    time_limit: Optional[float] = None,
    print_timings: bool = False,
    compute_empirical_radii: bool = False,
    test_kmeans_feasibility: bool = False
):
    solver = get_solver(method=args.method)

    distribution = get_distribution(**vars(args))

    partitions = get_dict_of_partitions(args, num_samples_options=N_options, num_clusters_options=M_options)
    
    quantization_times, radius_computation_times, computation_times = TimeLogger(), TimeLogger(), TimeLogger()
    quantizations, data_driven_radii, fournier_radii, empirical_radii = Quantizations(), DataDrivenRadii(), FournierRadii(), EmpiricalRadii()
    for N in N_options:
        samples_quantization = distribution.sample((N,))

        for M in M_options:
            if test_kmeans_feasibility:
                is_feasible, reasons, stats = assess_kmeans_feasibility(
                    N=N, k=M, num_dims=2, 
                    max_memory_mb=max_memory_mb, 
                    max_operations=1e9
                )
                if not is_feasible:
                    print(f"Skipping M={M}, N={N}. Reasons: {'; '.join(reasons)}")
                    print(f"  Stats: {stats['memory_mb']:.1f}MB, {stats['operations']:.1e} ops, ~{stats['estimated_time_s']:.1f}s")
                    continue
                print(f"Proceeding M={M}, N={N} (estimated: {stats['memory_mb']:.1f}MB, ~{stats['estimated_time_s']:.1f}s)")
            else:
                print(f"Processing M={M}, N={N}")
            
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
                    solver=solver, 
                    wasserstein_order=args.wasserstein_order,
                    compute_discrete_bound=args.compute_discrete_bound,
                    compute_moment_bound=args.compute_moment_bound,
                    time_limit=time_limit
                ))
                radius_computation_times.append((N, M), torch.as_tensor(time.time() - start))
                if print_timings:
                    print(f"Data-driven bounding completed in {radius_computation_times.at((N, M)):.2f} seconds")

                fournier_radii.append((N, M), compute_fournier_radius(
                    support=partition.support, 
                    nsamples=2*N, 
                    wasserstein_order=args.wasserstein_order,
                    beta=args.beta,
                ))

                computation_times.append((N, M), quantization_times.at((N, M)) + radius_computation_times.at((N, M)))

            except Exception as e:
                print(f"Unexpected error for M={M}, N={N}: {e}. Skipping this configuration.")
                continue

            if compute_empirical_radii:
                empirical_radii.append((N, M), EmpiricalRadius(
                    samples=samples_quantization,
                    quantization=quantizations.at((N, M)), 
                    dist=distribution, 
                    wasserstein_order=args.wasserstein_order
                ))

    return (quantizations, data_driven_radii, fournier_radii, empirical_radii), (quantization_times, radius_computation_times, computation_times)

