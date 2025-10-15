import torch
import time
import matplotlib.pyplot as plt

from sets import BoundedVoronoiPartition
from quantization import UncertainQuantization
from bound import DataDrivenRadius, fournier_radius

import plotting.plot as plot
from configs.handlers import parse_arguments
from configs.construct import get_support_assumption, get_distribution
from experiments.utils import TimeLogger, DataDrivenRadii, FournierRadii, Quantizations


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


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="GaussianMixture",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_clusters=1000,
        beta=1e-4,
        method='stochastic_vertice_ascent',
        plot=False
    )

    support_assumption = get_support_assumption(**vars(args))
    distribution = get_distribution(**vars(args))

    N_options = [100, 1000] # [100, 1000, 5000, 10000, 50000, 100000]
    M_options = [10, 50] # [10, 50, 100]
    MAX_MEMORY_MB = 2000  # 2GB memory limit for fast testing

    quantization_times, radius_computation_times, computation_times = TimeLogger(), TimeLogger(), TimeLogger()
    quantizations, data_driven_radii, fournier_radii = Quantizations(), DataDrivenRadii(), FournierRadii()
    for N in N_options:
        samples_partition = distribution.sample((N,))
        samples_quantization = distribution.sample((N,))

        for M in M_options:
            is_feasible, reasons, stats = assess_kmeans_feasibility(
                N=N, k=M, num_dims=2, 
                max_memory_mb=MAX_MEMORY_MB, 
                max_operations=1e9
            )
            
            if not is_feasible:
                print(f"Skipping M={M}, N={N}. Reasons: {'; '.join(reasons)}")
                print(f"  Stats: {stats['memory_mb']:.1f}MB, {stats['operations']:.1e} ops, ~{stats['estimated_time_s']:.1f}s")
                continue
            
            print(f"Proceeding with M={M}, N={N} (estimated: {stats['memory_mb']:.1f}MB, ~{stats['estimated_time_s']:.1f}s)")
                
            try:
                print(f"### Kmeans for: clusters (M) / num_samples (N): {M} / {N}--- ###")    
                start = time.time()
                partition = BoundedVoronoiPartition(
                    support=support_assumption, 
                    samples=samples_partition, 
                    M=M,
                )
                quantizations.append((N, M), UncertainQuantization(partition=partition, samples=samples_quantization, beta=args.beta))
                quantization_times.append((N, M), torch.as_tensor(time.time() - start))
                print(f"K-means completed in {quantization_times.at((N, M)):.2f} seconds")

                print(f"### Bounding for: clusters (M) / num_samples (N): {M} / {N}--- ###")    
                start = time.time()
                data_driven_radii.append((N, M), DataDrivenRadius(quantization=quantizations.at((N, M)), method=args.method))
                radius_computation_times.append((N, M), torch.as_tensor(time.time() - start))
                print(f"Data-driven bounding completed in {radius_computation_times.at((N, M)):.2f} seconds")

                fournier_radii.append((N, M), fournier_radius(support=partition.support, nsamples=N, beta=args.beta))
                print(f"Fournier bound completed")
                print(f"Successfully completed M={M}, N={N}")

                computation_times.append((N, M), quantization_times.at((N, M)) + radius_computation_times.at((N, M)))
                
            except Exception as e:
                print(f"Unexpected error for M={M}, N={N}: {e}. Skipping this configuration.")
                continue


    # Plot Bounds
    xlabel, ylabel = "Number of samples (N)", "Number of clusters (M)"

    fig, ax = plt.subplots(3, 1, figsize=(6, 12), constrained_layout=True)
    ax[0] = plot.plot_data_driven_radii(ax[0], data_driven_radii, field='moment_bound', ylabel=ylabel, title="Bound on Moment-Term (e1)")
    ax[1] = plot.plot_data_driven_radii(ax[1], data_driven_radii, field='discrete_bound', ylabel=ylabel, title="Bound on Discrete-Term Bound (e2)")
    ax[2] = plot.plot_data_driven_radii(ax[2], data_driven_radii, field='radius', xlabel=xlabel, ylabel=ylabel, title="Data-Driven Bound")
    plt.show()

    # Plot Computation Times
    fig, ax = plt.subplots(3, 1, figsize=(6, 12), constrained_layout=True)
    ax[0] = plot.plot_time_logger(ax[0], quantization_times, ylabel=ylabel, title="Quantization time")
    ax[1] = plot.plot_time_logger(ax[1], radius_computation_times, ylabel=ylabel, title="Radius computation time")
    ax[2] = plot.plot_time_logger(ax[2], computation_times, xlabel=xlabel, ylabel=ylabel, title="Total computation time")
    plt.show()