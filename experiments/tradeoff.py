import torch
import time
import os

from sets import BoundedVoronoiPartition
from quantization import Quantization, UncertainQuantization
from bound import DataDrivenRadius, fournier_radius
from plotting.plot import colored_scatter

from configs.handlers import parse_arguments
from configs.construct import get_support_assumption, get_distribution


def estimate_memory_usage(N, M, dimension=2):
    """Estimate memory usage for given N, M parameters"""
    # Samples tensor: N x dimension
    samples_memory = N * dimension * 8 / (1024 * 1024)  # 8 bytes for float64
    
    # Distance matrix for K-means: N x M (potentially)
    distance_matrix_memory = N * M * 8 / (1024 * 1024)
    
    # Cluster assignments: N elements
    assignments_memory = N * 8 / (1024 * 1024)
    
    total_estimated_mb = samples_memory + distance_matrix_memory + assignments_memory
    return total_estimated_mb

def estimate_kmeans_complexity(N, k, dimension=2, max_iterations=100):
    """Estimate computational complexity of K-means"""
    # K-means complexity is roughly O(n * k * d * iterations)
    # where n=samples, k=clusters, d=dimensions, iterations=convergence steps
    operations = N * k * dimension * max_iterations
    return operations

def assess_kmeans_feasibility(N, k, dimension=2, max_memory_mb=2000, max_operations=1e9):
    """Pre-assess if K-means is likely to be feasible"""
    reasons = []
    
    # Memory check
    memory_mb = estimate_memory_usage(N, k, dimension)
    if memory_mb > max_memory_mb:
        reasons.append(f"Memory: {memory_mb:.1f}MB > {max_memory_mb}MB limit")
    
    # Computational complexity check
    operations = estimate_kmeans_complexity(N, k, dimension)
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
        dimension=2,
        setting=0,
        num_samples=None,
        num_clusters=None,
        beta=1e-4,
        plot=False
    )

    beta = args.beta
    method = 'stackelberg_equilibrium'
    support_assumption = get_support_assumption(**vars(args))
    distribution = get_distribution(**vars(args))

    N_options = [100, 1000, 5000, 10000, 50000, 100000]
    M_options = [10, 50, 100]
    MAX_MEMORY_MB = 2000  # 2GB memory limit for fast testing

    Ns, Ms, kmean_times, data_driven_times, data_driven_bounds, fournier_bounds = [], [], [], [], [], []
    for N in N_options:
        samples_partition = distribution.sample((N,))
        samples_quantization = distribution.sample((N,))

        for M in M_options:
            # Pre-assess K-means feasibility
            is_feasible, reasons, stats = assess_kmeans_feasibility(
                N=N, k=M, dimension=2, 
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
                    use_voronoi_radii=False # set to false to speed up
                )
                quantization = Quantization(partition=partition, samples=samples_quantization)
                quantization = UncertainQuantization(quantization=quantization, beta=beta)
                kmeans_time = time.time() - start
                print(f"K-means completed in {kmeans_time:.2f} seconds")

                print(f"### Bounding for: clusters (M) / num_samples (N): {M} / {N}--- ###")    
                start = time.time()
                data_driven_output = DataDrivenRadius(quantization=quantization, method=method)
                bounding_time = time.time() - start
                print(f"Data-driven bounding completed in {bounding_time:.2f} seconds")

                fournier_result = fournier_radius(support=partition.support, nsamples=N, beta=beta)
                print(f"Fournier bound completed")

                kmean_times.append(kmeans_time)
                fournier_bounds.append(fournier_result)
                data_driven_bounds.append(data_driven_output.radius)
                data_driven_times.append(bounding_time)
                Ns.append(N)
                Ms.append(M)
                print(f"Successfully completed M={M}, N={N}")
                
            except Exception as e:
                print(f"Unexpected error for M={M}, N={N}: {e}. Skipping this configuration.")
                continue

    computation_times = torch.as_tensor(kmean_times) + torch.as_tensor(data_driven_times)
    for title, data in zip(
        ['Data-Driven Bound', 'Computation Time Kmeans', 'Computation Time Bounds', 'Computation Time', 'Fournier Bound'], 
        [data_driven_bounds, kmean_times, data_driven_times, computation_times, fournier_bounds]
    ):
        colored_scatter(
            x=torch.as_tensor(Ns), 
            y=torch.as_tensor(Ms), 
            c=torch.as_tensor(data).real, 
            title=title, 
            s=200, 
            file_name=f'figures{os.sep}{title.lower().replace(" ", "_")}.png'
        )
