import torch
from sets import ConvexHullPartition, VoronoiPartition
from quantization import Quantization
from plotting.plot import plot_kmeans_partition
from configs.handlers import parse_arguments
from bound import data_driven_radius, fournier_radius
from configs.construct import get_support_assumption, get_distribution

from configs.handlers import parse_arguments

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        dimension=2,
        setting=0,
        num_samples=1000,
        num_clusters=10,
        beta=1e-4,
        plot=True
    )

    # Set parameters
    M = args.num_clusters
    N = args.num_samples
    beta = args.beta
    method = 'dual_sinkhorn'
    support_assumption = get_support_assumption(**vars(args))

    # (Unknown) Generating probability
    distribution = get_distribution(**vars(args))

    # Generate Partitions
    samples = distribution.sample((N,))
    voronoi_partition = VoronoiPartition(support=support_assumption, samples=samples, k=M)
    convex_hull_partition = ConvexHullPartition(support=support_assumption, samples=samples, k=M)

    # Generate Quantization
    samples = distribution.sample((N,))
    voronoi_quantization = Quantization(partition=voronoi_partition, samples=samples)
    convex_hull_quantization = Quantization(partition=convex_hull_partition, samples=samples)

    # Plot samples and clusterized distribution
    if args.plot:
        plot_kmeans_partition(quantization=voronoi_quantization, title=f"Voronoi Partition (M={M}, N={N})")
        plot_kmeans_partition(quantization=convex_hull_quantization, title=f"Convex Hull Partition (M={M}, N={N})")

    # Compute bounds
    fournier_bound = fournier_radius(support=voronoi_partition.support, nsamples=N, beta=beta)

    data_driven_output_voronoi = data_driven_radius(quantization=voronoi_quantization, beta=beta, method=method)
    data_driven_output_convex_hull = data_driven_radius(quantization=convex_hull_quantization, beta=beta, method=method)

    print(f"Number of clusters (M) / num_samples (N): {M} / {N} \n"
          f"\t Fournier: {fournier_bound:.4f} \n"
          f"\t Ours (using convex hull partition): {data_driven_output_convex_hull.radius:.4f} \n"
          f"\t Ours (using voronoi partition): {data_driven_output_voronoi.radius:.4f} \n"
        )

