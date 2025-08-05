import torch
from sets import KMeansPartition
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
        num_clusters=1000,
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

    # Generate samples
    samples = distribution.sample((N,))

    # Clusterize samples (obtaining \hat{P}_M)
    partition = KMeansPartition(support=support_assumption, samples=samples, k=M, prefilter=args.distribution == "Discrete")

    # Plot samples and clusterized distribution
    if args.plot:
        plot_kmeans_partition(partition=partition)

    # Compute bounds
    data_driven_bound = data_driven_radius(partition=partition, beta=beta, method=method)
    fournier_bound = fournier_radius(partition=partition, beta=beta)

    print(f"Number of clusters (M) / num_samples (N): {M} / {N} \n"
        f"\t Ours: {data_driven_bound:.4f} \n"
        f"\t Fournier: {fournier_bound:.4f} \n")

