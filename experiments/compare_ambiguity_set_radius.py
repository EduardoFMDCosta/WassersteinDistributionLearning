import torch
from sets import Partition
from plotting.plot import plot_samples
from configs.handlers import parse_arguments
from utils import generate_grid_from_samples
from bound import data_driven_radius, fournier_radius
from configs.construct import get_support_assumption, get_support, get_distribution

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="GaussianMixture",
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
    support = get_support(**vars(args))
    distribution = get_distribution(**vars(args))

    # Generate samples
    samples = distribution.sample((N,))

    # Clusterize samples (obtaining \hat{P}_M)
    dim = support.lower.shape[-1]
    locs = generate_grid_from_samples(samples, int(M ** (1 / dim)))
    partition = Partition(locs=locs, support=support_assumption)
    for i in range(2):
        partition = partition.refine(samples=samples, prob_thr=0.01, diam_thr=0.1)

    # Plot samples and clusterized distribution
    if args.plot:
        plot_samples(samples=samples, regions=partition.regions, support_assumption=support_assumption)

    # Compute bounds
    data_driven_bound = data_driven_radius(samples=samples, partition=partition, beta=beta, method=method)
    print(f"Ours: {data_driven_bound}")

    fournier_bound = fournier_radius(samples=samples, partition=partition, beta=beta)
    print(f"Fournier: {fournier_bound}")