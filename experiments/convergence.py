import torch
from sets import KMeansPartition
from plotting.plot import plot_kmeans_partition
from configs.handlers import parse_arguments
from bound import data_driven_radius, fournier_radius
from configs.construct import get_support_assumption, get_distribution

from configs.handlers import parse_arguments

import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Discrete",
        dimension=2,
        setting=0,
        num_samples=5000,
        num_clusters=None,
        beta=1e-4,
        plot=False
    )

    beta = args.beta
    method = 'dual_sinkhorn'
    support_assumption = get_support_assumption(**vars(args))
    # M = args.num_clusters
    N = args.num_samples

    # (Unknown) Generating probability
    distribution = get_distribution(**vars(args))

    # Generate samples
    samples = distribution.sample((N,))
    
    data_driven_bounds, fournier_bounds = list(), list()
    M_options = torch.arange(20, 100, 10).tolist()
    for M in M_options:
        # Clusterize samples (obtaining \hat{P}_M)
        partition = KMeansPartition(support=support_assumption, samples=samples, k=int(M), prefilter=args.distribution == "Discrete")

        # Plot samples and clusterized distribution
        if args.plot:
            plot_kmeans_partition(partition=partition)

        # Compute bounds
        data_driven_output = data_driven_radius(partition=partition, beta=beta, method=method)
        data_driven_bounds.append(data_driven_output.radius)

        fournier_bound = fournier_radius(partition=partition, beta=beta)
        fournier_bounds.append(fournier_bound)

        print(f"Number of clusters (M) / num_samples (N): {M} / {N} \n"
            f"\t Ours: {data_driven_output.radius:.4f} \n"
            f"\t Fournier: {fournier_bound:.4f} \n")

    with torch.no_grad():
        plt.plot(M_options, torch.tensor(data_driven_bounds), label='Ours', marker='o')
        plt.plot(M_options, torch.tensor(fournier_bounds), label='Fournier', marker='x')
        plt.xlabel("Number of clusters (M)")
        plt.legend()
        plt.show()
