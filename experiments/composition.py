import torch
from confidence import ClopperPearsonConfidence
from sets import KMeansPartition
from plotting.plot import plot_kmeans_partition
from bound import data_driven_radius, fournier_radius, bound_moment, bound_discrete
from configs.construct import get_support_assumption, get_distribution
from configs.handlers import parse_arguments

import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="GaussianMixture",
        dimension=2,
        setting=0,
        num_samples=5000,
        num_clusters=10,
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

    moment_bounds, discrete_bounds = list(), list()
    M_options = [10, 20, 50, 70, 100, 150, 200, 500, 1000]
    for M in M_options:
        # Clusterize samples (obtaining \hat{P}_M)
        partition = KMeansPartition(support=support_assumption, samples=samples, k=int(M), prefilter=args.distribution == "Discrete")

        # Plot samples and clusterized distribution
        if args.plot:
            plot_kmeans_partition(partition=partition)

        # Compute bounds
        data_driven_output = data_driven_radius(partition=partition, beta=beta, method=method)
        moment_bounds.append(data_driven_output.moment_bound)
        discrete_bounds.append(data_driven_output.discrete_bound)


    with torch.no_grad():
        fig, ax = plt.subplots()
        ax.set_xscale('log')

        total_bounds = [m.item() + d for (m, d) in zip(moment_bounds, discrete_bounds)]

        # Plot stacked area
        ax.fill_between(M_options, 0, moment_bounds, label=r'$\epsilon_1$', alpha=0.4)
        ax.fill_between(M_options, moment_bounds, total_bounds, label=r'$\epsilon_2$', alpha=0.4)

        # Optional: plot the lines for clarity
        ax.plot(M_options, moment_bounds, color='black', linestyle='--', linewidth=1)
        ax.plot(M_options, total_bounds, color='black', linewidth=1)

        # Labels and legend
        ax.set_xlabel(r"$\log M$")
        ax.set_ylabel("Bound")
        ax.legend()
        plt.tight_layout()
        plt.show()
