from typing import List
import torch

from sets import BoundedVoronoiPartition
from quantization import UncertainQuantization
from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from bound import DataDrivenRadius, fournier_radius
from experiments.statistics import RadiiStatistics, UncertainQuantizationStatistics

from configs.construct import get_support_assumption, get_distribution
from configs.handlers import parse_arguments

import matplotlib.pyplot as plt


def num_samples(args, M):
    support_assumption = get_support_assumption(**vars(args))

    distribution = get_distribution(**vars(args))
    
    quantizations, data_driven_radii, fournier_radii = list(), list(), list()
    N_options = [1000, 5000, 10000, 50000]
    for N in N_options:
        print(f"Number of clusters (M) / num_samples (N): {M} / {N}")
        samples_partition = distribution.sample((N,))
        samples_quantization = distribution.sample((N,))

        # Clusterize samples (obtaining \hat{P}_M)
        partition = BoundedVoronoiPartition(
            support=support_assumption, 
            samples=samples_partition, 
            M=M,
            use_voronoi_radii=False # set to false to speed up
        )
        quantization = UncertainQuantization(partition=partition, samples=samples_quantization, beta=args.beta)

        # Plot samples and clusterized distribution
        if args.plot:
            plot_quantization(quantization=quantization)

        # Compute bounds
        data_driven_radii.append(DataDrivenRadius(quantization=quantization, method=args.method))
        fournier_radii.append(fournier_radius(support=partition.support, nsamples=N, beta=args.beta))

        quantizations.append(quantization)

    return N_options, quantizations, data_driven_radii, fournier_radii


def num_clusters(args, N):
    support_assumption = get_support_assumption(**vars(args))

    distribution = get_distribution(**vars(args))
    samples_partition = distribution.sample((N,))
    samples_quantization = distribution.sample((N,))

    quantizations, data_driven_radii, fournier_radii = list(), list(), list()
    M_options = [10, 15]
    for M in M_options:
        print(f"Number of clusters (M) / num_samples (N): {M} / {N}")
        # Clusterize samples (obtaining \hat{P}_M)
        partition = BoundedVoronoiPartition(
            support=support_assumption, 
            samples=samples_partition, 
            M=M,
            use_voronoi_radii=False # set to false to speed up
        )
        quantization = UncertainQuantization(partition=partition, samples=samples_quantization, beta=args.beta)

        # Plot samples and clusterized distribution
        if args.plot:
            plot_quantization(quantization=quantization)

        # Compute bounds
        data_driven_radii.append(DataDrivenRadius(quantization=quantization, method=args.method))
        fournier_radii.append(fournier_radius(support=partition.support, nsamples=N, beta=args.beta))
        quantizations.append(quantization)

    return M_options, quantizations, data_driven_radii, fournier_radii


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Uniform",
        dimension=2,
        setting=0,
        num_samples=5000,
        num_clusters=40,
        beta=1e-4,
        plot=False
    )
    args.method = 'stackelberg_equilibrium'
    investigate_clusters = True

    if investigate_clusters:
        options, quantizations, data_driven_radii, fournier_radii = num_clusters(args, N=args.num_samples)
    else:
        options, quantizations,data_driven_radii, fournier_radii = num_samples(args, M=args.num_clusters)

    radii_stats = RadiiStatistics(data_driven_radii=data_driven_radii)
    quantization_stats = UncertainQuantizationStatistics(quantizations=quantizations)

    with torch.no_grad():
        fig, ax = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

        ax[0].plot(options, radii_stats.radius, label='w2', marker='o')
        ax[0].plot(options, radii_stats.epsilon1, label='e1', linestyle='--')
        ax[0].plot(options, radii_stats.epsilon2, label='e2', linestyle=':')
        ax[0].set_xlabel(f"Number of {'clusters (M)' if investigate_clusters else 'samples (N)'}")
        ax[0].set_title(f"Number of {'samples (N)' if investigate_clusters else 'clusters (M)'} = {args.num_samples if investigate_clusters else args.num_clusters}")
        # ax[0].set_xscale('log')
        ax[0].legend(loc='best')

        ax[1].plot(options, quantization_stats.lower_probs_avg, label='avg lower prob', color='red')
        ax[1].plot(options, quantization_stats.probs_avg, label='avg prob', color='black')
        ax[1].plot(options, quantization_stats.upper_probs_avg, label='avg upper prob', color='green')
        # ax[1].set_xscale('log')
        ax[1].legend(loc='best')

        ax[2].plot(options, quantization_stats.cluster_radius_min, label='min cluster radii', color='red')
        ax[2].plot(options, quantization_stats.cluster_radius_avg, label='avg cluster radii', color='black')
        ax[2].plot(options, quantization_stats.cluster_radius_max, label='max cluster radii', color='green')
        # ax[2].set_xscale('log')
        ax[2].legend(loc='best')

        plt.show()