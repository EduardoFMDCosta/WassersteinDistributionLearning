from typing import List
import torch
import os

from sets import BoundedVoronoiPartition
from quantization import UncertainQuantization
from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from bound import DataDrivenRadius, fournier_radius
from experiments.utils import RadiiStatistics, UncertainQuantizationStatistics

from configs.construct import get_support_assumption, get_distribution
from configs.handlers import parse_arguments

import matplotlib.pyplot as plt


def num_samples(args, M, N_options):
    support_assumption = get_support_assumption(**vars(args))

    distribution = get_distribution(**vars(args))
    
    quantizations, data_driven_radii, fournier_radii = list(), list(), list()
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
        data_driven_radii.append(DataDrivenRadius(
            quantization=quantization, 
            method=args.method, 
            compute_moment_bound=args.compute_moment_bound, 
            compute_discrete_bound=args.compute_discrete_bound
        ))
        fournier_radii.append(fournier_radius(support=partition.support, nsamples=N, beta=args.beta))

        quantizations.append(quantization)

    return N_options, quantizations, data_driven_radii, fournier_radii


def num_clusters(args, N, M_options):
    support_assumption = get_support_assumption(**vars(args))

    distribution = get_distribution(**vars(args))
    samples_partition = distribution.sample((N,))
    samples_quantization = distribution.sample((N,))

    quantizations, data_driven_radii, fournier_radii = list(), list(), list()
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
        data_driven_radii.append(DataDrivenRadius(
            quantization=quantization, 
            method=args.method, 
            compute_moment_bound=args.compute_moment_bound, 
            compute_discrete_bound=args.compute_discrete_bound
        ))
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
        num_clusters=250,
        beta=1e-4,
        plot=False
    )
    args.method = 'stackelberg_equilibrium'
    args.compute_moment_bound = True
    args.compute_discrete_bound = False

    investigate_clusters = False

    N_options = [1000, 2500, 5000, 7500, 10000]
    M_options = [10, 25, 75, 100, 200, 500, 1000]

    if investigate_clusters:
        options, quantizations, data_driven_radii, fournier_radii = num_clusters(
            args, N=args.num_samples, M_options=M_options)
    else:
        options, quantizations,data_driven_radii, fournier_radii = num_samples(
            args, M=args.num_clusters, N_options=N_options)

    radii_stats = RadiiStatistics(data_driven_radii=data_driven_radii)
    quantization_stats = UncertainQuantizationStatistics(quantizations=quantizations)

    with torch.no_grad():
        fig, ax = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)

        ax[0].plot(options, radii_stats.radius, label='w2', marker='o')
        ax[0].plot(options, radii_stats.moment_bound, label='e1', linestyle='--')
        ax[0].plot(options, radii_stats.discrete_bound, label='e2', linestyle=':')
        ax[0].set_xlabel(f"Number of {'clusters (M)' if investigate_clusters else 'samples (N)'}")
        ax[0].set_title(f"Number of {'samples (N)' if investigate_clusters else 'clusters (M)'} = {args.num_samples if investigate_clusters else args.num_clusters}")
        # ax[0].set_xscale('log')
        ax[0].legend(loc='best')

        ax[1].plot(options, quantization_stats.range_probs_avg, label='avg probs range', color='black')
        ax[1].fill_between(
            options,
            quantization_stats.range_probs_avg - quantization_stats.range_probs_std,
            quantization_stats.range_probs_avg + quantization_stats.range_probs_std,
            color='black',
            alpha=0.2,
            label='std dev'
        )
        # ax[1].set_xscale('log')
        ax[1].legend(loc='best')

        ax[2].plot(options, quantization_stats.cluster_radius_avg, label='avg cluster radii', color='black')
        ax[2].fill_between(
            options,
            quantization_stats.cluster_radius_avg - quantization_stats.cluster_radius_std,
            quantization_stats.cluster_radius_avg + quantization_stats.cluster_radius_std,
            color='black',
            alpha=0.2,
            label='std dev'
        )
        # ax[2].set_xscale('log')
        ax[2].legend(loc='best')

        ax[3].plot(options, quantization_stats.distances_locs_avg, label='avg distances', color='black')
        ax[3].fill_between(
            options,
            quantization_stats.distances_locs_avg - quantization_stats.distances_locs_std,
            quantization_stats.distances_locs_avg + quantization_stats.distances_locs_std,
            color='black',
            alpha=0.2,
            label='std dev'
        )
        # ax[3].set_xscale('log')
        ax[3].legend(loc='best')

        tag = f"convergence_{args.distribution}_setting={args.setting}"
        if investigate_clusters:
            tag += f"_N={args.num_samples}_M={M_options}"
        else:
            tag += f"_N={N_options}_M={args.num_clusters}"

        plt.savefig(f"figures{os.sep}convergence{os.sep}{tag}.png")
        plt.show()