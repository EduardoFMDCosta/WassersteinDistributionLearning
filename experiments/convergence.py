from typing import Optional
import torch
import os

from sets import BoundedVoronoiPartition
from quantization import UncertainQuantization
from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from bound import DataDrivenRadius, fournier_radius as compute_fournier_radius
from experiments.utils import DataDrivenRadii, FournierRadii, Quantizations

from configs.construct import get_support_assumption, get_distribution
from configs.handlers import parse_arguments

import matplotlib.pyplot as plt
import plotting.plot as plot


def run_combinations(args, M_options, N_options):
    distribution = get_distribution(**vars(args))
    support_assumption = get_support_assumption(**vars(args))
    
    quantizations, data_driven_radii, fournier_radii = Quantizations(), DataDrivenRadii(), FournierRadii()
    for N in N_options:
        samples_partition = distribution.sample((N,))
        samples_quantization = distribution.sample((N,))

        for M in M_options:
            print(f"Number of clusters (M) / num_samples (N): {M} / {N}")
            
            partition = BoundedVoronoiPartition(
                support=support_assumption, 
                samples=samples_partition, 
                M=M
            )
            quantizations.append((N, M),  UncertainQuantization(
                partition=partition, 
                samples=samples_quantization, 
                beta=args.beta
            ))

            if args.plot:
                plot_quantization(quantization=quantizations.at((N, M)))

            data_driven_radii.append((N, M), DataDrivenRadius(
                quantization=quantizations.at((N, M)),
                method=args.method, 
                compute_moment_bound=args.compute_moment_bound, 
                compute_discrete_bound=args.compute_discrete_bound
            ))
            fournier_radii.append((N, M), compute_fournier_radius(support=partition.support, nsamples=N, beta=args.beta))

            if args.plot:
                plot_quantization(quantization=quantizations.at((N,M)))

    return quantizations, data_driven_radii, fournier_radii


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Uniform",
        dimension=2,
        setting=0,
        num_samples_training=1000,
        num_samples=1000,
        num_clusters=10,
        beta=1e-4,
        plot=False
    )

    args.method = 'cutting_plane'
    args.compute_moment_bound = True
    args.compute_discrete_bound = False

    investigate_clusters = False

    N_options = [1000] #  [1000, 2500, 5000, 7500, 10000]
    M_options = [10, 25]  # [10, 25, 75, 100, 200, 500, 1000]

    quantizations, data_driven_radii, fournier_radii = run_combinations(args, M_options=M_options, N_options=N_options)

    fig, ax = plt.subplots(4, 1, figsize=(8, 12), constrained_layout=True)

    ax[0] = plot.plot_w2_slice(ax[0], data_driven_radii, N=N_options[0])
    ax[1] = plot.plot_quantization_slice(ax[1], quantizations, stat='probs', N=N_options[0])
    ax[2] = plot.plot_quantization_slice(ax[2], quantizations, stat='radii', N=N_options[0])
    ax[3] = plot.plot_quantization_slice(ax[3], quantizations, stat='counts', N=N_options[0])
    # ax[4] = plot.plot_quantization_slice(ax[4], quantizations, stat='locs', N=N_options[0])

    tag = f"convergence_{args.distribution}_setting={args.setting}"
    if investigate_clusters:
        tag += f"_N={args.num_samples}_M={M_options}"
    else:
        tag += f"_N={N_options}_M={args.num_clusters}"

    # plt.savefig(f"figures{os.sep}convergence{os.sep}{tag}.png")
    plt.show()
        