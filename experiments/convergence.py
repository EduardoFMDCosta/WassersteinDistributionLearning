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

@torch.no_grad()
def plot_w2_slice(
    ax, 
    data_driven_radii: DataDrivenRadii, 
    N: Optional[int] = None, 
    M: Optional[int] = None
):
    if (N is not None and M is not None) or (N is None and M is None):
        raise ValueError("Only N or M should be specified.")

    data_sliced = data_driven_radii._slice(N=N, M=M)
    idx = 1 if N is not None else 0
    options = [key[idx] for key in data_driven_radii.keys()]

    ax.plot(options, data_sliced.radius, label='w2', marker='o')
    ax.plot(options, data_sliced.moment_bound, label='e1', linestyle='--')
    ax.plot(options, data_sliced.discrete_bound, label='e2', linestyle=':')
    ax.plot(options, data_sliced.lower_bound, label='lower_bound', linestyle='--')
    ax.set_xlabel(f"Number of {'clusters (M)' if M is None else 'samples (N)'}")
    ax.set_title(f"Number of {'samples (N) = ' if M is None else 'clusters (M)'} = {N if M is None else M}")
    # ax.set_xscale('log')
    ax.legend(loc='best')
    return ax

@torch.no_grad()
def plot_quantization_slice(
    ax, 
    quantizations: Quantizations, 
    stat: str,
    N: Optional[int] = None, 
    M: Optional[int] = None
):
    if (N is not None and M is not None) or (N is None and M is None):
        raise ValueError("Only N or M should be specified.")
    
    data_sliced = quantizations._slice(N=N, M=M)
    idx = 1 if N is not None else 0
    options = [key[idx] for key in quantizations.keys()]

    if stat == 'probs':
        ax.plot(options, data_sliced.mean_range_probs, label='avg probs range', color='black')
        ax.fill_between(
            options,
            data_sliced.mean_range_probs - data_sliced.std_range_probs,
            data_sliced.mean_range_probs + data_sliced.std_range_probs,
            color='black',
            alpha=0.2,
            label='std dev'
        )
    elif stat == 'radii':
        ax.plot(options, data_sliced.mean_cluster_radii, label='avg cluster radii', color='black')
        ax.fill_between(
            options,
            data_sliced.mean_cluster_radii - data_sliced.std_cluster_radii,
            data_sliced.mean_cluster_radii + data_sliced.std_cluster_radii,
            color='black',
            alpha=0.2,
            label='std dev'
        )
    elif stat == 'counts':    
        ax.plot(options, data_sliced.outer_counts, label='outer counts', marker='o', color='red')
        ax.plot(options, data_sliced.mean_cluster_counts, label='avg cluster counts', marker='o', color='blue')
        ax.fill_between(
            options,
            data_sliced.mean_cluster_counts - data_sliced.std_cluster_counts,
            data_sliced.mean_cluster_counts + data_sliced.std_cluster_counts,
            color='blue',
            alpha=0.2,
            label='std dev'
        )
    elif stat == 'locs':
        ax.plot(options, data_sliced.mean_distances_locs, label='avg distances', color='black')
        ax.fill_between(
            options,
            data_sliced.mean_distances_locs - data_sliced.std_distances_locs,
            data_sliced.mean_distances_locs + data_sliced.std_distances_locs,
            color='black',
            alpha=0.2,
            label='std dev'
        )
    else:
        raise ValueError(f"Stat {stat} not recognized. Choose from 'probs', 'radii', 'counts', 'locs'.")
    
    # ax.set_xscale('log')
    ax.legend(loc='best')
    return ax


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

    ax[0] = plot_w2_slice(ax[0], data_driven_radii, N=N_options[0])
    ax[1] = plot_quantization_slice(ax[1], quantizations, stat='probs', N=N_options[0])
    ax[2] = plot_quantization_slice(ax[2], quantizations, stat='radii', N=N_options[0])
    ax[3] = plot_quantization_slice(ax[3], quantizations, stat='counts', N=N_options[0])
    # ax[4] = plot_quantization_slice(ax[4], quantizations, stat='locs', N=N_options[0])

    tag = f"convergence_{args.distribution}_setting={args.setting}"
    if investigate_clusters:
        tag += f"_N={args.num_samples}_M={M_options}"
    else:
        tag += f"_N={N_options}_M={args.num_clusters}"

    # plt.savefig(f"figures{os.sep}convergence{os.sep}{tag}.png")
    plt.show()
        