import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import quantizations_for_combinations
import plotting.plot as plot


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="Gaussian",
        num_dims=2,
        setting=2,
        num_samples=1_000,
        num_samples_training=1_000,
        num_clusters=10,
        plot=True, 
        save=True,
    )
    investigate_clusters = True

    if investigate_clusters:
        N_options = [args.num_samples]
        M_options = [5, 20, 30, 40, 50, 75]
    else:
        N_options = [1000, 2500]
        M_options = [args.num_clusters]

    combinations = [(N, M) for N in N_options for M in M_options]

    quantizations = quantizations_for_combinations(args, combinations=combinations, generate_partition_if_missing=False)

    if investigate_clusters:
        tag = f"N_train={args.num_samples_training}_N={args.num_samples}_M={M_options}"
    else:
        tag = f"N_train={args.num_samples_training}_N={N_options}_M={args.num_clusters}"
    tag += f"_seed={args.random_seed}"

    # Illustrate Quantizations
    if args.num_dims == 2:
        fig, ax = plt.subplots(ncols=len(quantizations.keys()), nrows=1, figsize=(6 * len(quantizations.keys()), 6))
        for i, (N_train, N, M) in enumerate(quantizations.keys()):
            ax[i] = plot.plot_partition(
                ax=ax[i], 
                partition=quantizations.at((N_train, N, M)), 
                # samples=quantizations.samples[:N] if quantizations.samples is not None else None,
                title=f"M={M}, N={N}"
            )

        if args.save:
            plt.savefig(os.path.join(args.figures_dir, f"quantizations_{tag}.png"))
        else:
            plt.show()

    # Plot Statistics
    fig, ax = plt.subplots(4, 1, figsize=(6, 12), constrained_layout=True)
    plot_args = dict(quantizations=quantizations, num_samples_training=args.num_samples_training)
    if investigate_clusters:
        plot_args.update(N=args.num_samples)
    else:
        plot_args.update(M=args.num_clusters)

    ax[0] = plot.plot_quantization_slice(ax[0], stat='probs', **plot_args)
    ax[1] = plot.plot_quantization_slice(ax[1], stat='radii', **plot_args)
    ax[2] = plot.plot_quantization_slice(ax[2], stat='counts', **plot_args)
    ax[3] = plot.plot_quantization_slice(ax[3], stat='locs', **plot_args)
    ax[0].set_title(f"Number of {'samples (N)' if investigate_clusters else 'clusters (M)'} = {args.num_samples if investigate_clusters else args.num_clusters}")
    ax[3].set_xlabel(f"Number of {'clusters (M)' if investigate_clusters else 'samples (N)'}")

    if args.save:
        plt.savefig(os.path.join(args.figures_dir, f"quantizations_statistics_{tag}.png"))
    else:
        plt.show()
        