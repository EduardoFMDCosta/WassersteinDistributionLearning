import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import data_driven_radii_for_combinations, fournier_radii_for_combinations
import plotting.plot as plot


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="Gaussian",
        num_dims=2,
        setting=2,
        wasserstein_order=2,
        num_samples=1_000,
        num_samples_training=1_000,
        num_clusters=10,
        beta=1e-6,
        method='diagonal_constrained_tp',
        plot=True, 
        save=True,
        compute_moment_bound=True,
        compute_discrete_bound=True,
    )
    investigate_clusters = True

    if investigate_clusters:
        N_options = [args.num_samples]
        M_options = [5, 20, 30, 40, 50, 75]
    else:
        N_options = [1000, 2500]
        M_options = [args.num_clusters]

    combinations = [(N, M) for N in N_options for M in M_options]

    data_driven_radii, _ = data_driven_radii_for_combinations(args, combinations=combinations, generate_partition_if_missing=False)
    fournier_radii = fournier_radii_for_combinations(args, combinations)

    # format plot
    file_name = f"W{args.wasserstein_order}_{args.method}_seed={args.random_seed}"
    if investigate_clusters:
        file_name += f"_N_train={args.num_samples_training}_N={args.num_samples}_M={M_options}"
    else:
        file_name += f"_N_train={args.num_samples_training}_N={N_options}_M={args.num_clusters}"

    # Plot Statistics
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax = plot.plot_data_driven_radii_slice(ax, data_driven_radii, num_samples_training=args.num_samples_training , N=N_options[0], cummulative=True)
    ax.set_title(f"Number of {'samples (N)' if investigate_clusters else 'clusters (M)'} = {args.num_samples if investigate_clusters else args.num_clusters}")
    ax.set_xlabel(f"Number of {'clusters (M)' if investigate_clusters else 'samples (N)'}")

    if args.save:
        plt.savefig(os.path.join(args.figures_dir, f"{file_name}.png"))
    else:
        plt.show()
        