import os
import itertools
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from configs.handlers import parse_arguments, load_json, process_args
from experiments.utils import quantizations_for_combinations, load_quantization_samples
import plotting.plot as plot


def main(args, M_options, N_options, plot_samples = True):
    combinations = [(N, M) for N in N_options for M in M_options]

    quantizations = quantizations_for_combinations(args, combinations=combinations, generate_partition_if_missing=False)

    tag = f"N_train={args.num_samples_training}_seed={args.random_seed}"

    # Illustrate Quantizations
    if args.num_dims == 2:
        fig, ax = plt.subplots(ncols=len(M_options), nrows=len(N_options), figsize=(5. * len(M_options), 6 * len(N_options)), squeeze=False)
        for i, N in enumerate(N_options):
            if plot_samples:
                samples = load_quantization_samples(args, N, generate_samples_if_missing=False)
            else:
                samples = None

            for j, M in enumerate(M_options):
                if (args.num_samples_training, N, M) in quantizations.keys():
                    ax[i, j] = plot.plot_quantization(
                        ax=ax[i, j], 
                        quantization=quantizations.at((args.num_samples_training, N, M)), 
                        samples=samples,
                        # title=f"M={M}, N={N}"
                    )
                else:
                    ax[i, j].set_visible(False)

        if args.save:
            fig.tight_layout()
            plt.savefig(os.path.join(args.figures_dir, f"quantizations_{tag}.png"))
            plt.close('all')

    # Plot Statistics
    fig, ax = plt.subplots(4, 1, figsize=(6, 12), constrained_layout=True)

    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(len(N_options) - 1, 1)) for i in range(len(N_options))]
    for N, color in zip(N_options, colors):
        quantizations_slice = quantizations._slice(N_train=args.num_samples_training, N=N)
        M_options_plot = [key[2] for key in quantizations_slice.keys()]

        # Probs
        ax[0].plot(M_options_plot, quantizations_slice.mean_range_probs, label=str(N), color=color, marker='o')
        ax[0].fill_between(
            M_options_plot,
            quantizations_slice.mean_range_probs - quantizations_slice.std_range_probs,
            quantizations_slice.mean_range_probs + quantizations_slice.std_range_probs,
            color=color,
            alpha=0.2,
        )

        # Counts
        ax[1].plot(M_options_plot, quantizations_slice.outer_counts + 1, color=color, marker='*')
        ax[1].plot(M_options_plot, quantizations_slice.mean_cluster_counts, color=color, marker='o')
        ax[1].fill_between(
            M_options_plot,
            quantizations_slice.mean_cluster_counts - quantizations_slice.std_cluster_counts,
            quantizations_slice.mean_cluster_counts + quantizations_slice.std_cluster_counts,
            color=color,
            alpha=0.2,
        )

        # l2 radii region
        ax[2].plot(M_options_plot, quantizations_slice.mean_region_l2_radii, color=color, marker='o')
        ax[2].fill_between(
            M_options_plot,
            quantizations_slice.mean_region_l2_radii - quantizations_slice.std_region_l2_radii,
            quantizations_slice.mean_region_l2_radii + quantizations_slice.std_region_l2_radii,
            color=color,
            alpha=0.2,
        )

        # distance between locs
        ax[3].plot(M_options_plot, quantizations_slice.mean_l2_distance_locs_to_locs, color=color, marker='o')
        ax[3].fill_between(
            M_options_plot,
            quantizations_slice.mean_l2_distance_locs_to_locs - quantizations_slice.std_l2_distance_locs_to_locs,
            quantizations_slice.mean_l2_distance_locs_to_locs + quantizations_slice.std_l2_distance_locs_to_locs,
            color=color,
            alpha=0.2,
        )

    ax[0].set_title(f"N_train: {args.num_samples_training}")
    ax[0].legend(title=f"N", loc="upper right")

    ax[1].set_ylim(1., ax[1].get_ylim()[1])
    ax[1].set_yscale('log')
    ax[1].legend(
        handles=[
            Line2D([0], [0], marker='*', color='none', markerfacecolor='black', markersize=10, label="outer count + 1"),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markersize=8, label="avg cluster counts")
        ],
        title=f"N", 
        loc="lower left"
    )
    
    ax[3].set_xlabel(f"Number of clusters (M)")

    if args.save:
        plt.savefig(os.path.join(args.figures_dir, f"quantizations_statistics_{tag}.png"))
        plt.close('all')
            
if __name__ == '__main__':
    args = parse_arguments( # Only parse arguments once, updated afterwards
        random_seed=0,
        distribution='Gaussian', # PLACEHOLDER
        num_dims=2, # PLACEHOLDER
        setting=0, # PLACEHOLDER
        num_samples=10_000,
        num_samples_training=5_000,
        num_clusters=10,
        save=True,
    )

    M_options = [5, 30, 75]
    N_options = [10_000]

    params = load_json("parameters")
    settings = [(d, int(n), int(s)) for d in params.keys() for n in params[d]["num_dims"].keys() for s in params[d]["num_dims"][n]["settings"].keys()]

    settings = [('Uniform', 2, 0), ('Uniform', 2, 1), ('GaussianMixture', 2, 0)]  # TEMPORARY LIMITATION FOR DEBUGGING
    for distribution, num_dims, setting in settings:
        args.distribution = distribution
        args.num_dims = num_dims
        args.setting = setting
        args = process_args(args)

        try:
            main(args, M_options=M_options, N_options=N_options)
        except Exception as e:
            print(f"Failed for distribution={distribution}, num_dims={num_dims}, setting={setting} with error: {e}")

    if not args.save:
        plt.show()