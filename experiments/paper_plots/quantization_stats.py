import os
import itertools
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from configs.handlers import parse_arguments, load_json, process_args
from experiments.utils import quantizations_for_combinations, load_quantization_samples

from plotting.utils_plot import set_style, convert_to_sci_notation
import plotting.plot as plot


def main(args, M_options, N_options, plot_samples = True):
    combinations = [(N, M) for N in N_options for M in M_options]

    quantizations = quantizations_for_combinations(args, combinations=combinations, generate_partition_if_missing=False)

    tag = f"N_train={args.num_samples_training}_seed={args.random_seed}"

    fig0, ax0 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(len(N_options) - 1, 1)) for i in range(len(N_options))]

    for N, color in zip(N_options, colors):
        quantizations_slice = quantizations._slice(N_train=args.num_samples_training, N=N)
        M_options_plot = [key[2] for key in quantizations_slice.keys()]

        # Probs
        ax0.plot(M_options_plot, quantizations_slice.mean_range_probs, color=color, marker='o')
        ax0.fill_between(
            M_options_plot,
            quantizations_slice.mean_range_probs - quantizations_slice.std_range_probs,
            quantizations_slice.mean_range_probs + quantizations_slice.std_range_probs,
            color=color,
            alpha=0.2,
        )

        # Counts
        ax1.plot(M_options_plot, quantizations_slice.outer_counts / N, color=color, marker='*')
        ax1.plot(M_options_plot, quantizations_slice.mean_cluster_counts / N, color=color, marker='o')
        ax1.fill_between(
            M_options_plot,
            (quantizations_slice.mean_cluster_counts - quantizations_slice.std_cluster_counts) / N,
            (quantizations_slice.mean_cluster_counts + quantizations_slice.std_cluster_counts) / N,
            color=color,
            alpha=0.2,
        )

        if N == N_options[0]:
            # l2 radii region
            ax2.plot(M_options_plot, quantizations_slice.mean_region_l2_radii, color="black", marker='o')
            ax2.fill_between(
                M_options_plot,
                quantizations_slice.mean_region_l2_radii - quantizations_slice.std_region_l2_radii,
                quantizations_slice.mean_region_l2_radii + quantizations_slice.std_region_l2_radii,
                color="black",
                alpha=0.2,
            )

            # distance between locs
            ax3.plot(M_options_plot, quantizations_slice.mean_l2_distance_locs_to_locs, color="black", marker='o')
            ax3.fill_between(
                M_options_plot,
                quantizations_slice.mean_l2_distance_locs_to_locs - quantizations_slice.std_l2_distance_locs_to_locs,
                quantizations_slice.mean_l2_distance_locs_to_locs + quantizations_slice.std_l2_distance_locs_to_locs,
                color="black",
                alpha=0.2,
            )

    ax0.set_ylabel(r"Mean mass $\frac{1}{2}(p^l_i + p^u_i)$ over $i=1,\ldots, M-1$")
    ax0.set_xlabel(r"Support size $M$")
    ax0.legend(
        handles=[ 
            Line2D([0], [0], color=color, lw=2, label=rf"${convert_to_sci_notation(N)}$")
            for N, color in zip(N_options, colors)
        ],
        loc="upper right"
    )

    ax1.set_ylabel(r"Fraction of samples in $\mathcal{R}_i$")
    ax1.set_xlabel(r"Support size $M$")
    legend_colors = ax1.legend(
        handles=[ 
            Line2D([0], [0], color=color, lw=2, label=rf"${convert_to_sci_notation(N)}$")
            for N, color in zip(N_options, colors)
        ],
        loc="upper right"
    )
    ax1.add_artist(legend_colors)
    ax1.legend(
        handles=[
            Line2D([0], [0], marker='*', color='none', markerfacecolor='black', markersize=10, 
                    label=r"Region $i=M$"),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='black', markersize=8, 
                    label=r"Mean over regions $i=1,\ldots,M-1$")
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.85)
    )

    ax2.set_ylabel(r"Mean radius $r_i$ ($i=1,...M-1$)")
    ax2.set_xlabel(r"Support size $M$")

    ax3.set_ylabel(r"Mean $\|c_i-c_j\|$ ($i,j=1,...M$, $i\neq j$)")
    ax3.set_xlabel(r"Support size $M$")
    ax3.set_ylim(0., 0.4)


    if args.save:
        fig0.savefig(os.path.join(args.figures_dir, f"quantizations_statistics_{tag}_probs.png"))
        fig1.savefig(os.path.join(args.figures_dir, f"quantizations_statistics_{tag}_counts.png"))
        fig2.savefig(os.path.join(args.figures_dir, f"quantizations_statistics_{tag}_l2_radii.png"))
        fig3.savefig(os.path.join(args.figures_dir, f"quantizations_statistics_{tag}_l2_distance_locs.png"))
        plt.close('all')
    else:
        plt.show()
            
if __name__ == '__main__':
    args = parse_arguments( # Only parse arguments once, updated afterwards
        random_seed=0,
        distribution='Gaussian', # PLACEHOLDER
        num_dims=2, # PLACEHOLDER
        setting=0, # PLACEHOLDER
        num_samples=10_000,
        num_samples_training=5_000,
        num_clusters=10,
        save=False,
    )

    M_options = [5, 30, 75]
    N_options = [10_000, 100_000]

    settings = [('GaussianMixture', 2, 0)]
    for distribution, num_dims, setting in settings:
        args.distribution = distribution
        args.num_dims = num_dims
        args.setting = setting
        args = process_args(args)

        try:
            main(args, M_options=M_options, N_options=N_options)
        except Exception as e:
            print(f"Failed for distribution={distribution}, num_dims={num_dims}, setting={setting} with error: {e}")

    # if not args.save:
    #     plt.show()