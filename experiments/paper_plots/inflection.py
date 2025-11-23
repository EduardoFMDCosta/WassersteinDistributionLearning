import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import data_driven_radii_for_combinations

from plot_utils import set_style, convert_to_sci_notation
set_style()

if __name__ == '__main__':
    args = parse_arguments(
        distribution="GaussianMixture",
        num_dims=3,
        setting=0,
        wasserstein_order=1,
        num_samples_training=1000,
        beta=1e-6,
        method='diagonal_constrained_tp',
        plot=True,
        save=False,
        compute_moment_bound=True,
        compute_discrete_bound=True,
    )

    N_options = [1000, 5000, 10000, 100000, 1000000]
    M_options = [5, 10, 20, 30, 40, 50, 100]

    combinations = [(N, M) for N in N_options for M in M_options]

    (quantizations, data_driven_radii), _ = data_driven_radii_for_combinations(args, combinations=combinations,
                                                                               generate_partition_if_missing=True)

    # Plot inflection curves
    fig, ax = plt.subplots(figsize=(6, 4))

    for N in N_options:
        y_vals = []
        for M in M_options:
            radius = data_driven_radii.data[(args.num_samples_training, N, M)].radius
            y_vals.append(radius)

        ax.plot(M_options, y_vals, marker="o", label=fr"${convert_to_sci_notation(N)}$")

    ax.set_xlabel(r"$M$")
    ax.set_ylabel("Our bound")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"$N$", loc="best")

    plt.tight_layout()
    plt.show()

    # Plot composition
    N_options = [5000, 1000000]
    fig, axes = plt.subplots(
        len(N_options), 1, figsize=(7, 4 * len(N_options)), sharex=True
    )

    if len(N_options) == 1:
        axes = [axes]  # make iterable

    for ax, N in zip(axes, N_options):
        moment_vals = []
        discrete_vals = []
        radius_vals = []

        for M in M_options:
            entry = data_driven_radii.data[(args.num_samples_training, N, M)]

            moment_vals.append(entry.moment_bound)
            discrete_vals.append(entry.discrete_bound)
            radius_vals.append(entry.radius)

        # Plot the lines
        ax.plot(M_options, radius_vals, color="black", linewidth=1.0)
        ax.plot(M_options, moment_vals, color="black", linestyle="--", linewidth=1.0)

        # Stacked area shading
        ax.fill_between(
            M_options,
            0,
            moment_vals,
            alpha=0.3,
            label=r"$\epsilon_1(D_N)$"
        )

        ax.fill_between(
            M_options,
            moment_vals,
            radius_vals,
            alpha=0.3,
            label=r"$\epsilon_2(D_N)$"
        )

        ax.set_ylabel(fr"Our bound (for $N={convert_to_sci_notation(N)}$)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(left=M_options[0])
        ax.legend(loc="lower left")

    axes[-1].set_xlabel(r"$M$")
    plt.tight_layout()
    plt.show()
