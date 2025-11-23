import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import data_driven_radii_for_combinations, fournier_radii_for_combinations

from plot_utils import set_style, convert_to_sci_notation
set_style()

if __name__ == '__main__':
    args = parse_arguments(
        distribution="Gaussian",
        num_dims=10,
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
    fournier_radii = fournier_radii_for_combinations(args, combinations)

    # Plot gap to Fournier
    fig, ax = plt.subplots(figsize=(6, 4))

    for N in N_options:
        y_vals = []
        for M in M_options:
            key = (args.num_samples_training, N, M)
            radius = data_driven_radii.data[key].radius
            fournier_radius = fournier_radii.data[key]
            y_vals.append(radius / fournier_radius)

        ax.plot(M_options, y_vals, marker="o", label=rf"{convert_to_sci_notation(N)}")

    ax.set_xlabel(r"$M$")
    ax.set_ylabel("Our bound / Fournier")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"$N$", loc="best")

    plt.tight_layout()
    plt.show()