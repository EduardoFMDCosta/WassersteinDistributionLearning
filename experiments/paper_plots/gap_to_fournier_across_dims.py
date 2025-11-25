import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, process_args
from experiments.utils import data_driven_radii_for_combinations, fournier_radii_for_combinations

from plot_utils import set_style
set_style()


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="Uniform",
        num_dims=2,
        setting=0,
        wasserstein_order=1,
        num_samples_training=1000,
        num_samples=1000000,
        beta=1e-6,
        method='diagonal_constrained_tp',
        plot=True,
        save=False,
        compute_moment_bound=True,
        compute_discrete_bound=True,
    )
        
    dimensions = [3, 10, 100]
    M_options = [10]

    # Collect data
    ratios = []
    for d in dimensions:
        args.num_dims = d
        process_args(args)

        combinations = [(args.num_samples, M) for M in M_options]

        (quantizations, data_driven_radii), _ = data_driven_radii_for_combinations(args, combinations=combinations,
                                                                                   generate_partition_if_missing=True)
        fournier_radii = fournier_radii_for_combinations(args, combinations)

        # Get our best bound
        min_bound = 1e8
        for M in M_options:
            radius = data_driven_radii.data[(args.num_samples_training, args.num_samples, M)].radius
            if radius < min_bound:
                min_bound = radius

        ratio = min_bound / fournier_radii.data[(args.num_samples_training, args.num_samples, M_options[0])]
        ratios.append(ratio)

    # Plot gap to Fournier
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dimensions, ratios, marker="o")
    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel("Our bound / Fournier")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

