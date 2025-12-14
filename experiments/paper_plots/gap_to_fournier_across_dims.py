import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, process_args
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()

if __name__ == '__main__':
    dimensions = [2, 3, 10, 25, 50, 75, 100]
    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    args = parse_arguments(
        distribution="Uniform",
        num_dims=2,
        setting=0,
        wasserstein_order=1,
        num_samples_training=5000,
        num_samples=10000,
        beta=1e-6,
        method='joint_diagonal_milp',
        plot=True,
        save=False,
        compute_moment_bound=True,
        compute_discrete_bound=True,
    )

    N_options = [1000, 5000, 10000, 100000, 1000000]

    # Plot gap to Fournier
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(N_options) - 1)) for i in range(len(N_options))]

    for N, color in zip(N_options, colors):
        args.num_samples = N
        ratios, ratios_minus, ratios_plus = list(), list(), list()
        for d in dimensions:
            args.num_dims = d
            process_args(args)

            combinations = [(args.num_samples, M) for M in M_options]

            data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
            fournier_radii = fournier_radii_for_combinations(args, combinations)

            # Get our best bound
            min_bound = 1e8
            for M in M_options:
                if (args.num_samples_training, args.num_samples, M) in data.keys():
                    radius = data.mean_radius_at((args.num_samples_training, args.num_samples, M))
                    std = data.std_radius_at((args.num_samples_training, args.num_samples, M))
                    if radius < min_bound:
                        min_bound = radius
                        min_std = std

            fournier_bound = fournier_radii.radius_at((args.num_samples_training, args.num_samples, M_options[0]))

            ratio = min_bound / fournier_bound
            ratio_minus = (min_bound - min_std) / fournier_bound
            ratio_plus = (min_bound + min_std) / fournier_bound

            ratios.append(ratio)
            ratios_minus.append(ratio_minus)
            ratios_plus.append(ratio_plus)

        ax.plot(dimensions, ratios, label=rf"${convert_to_sci_notation(N)}$", color=color, marker="o")
        ax.fill_between(
            dimensions,
            ratios_minus,
            ratios_plus,
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel("Ours / Fournier (2023)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"$N$")

    plt.tight_layout()
    plt.show()

