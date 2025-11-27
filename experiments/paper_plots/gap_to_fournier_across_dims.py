import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, process_args
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()


if __name__ == '__main__':
    args = parse_arguments(
        distribution="Uniform",
        num_dims=2,
        setting=0,
        wasserstein_order=1,
        num_samples_training=5000,
        num_samples=1000,
        beta=1e-6,
        method='diagonal_constrained_tp',
        plot=True,
        save=False,
        compute_moment_bound=True,
        compute_discrete_bound=True,
    )
        
    dimensions = [2, 10, 100]
    M_options = [5, 20, 30, 40, 50]
    random_seed_options = [0]

    # Collect data
    ratios = []
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
                if radius < min_bound:
                    min_bound = radius

        ratio = min_bound / fournier_radii.radius_at((args.num_samples_training, args.num_samples, M_options[0]))
        ratios.append(ratio)

    # Plot gap to Fournier
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dimensions, ratios, marker="o")
    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel("Our bound / Fournier")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

