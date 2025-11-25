import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, process_args
from experiments.utils import data_driven_radii_for_combinations, fournier_radii_for_combinations

from plot_utils import set_style, convert_to_sci_notation
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

    settings = [0, 1, 2, 3, 4]
    M_options = [10, 20]

    fig, ax = plt.subplots(figsize=(6, 4))
    for setting in settings:
        args.setting = setting
        args = process_args(args)
        
        combinations = [(args.num_samples, M) for M in M_options]

        (quantizations, data_driven_radii), _ = data_driven_radii_for_combinations(args, combinations=combinations, generate_partition_if_missing=True)
        fournier_radii = fournier_radii_for_combinations(args, combinations)

        ratios = []
        for M in M_options:
            key = (args.num_samples_training, args.num_samples, M)
            radius = data_driven_radii.data[key].radius
            fournier_radius = fournier_radii.data[key]
            ratios.append(radius / fournier_radius)
        ax.plot(M_options, ratios, marker="o", label=rf"${setting}$")

    ax.set_xlabel(r"$M$")
    ax.set_ylabel("Our bound / Fournier")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"Setting", loc="best")

    plt.tight_layout()
    plt.show()

