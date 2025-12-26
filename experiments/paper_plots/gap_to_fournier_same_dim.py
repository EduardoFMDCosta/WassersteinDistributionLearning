import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, process_args
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        wasserstein_order=2,
        num_samples_training=5000,
        num_samples=1000,
        beta=1e-6,
        method='joint_diagonal_milp',
        plot=True,
        save=True,
        compute_moment_bound=True,
        compute_discrete_bound=True,
    )

    setting_variance_dict = {0: 0.03 ** 0.5,
                             1: 0.01 ** 0.5,
                             2: 0.001 ** 0.5,
                             3: 0.0001 ** 0.5,
                             4: 0.00001 ** 0.5,
                             5: 0.000001 ** 0.5}

    settings = [1, 2, 3, 4]
    N_options = [1000, 5000, 10000, 100000, 1000000]
    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(settings) - 1)) for i in range(len(settings))]

    for setting, color in zip(settings, colors):
        args.setting = setting
        args = process_args(args)

        combinations = [(N, M) for N in N_options for M in M_options]

        data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
        fournier_radii = fournier_radii_for_combinations(args, combinations)

        ratios, ratios_minus, ratios_plus = list(), list(), list()

        for N in N_options:
            M_options_plot = [key[2] for key in data.keys(N=N, N_train=args.num_samples_training)]

            # Get our best bound
            min_bound = 1e8
            for M in M_options_plot:
                if (args.num_samples_training, N, M) in data.keys():
                    radius = data.mean_radius_at((args.num_samples_training, args.num_samples, M))
                    std = data.std_radius_at((args.num_samples_training, args.num_samples, M))
                    if radius < min_bound:
                        min_bound = radius
                        min_std = std

            fournier_bound = fournier_radii.radius_at((args.num_samples_training, N, M_options_plot[0]))

            ratio = min_bound / fournier_bound
            ratio_minus = (min_bound - min_std) / fournier_bound
            ratio_plus = (min_bound + min_std) / fournier_bound

            ratios.append(ratio)
            ratios_minus.append(ratio_minus)
            ratios_plus.append(ratio_plus)

        N_options_plot, ratios = torch.as_tensor(N_options), torch.as_tensor(ratios)
        ratios_minus, ratios_plus = torch.as_tensor(ratios_minus), torch.as_tensor(ratios_plus)
        idx = N_options_plot.argsort()
        ax.plot(N_options_plot[idx], ratios[idx], label=rf"${convert_to_sci_notation(setting_variance_dict[setting])}$", color=color, marker="o")

        ax.fill_between(
            N_options_plot[idx],
            ratios_minus[idx],
            ratios_plus[idx],
            alpha=0.2,
            color=color
        )

    plt.xscale('log')
    ax.set_xlabel(r"Number of samples $N$")
    ax.set_ylabel("Ours / Fournier (2023)")
    plt.axhline(y=1, color='black', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"Std deviation", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if args.save:
        file_name = f"introduction"
        plt.savefig(os.path.join(args.figures_dir, f"{file_name}.pdf"))

    plt.show()

