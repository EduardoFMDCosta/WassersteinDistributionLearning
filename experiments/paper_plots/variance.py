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
        num_samples=1000000,
        beta=1e-6,
        method='joint_diagonal_milp',
        plot=True,
        save=False,
        compute_moment_bound=True,
        compute_discrete_bound=True,
    )

    settings = [-1, 0, 1, 2, 3, 4, 5]
    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    fig, ax = plt.subplots(figsize=(6, 4))
    for setting in settings:
        args.setting = setting
        args = process_args(args)

        combinations = [(args.num_samples, M) for M in M_options]

        data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
        fournier_radii = fournier_radii_for_combinations(args, combinations)

        M_options_plot = [key[2] for key in data.keys(N=args.num_samples, N_train=args.num_samples_training)]
        ratios, ratios_minus, ratios_plus = list(), list(), list()
        for M in M_options_plot:
            key = (args.num_samples_training, args.num_samples, M)

            if key in fournier_radii.keys():
                ratios.append(data.mean_radius_at(key) / fournier_radii.radius_at(key))
                ratios_minus.append((data.mean_radius_at(key) - data.std_radius_at(key)) / fournier_radii.radius_at(key))
                ratios_plus.append((data.mean_radius_at(key) + data.std_radius_at(key)) / fournier_radii.radius_at(key))
            else:
                continue

        M_options_plot, ratios = torch.as_tensor(M_options_plot), torch.as_tensor(ratios)
        ratios_minus, ratios_plus = torch.as_tensor(ratios_minus), torch.as_tensor(ratios_plus)
        idx = M_options_plot.argsort()
        ax.plot(M_options_plot[idx], ratios[idx], marker="o", label=rf"${setting}$")

        ax.fill_between(
            M_options_plot[idx],
            ratios_minus[idx],
            ratios_plus[idx],
            alpha=0.2,
        )


    ax.set_xlabel(r"$M$")
    ax.set_ylabel("Our bound / Fournier")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"Setting", loc="best")

    plt.tight_layout()
    plt.show()

