import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import load_list_of_data_driven_radii

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()

if __name__ == '__main__':
    args = parse_arguments(
        distribution="Gaussian",
        num_dims=3,
        setting=0,
        wasserstein_order=2,
        num_samples_training=5000,
        beta=1e-6,
        method='joint_diagonal_milp',
        plot=True,
        save=True,
        compute_moment_bound=True,
        compute_discrete_bound=True,
    )

    N_options = [1000, 5000, 10000, 100000, 1000000]
    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    combinations = [(N, M) for N in N_options for M in M_options]

    data = load_list_of_data_driven_radii(args, combinations, random_seed_options)

    fig, ax = plt.subplots(figsize=(6, 4))

    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(N_options) - 1)) for i in range(len(N_options))]
    for N, color in zip(N_options, colors):
        data_slice = data._slice(N_train=args.num_samples_training, N=N)
        M_options_plot = torch.as_tensor([key[2] for key in data_slice.keys()])
        idx = M_options_plot.argsort()

        ax.plot(M_options_plot[idx], data_slice.mean_radius[idx], label=rf"${convert_to_sci_notation(N)}$", color=color, marker="o")
        ax.fill_between(
            M_options_plot[idx],
            data_slice.mean_radius[idx] - data_slice.std_radius[idx],
            data_slice.mean_radius[idx] + data_slice.std_radius[idx],
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel(r"Support size $M$")
    ax.set_ylabel("Ours")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"$N$", loc="best")
    plt.tight_layout()

    if args.save:
        file_name = f"inflection_W{args.wasserstein_order}_N_train={args.num_samples_training}_{args.method}"
        plt.savefig(os.path.join(args.figures_dir, f"{file_name}.pdf"))

    plt.show()
