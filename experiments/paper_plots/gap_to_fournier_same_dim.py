import torch
import matplotlib.pyplot as plt
import os

from configs.handlers import parse_arguments
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations

from plotting.utils_plot import set_style, convert_to_sci_notation

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
    random_seed_options = [0]

    combinations = [(N, M) for N in N_options for M in M_options]

    data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
    fournier_radii = fournier_radii_for_combinations(args, combinations)

    fig, ax = plt.subplots(figsize=(6, 4))

    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(N_options) - 1)) for i in range(len(N_options))]

    for N, color in zip(N_options, colors):
        data_slice = data._slice(N_train=args.num_samples_training, N=N)
        fournier_radii_slice = fournier_radii._slice(N_train=args.num_samples_training, N=N)

        M_options_plot, ratios = list(), list()
        for M in [key[2] for key in data_slice.keys(N=N, N_train=args.num_samples_training)]:
            key = (args.num_samples_training, N, M)
            
            if key in fournier_radii_slice.keys():
                ratios.append(data_slice.mean_radius_at(key) / fournier_radii_slice.radius_at(key))
            else:
                continue

            M_options_plot.append(M)

        M_options_plot, ratios = torch.as_tensor(M_options_plot), torch.as_tensor(ratios)
        idx = M_options_plot.argsort()
        ax.plot(M_options_plot[idx], [ratios[i] for i in idx], label=str(N), color=color, marker="o")

    
    ax.set_xlabel(f"Number of clusters (M)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=f"N", loc="upper right")
    ax.set_ylabel("Our bound / Fournier")

    if args.save: # Use the same naming convention as in compare_methods.py
        file_name = f"W{args.wasserstein_order}_N_train={args.num_samples_training}_{args.method}_vs_fournier"
        plt.savefig(os.path.join(args.figures_dir, f"{file_name}.png"))

    plt.tight_layout()
    plt.show()
    plt.show()