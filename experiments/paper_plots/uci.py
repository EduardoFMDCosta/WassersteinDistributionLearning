import os
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from configs.handlers import parse_arguments, num_samples_training_from_num_samples, process_args
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()


def main(args):
    N_options = [1000, 5000, 7500, 10000, 25000]

    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500]
    if args.method != 'joint_diagonal_milp':
        M_options += [1000]
    
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    combinations = [(num_samples_training_from_num_samples(N), N, M) for N in N_options for M in M_options]

    data = load_list_of_data_driven_radii(args, combinations, random_seed_options)

    fig, ax = plt.subplots(figsize=(8, 4))  # TODO change back to 6, 4

    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(N_options) - 1)) for i in range(len(N_options))]
    for N, color in zip(N_options, colors):
        data_slice = data._slice(N_train=num_samples_training_from_num_samples(N), N=N)
        if data_slice.keys() == []:
            continue

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

    handles, _ = ax.get_legend_handles_labels()

    fournier_data = fournier_radii_for_combinations(
        args, 
        combinations=[(args.num_samples, int(M)) for M in M_options_plot]
    )
    M_options_fornier_plots = torch.as_tensor([key[2] for key in fournier_data.keys(N=args.num_samples, N_train=args.num_samples_training)])
    idx_fournier = M_options_fornier_plots.argsort()

    ax.plot(M_options_fornier_plots[idx_fournier], fournier_data.radius[idx_fournier], color='grey', linestyle="--")

    ax.set_xlabel(r"Support size $M$")
    if args.distribution == 'UCI-Turbine':
        ax.set_ylabel(r"$\mathbb{W}_2$")

    ax.grid(True, linestyle="--", alpha=0.4)

    fournier_legend = Line2D([], [], color="grey", linestyle="--", label=rf"[13] ($N=10^4$)")
    divider = Line2D([], [], linestyle="none", label=r"$\rule{2cm}{0.4pt}$")
    if args.distribution == 'UCI-Turbine':
        ax.legend(
            handles=handles + [divider, fournier_legend],
            title=r"$N$",
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            bbox_transform=fig.transFigure,
            frameon=False
        )
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)


    if args.save:
        file_name = f"inflection_incl_fournier_W{args.wasserstein_order}_{args.distribution.lower()}_dims_{args.num_dims}_setting_{args.setting}_{args.method}"
        folder = os.path.dirname(os.path.dirname(args.figures_dir)) # USE figures_dir! results_dir is solely for data
        plt.savefig(os.path.join(folder, f"{file_name}.pdf"))  
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_arguments(
        distribution="UCI-Turbine",  # PLACEHOLDE
        num_dims=11,  # PLACEHOLDE
        setting=0,
        wasserstein_order=2,
        num_samples=10_000,
        beta=1e-6,
        method='triangle_inequality_vertex',
        plot=True,
        save=True
    )

    settings = [
        ("UCI-Turbine", 11),
        # ("UCI-MiniBooNE", 50),
    ]

    for distribution, num_dims in settings:
        args.distribution = distribution
        args.num_dims = num_dims
        args = process_args(args)
        main(args)

