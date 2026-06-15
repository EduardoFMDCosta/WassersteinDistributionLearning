import os
import itertools
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from configs.handlers import parse_arguments, process_args
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()


def main(args):
    if args.num_dims == 2:
        settings = [-1, 1, 2, 3]
    elif args.num_dims == 10:
        settings = [2, 3, 4, 5]
    else:
        raise ValueError

    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, ]
    if args.method == 'triangle_inequality_vertex':
        M_options += [500, 1000]
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    fig, ax = plt.subplots(figsize=(5.4, 4))
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(settings) - 1)) for i in range(len(settings))]

    for setting, color in zip(settings, colors):
        args.setting = setting
        args = process_args(args)

        combinations = [(args.num_samples_training, args.num_samples, M) for M in M_options]

        data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
        M_options_plot = torch.as_tensor([key[2] for key in data.keys(N=args.num_samples, N_train=args.num_samples_training)])
        idx = M_options_plot.argsort()

        ax.plot(M_options_plot[idx], data.mean_radius[idx], label=rf"${convert_to_sci_notation(args.variance**0.5)}$", color=color, marker="o")
        ax.fill_between(
            M_options_plot[idx],
            data.mean_radius[idx] - data.std_radius[idx],
            data.mean_radius[idx] + data.std_radius[idx],
            alpha=0.2,
            color=color
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
    ax.grid(True, linestyle="--", alpha=0.4)
    if args.method == 'joint_diagonal_milp':
        ax.set_ylabel(r"$\mathbb{W}_2$")


    plt.tight_layout()


    fournier_legend = Line2D([], [], color="grey", linestyle="--", label=rf"[13] ($N=10^4$)")
    divider = Line2D([], [], linestyle="none", label=r"$\rule{2cm}{0.4pt}$")
    fig_leg = plt.figure(figsize=(2.8, 3.0))
    fig_leg.legend(
        handles=handles + [divider, fournier_legend],
        title=r"Std deviation",
        loc="center",
        frameon=False,
    )

    if args.save:
        file_name = f"variance_W{args.wasserstein_order}_{args.distribution.lower()}_dims_{args.num_dims}_{args.method}"
        folder = os.path.dirname(os.path.dirname(args.figures_dir)) # USE figures_dir! results_dir is solely for data
        fig.savefig(os.path.join(folder, f"{file_name}.pdf"))  
        fig_leg.savefig(os.path.join(folder, f"variance_legend.pdf"))
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="TruncatedGaussian",
        num_dims=10,
        setting=0,
        wasserstein_order=2,
        num_samples=10_000,
        beta=1e-6,
        method='triangle_inequality_vertex',
        plot=True,
        save=True,
    )

    for num_dims, method in itertools.product([2, 10], ['triangle_inequality_vertex', 'joint_diagonal_milp']):
        args.num_dims = num_dims
        args.method = method
        args = process_args(args)
        main(args)

