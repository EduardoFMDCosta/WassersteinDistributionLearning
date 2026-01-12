import os
import ot
import torch
import itertools
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, process_args
from experiments.utils import load_list_of_data_driven_radii, load_list_of_empirical_radii

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()


def main(args):
    if args.num_dims == 2:
        settings = [1, 3, 5]
    elif args.num_dims == 10:
        settings = [2, 4, 6]
    else:
        raise ValueError
    
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(settings) - 1)) for i in range(len(settings))]

    for setting, color in zip(settings, colors):
        args.setting = setting
        args = process_args(args)

        combinations = [(args.num_samples_training, args.num_samples, M) for M in M_options]

        data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
        M_options_plot = torch.as_tensor([key[2] for key in data.keys()])
        idx = M_options_plot.argsort()

        ax.plot(M_options_plot[idx], data.mean_radius[idx], label=rf"${convert_to_sci_notation(args.variance**0.5)}$", color=color, marker="o")
        ax.fill_between(
            M_options_plot[idx],
            data.mean_radius[idx] - data.std_radius[idx],
            data.mean_radius[idx] + data.std_radius[idx],
            alpha=0.2,
            color=color
        )

        data_emp = load_list_of_empirical_radii(args, combinations, random_seed_options, save=True)

        M_emp_options_plot = torch.as_tensor([key[2] for key in data_emp.keys()])
        idx_emp = M_emp_options_plot.argsort()

        ax.plot(M_emp_options_plot[idx_emp], data_emp.mean_radius[idx_emp], color=color, marker="o", linestyle="--")
        ax.fill_between(
            M_emp_options_plot[idx_emp],
            data_emp.mean_radius[idx_emp] - data_emp.std_radius[idx_emp],
            data_emp.mean_radius[idx_emp] + data_emp.std_radius[idx_emp],
            alpha=0.2,
            color=color
        )

    ax.set_xlabel(r"Support size of $\widehat{\mathbb{P}}$")
    ax.set_ylabel("Our bound")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"Std deviation", loc="upper right")
    plt.tight_layout()

    if args.save:
        file_name = f"multiscale_W{args.wasserstein_order}_{args.distribution.lower()}_dims_{args.num_dims}_{args.method}"
        folder = os.path.dirname(os.path.dirname(args.figures_dir)) # USE figures_dir! results_dir is solely for data
        plt.savefig(os.path.join(folder, f"{file_name}.pdf"))  
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="Gaussian",
        num_dims=2, # PLACEHOLDER
        setting=0,
        wasserstein_order=2,
        num_samples=1000000,
        beta=1e-6,
        method='triangle_inequality_vertex',
        plot=True,
        save=True,
    )

    for num_dims, method in itertools.product([2, 10], ['triangle_inequality_vertex']):
        args.num_dims = num_dims
        args.method = method
        args = process_args(args)
        main(args)
