import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, process_args, num_samples_training_from_num_samples
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()


def main(args):
    N_options = [5000, 10000, 100000, 1000000]
    dimensions = [2, 10, 25, 50, 75, 100]
    M_options = [5, 20, 30, 40, 50, 75]

    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Plot gap to Fournier
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(N_options) - 1)) for i in range(len(N_options))]

    for N, color in zip(N_options, colors):
        args.num_samples = N
        N_train = num_samples_training_from_num_samples(N)

        combinations = [(N_train, N, M) for M in M_options]

        ratios, ratios_minus, ratios_plus, dimensions_plot = list(), list(), list(), list()
        for d in dimensions:
            args.num_dims = d
            process_args(args)

            data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
            fournier_radii = fournier_radii_for_combinations(args, [(combi[1], combi[2]) for combi in combinations])

            # Get our best bound
            i = 0
            for M in M_options:
                if (N_train, N, M) in data.keys():
                    radius = data.mean_radius_at((N_train, N, M))
                    std = data.std_radius_at((N_train, N, M))

                    if i == 0:
                        min_bound = radius
                        min_std = std
                    elif radius < min_bound:
                        min_bound = radius
                        min_std = std

                    i += 1

            if i > 1 and ((N_train, N, M_options[0]) in fournier_radii.keys()):
                fournier_bound = fournier_radii.radius_at((N_train, N, M_options[0]))

                ratio = min_bound / fournier_bound
                ratio_minus = (min_bound - min_std) / fournier_bound
                ratio_plus = (min_bound + min_std) / fournier_bound

                ratios.append(ratio)
                ratios_minus.append(ratio_minus)
                ratios_plus.append(ratio_plus)
                dimensions_plot.append(d)

        ax.plot(dimensions_plot, ratios, label=rf"${convert_to_sci_notation(N)}$", color=color, marker="o")
        ax.fill_between(
            dimensions_plot,
            ratios_minus,
            ratios_plus,
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel(r"Dimension $d$")
    ax.grid(True, linestyle="--", alpha=0.4)
    if args.method == 'triangle_inequality_vertex' and args.distribution == 'Uniform':
        ax.legend(title=r"$N$")
        ax.set_ylabel("Ours / Fournier")
    plt.tight_layout()

    if args.save:
        file_name = f"gap_dims_W{args.wasserstein_order}_{args.distribution.lower()}_setting={args.setting}_{args.method}"
        folder = os.path.dirname(os.path.dirname(args.figures_dir)) # USE figures_dir! results_dir is solely for data
        plt.savefig(os.path.join(folder, f"{file_name}.pdf"))  
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_arguments(
        distribution="Uniform", # PLACEHOLDER
        num_dims=2,
        setting=0,
        wasserstein_order=1, # PLACEHOLDE 
        num_samples=10000,
        beta=1e-6,
        method='joint_diagonal_milp', # PLACEHOLDE
        plot=True,
        save=True,
    )

    settings = [
        ('Uniform', 1, 'triangle_inequality_vertex'), ('Uniform', 1, 'joint_diagonal_milp'),
        ('TruncatedGaussian', 2, 'triangle_inequality_vertex'), ('TruncatedGaussian', 2, 'joint_diagonal_milp')
    ]

    for distribution, wasserstein_order, method in settings:
        args.distribution = distribution
        args.wasserstein_order = wasserstein_order
        args.method = method
        args = process_args(args)
        main(args)


