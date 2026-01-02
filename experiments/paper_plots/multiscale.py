import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, process_args, num_samples_training_from_num_samples
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
        num_samples=1000000,
        beta=1e-6,
        method='joint_diagonal_milp',
        plot=True,
        save=True,
    )

    if args.num_dims == 2:
        settings = [-1, 1, 2, 3, 4]
        random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7]
    elif args.num_dims == 10:
        settings = [2, 3, 4, 5]
        random_seed_options = [0, 1, 2, 3, 4]
    else:
        raise ValueError

    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    

    fig, ax = plt.subplots(figsize=(6, 4)) # (8,4) if legend outside
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(settings) - 1)) for i in range(len(settings))]

    for setting, color in zip(settings, colors):
        args.setting = setting
        args = process_args(args)

        combinations = [(args.num_samples_training, args.num_samples, M) for M in M_options]

        data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
        fournier_radii = fournier_radii_for_combinations(args, [(combi[1], combi[2]) for combi in combinations])

        M_options_plot = [key[2] for key in data.keys(N=args.num_samples, N_train=args.num_samples_training)]
        radii, radii_minus, radii_plus = list(), list(), list()
        for M in M_options_plot:
            key = (args.num_samples_training, args.num_samples, M)

            if key in fournier_radii.keys():
                radii.append(data.mean_radius_at(key))
                radii_minus.append((data.mean_radius_at(key) - data.std_radius_at(key)))
                radii_plus.append((data.mean_radius_at(key) + data.std_radius_at(key)))
            else:
                continue

        M_options_plot, radii = torch.as_tensor(M_options_plot), torch.as_tensor(radii)
        radii_minus, radii_plus = torch.as_tensor(radii_minus), torch.as_tensor(radii_plus)
        idx = M_options_plot.argsort()
        ax.plot(M_options_plot[idx], radii[idx], label=rf"${convert_to_sci_notation(args.variance**0.5)}$", color=color, marker="o")

        ax.fill_between(
            M_options_plot[idx],
            radii_minus[idx],
            radii_plus[idx],
            alpha=0.2,
            color=color
        )


    ax.set_xlabel(r"Support size of $\widehat{\mathbb{P}}$")
    ax.set_ylabel("Our bound")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"Std deviation", 
            #   loc="center left", bbox_to_anchor=(1, 0.5),
              loc="upper right",
              )
    plt.tight_layout()

    if args.save:
        file_name = f"introduction_W{args.wasserstein_order}_{args.method}"
        plt.savefig(os.path.join(os.path.dirname(args.figures_dir), f"{file_name}.pdf"))  # USE figures_dir! results_dir is solely for data

    plt.show()

