import os
import ot
import torch
import itertools
import matplotlib.pyplot as plt

from configs.construct import get_support_assumption, get_distribution
from configs.handlers import parse_arguments, process_args
from experiments.partitions import get_partition
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations, load_quantization

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

    metric = {1: "euclidean", 2: "sqeuclidean"}

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(settings) - 1)) for i in range(len(settings))]

    for setting, color in zip(settings, colors):
        args.setting = setting
        args = process_args(args)

        distribution = get_distribution(**vars(args))

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

        empirical = list()
        for M in M_options_plot:
            partition = get_partition(args=args, num_samples=args.num_samples_training, num_clusters=M)
            quantization = load_quantization(args=args, partition=partition, N=args.num_samples)

            emp_dist = distribution.sample((100000,))

            radius_quantization = ot.solve_sample(X_a=emp_dist, X_b=quantization.locs, b=quantization.probs,
                                                  metric=metric[args.wasserstein_order]).value.pow(1 / args.wasserstein_order).item()

            empirical.append(radius_quantization)

        M_options_plot, radii = torch.as_tensor(M_options_plot), torch.as_tensor(radii)
        radii_minus, radii_plus = torch.as_tensor(radii_minus), torch.as_tensor(radii_plus)
        empirical = torch.as_tensor(empirical)
        idx = M_options_plot.argsort()

        ax.plot(M_options_plot[idx], radii[idx], label=rf"${convert_to_sci_notation(args.variance**0.5)}$", color=color, marker="o")
        ax.fill_between(
            M_options_plot[idx],
            radii_minus[idx],
            radii_plus[idx],
            alpha=0.2,
            color=color
        )

        ax.plot(M_options_plot[idx], empirical[idx], color=color, marker="o", linestyle="--")


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
