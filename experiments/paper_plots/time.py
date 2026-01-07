import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, num_samples_training_from_num_samples, process_args
from experiments.utils import load_list_of_time_loggers
from experiments.partitions import load_list_of_time_logger_partition

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()

def main(args, M_setting: str):
    N_options = [1000, 5000, 10000, 100000, 1000000]
    if M_setting == 'small':
        M_options = [5, 20, 30, 40, 50, 75, 100, 150]
    elif M_setting == 'large':
        M_options = [200, 500, 1000]
    else:
        raise ValueError(f"Unknown M_setting: {M_setting}")
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    cmap = plt.cm.coolwarm

    radii_times = load_list_of_time_loggers(args, random_seed_options)

    fig_radii, ax_radii = plt.subplots(figsize=(6, 4))

    colors = [cmap(i / max(len(N_options) - 1, 1)) for i in range(len(N_options))]
    M_options_radii = torch.as_tensor(M_options)

    for N, color in zip(N_options, colors):
        N_train = num_samples_training_from_num_samples(N)

        mean_radii, std_radii = list(), list()
        for M in M_options:
            if (N_train, N, M) in radii_times.keys():
                mean_radii.append(radii_times.mean_time_at((N_train, N, M)))
                std_radii.append(radii_times.std_time_at((N_train, N, M)))
            else:
                mean_radii.append(0.0) # TODO: Discuss how to represent here, although we should have access to all keys
                std_radii.append(0.0)

        mean_radii, std_radii = torch.as_tensor(mean_radii), torch.as_tensor(std_radii)

        ax_radii.plot(M_options_radii, mean_radii, label=rf"${convert_to_sci_notation(N)}$", color=color, marker="o")
        ax_radii.fill_between(M_options_radii, mean_radii - std_radii, mean_radii + std_radii, color=color, alpha=0.2)

    ax_radii.set_ylim(bottom=0)

    if M_setting == 'small':
        ax_radii.set_ylim(top=8)
    elif M_setting == 'large':
        ax_radii.set_ylim(top=200)
    else:
        raise ValueError(f"Unknown M_setting: {M_setting}")

    ax_radii.set_xlabel(r"Support size $M$")
    ax_radii.set_ylabel("Time [s]")
    ax_radii.grid(True, linestyle="--", alpha=0.4)
    ax_radii.legend(title=r"$N$", loc="best")
    fig_radii.tight_layout()


    # partition_times = load_list_of_time_logger_partition(args, random_seed_options)

    # fig_partition, ax_partition = plt.subplots(figsize=(6, 4))

    # N_train = 5000
    # partition_times_slice = partition_times._slice(N_train=N_train)
    # M_options_partition = torch.as_tensor([key[1] for key in partition_times_slice.keys()])
    # idx_partition = M_options_partition.argsort()

    # mean_partition,  std_partition = partition_times_slice.mean_time[idx_partition], partition_times_slice.std_time[idx_partition]

    # ax_partition.plot(M_options_partition[idx_partition], mean_partition, label=rf"${convert_to_sci_notation(N_train)}$", color='black', marker="o")
    # ax_partition.fill_between(M_options_partition[idx_partition], mean_partition - std_partition, mean_partition + std_partition, color='black', alpha=0.2)

    # ax_partition.set_ylim(bottom=0)
    # ax_partition.set_xlabel(r"Support size $M$")
    # ax_partition.set_ylabel("Time [s]")
    # ax_partition.grid(True, linestyle="--", alpha=0.4)
    # # ax_partition.legend(title=r"$N_{train}$", loc="best")
    # fig_partition.tight_layout()


    if args.save:
        file_name = f"time_radii_W{args.wasserstein_order}_{args.distribution.lower()}_dims_{args.num_dims}_setting_{args.setting}_{args.method}_{M_setting}"
        folder = os.path.dirname(os.path.dirname(args.figures_dir)) # USE figures_dir! results_dir is solely for data
        plt.savefig(os.path.join(folder, f"{file_name}.pdf"))  
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_arguments(
        distribution="GaussianMixture",
        num_dims=3,
        setting=0,
        wasserstein_order=2,
        beta=1e-6,
        method='triangle_inequality_vertex', 
        save=True,
    )

    settings = [
        ("Gaussian", 100, 'joint_diagonal_milp', 'small'),
        ("Gaussian", 100, 'joint_diagonal_milp', 'large'),
        ("Gaussian", 100, 'triangle_inequality_vertex', 'small'),
        ("Gaussian", 100, 'triangle_inequality_vertex', 'large'),
    ]

    for distribution, num_dims, method, M_setting in settings:
        args.method = method
        args.distribution = distribution
        args.num_dims = num_dims
        args = process_args(args)
        main(args, M_setting)
