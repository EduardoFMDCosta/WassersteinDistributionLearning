import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, num_samples_training_from_num_samples
from experiments.utils import load_list_of_time_loggers
from experiments.partitions import load_list_of_time_logger_partition

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()

if __name__ == '__main__':
    args = parse_arguments(
        distribution="GaussianMixture",
        num_dims=3,
        setting=0,
        wasserstein_order=2,
        beta=1e-6,
        method='triangle_inequality_vertex', # 'joint_diagonal_milp'  'triangle_inequality_vertex'
        save=True,
    )

    N_options = [1000, 5000, 10000, 100000, 1000000]
    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    cmap = plt.cm.coolwarm

    radii_times = load_list_of_time_loggers(args, random_seed_options)

    fig_radii, ax_radii = plt.subplots(figsize=(6, 4))

    colors = [cmap(i / max(len(N_options) - 1, 1)) for i in range(len(N_options))]
    for N, color in zip(N_options, colors):
        radii_times_slice = radii_times._slice(N_train=num_samples_training_from_num_samples(N), N=N)
        M_options_radii = torch.as_tensor([key[2] for key in radii_times_slice.keys()])
        idx_radii = M_options_radii.argsort()

        mean_radii,  std_radii = radii_times_slice.mean_time[idx_radii], radii_times_slice.std_time[idx_radii]

        ax_radii.plot(M_options_radii[idx_radii], mean_radii, label=rf"${convert_to_sci_notation(N)}$", color=color, marker="o")
        ax_radii.fill_between(M_options_radii[idx_radii], mean_radii - std_radii, mean_radii + std_radii, color=color, alpha=0.2)

    ax_radii.set_ylim(bottom=0)
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
        # fig_partition.savefig(os.path.join(args.figures_dir, f"time_W{args.wasserstein_order}_partition.pdf"))
        fig_radii.savefig(os.path.join(args.figures_dir, f"time_W{args.wasserstein_order}_{args.method}_radii.pdf"))
    else:
        plt.show()