import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import load_list_of_time_loggers
from experiments.partitions import load_list_of_time_logger_partition

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()

if __name__ == '__main__':
    args = parse_arguments(
        distribution="Gaussian",
        num_dims=10,
        setting=1,
        wasserstein_order=2,
        num_samples_training=5000,
        beta=1e-6,
        method='joint_diagonal_milp',
        save=False,
    )

    N_options = [1000]
    M_options = [5, 20]
    random_seed_options = [0, 1]

    combinations = [(N, M) for N in N_options for M in M_options]

    radii_times = load_list_of_time_loggers(args, random_seed_options)
    partition_times = load_list_of_time_logger_partition(args, random_seed_options)._slice(N_train=args.num_samples_training)

    fig_radii, ax_radii = plt.subplots(figsize=(6, 4))
    fig_partition, ax_partition = plt.subplots(figsize=(6, 4))

    cmap = plt.cm.coolwarm
    colors = [cmap(i / max(len(N_options) - 1, 1)) for i in range(len(N_options))]
    for N, color in zip(N_options, colors):
        radii_times_slice = radii_times._slice(N_train=args.num_samples_training, N=N)
        M_options_plot = torch.as_tensor([key[2] for key in radii_times_slice.keys()])
        idx = M_options_plot.argsort()

        partition_times_slice = partition_times._slice(M=M_options_plot.tolist())
        if not len(partition_times_slice.keys()) == len(M_options_plot):
            M_missing = set(M_options_plot.tolist()) - set([key[1] for key in partition_times_slice.keys()])
            raise ValueError(f"Partition times missing for M={M_missing}.")
    
        mean_radii,  std_radii = radii_times_slice.mean_time[idx], radii_times_slice.std_time[idx]
        mean_partition,  std_partition = partition_times_slice.mean_time[idx], partition_times_slice.std_time[idx]

        ax_radii.plot(M_options_plot[idx], mean_radii, label=rf"${convert_to_sci_notation(N)}$", color=color, marker="o")
        ax_radii.fill_between(M_options_plot[idx], mean_radii - std_radii, mean_radii + std_radii, color=color, alpha=0.2)

        ax_partition.plot(M_options_plot[idx], mean_partition, label=rf"${convert_to_sci_notation(N)}$", color=color, marker="o")
        ax_partition.fill_between(M_options_plot[idx], mean_partition - std_partition, mean_partition + std_partition, color=color, alpha=0.2)

    ax_radii.set_xlabel(r"Support size $M$")
    ax_radii.set_ylabel("Time [s]")
    ax_radii.grid(True, linestyle="--", alpha=0.4)
    ax_radii.legend(title=r"$N$", loc="best")
    fig_radii.tight_layout()

    ax_partition.set_xlabel(r"Support size $M$")
    ax_partition.set_ylabel("Time [s]")
    ax_partition.grid(True, linestyle="--", alpha=0.4)
    ax_partition.legend(title=r"$N$", loc="best")
    fig_partition.tight_layout()

    if args.save:
        file_name = f"time_W{args.wasserstein_order}_N_train={args.num_samples_training}_{args.method}"
        fig_radii.savefig(os.path.join(args.figures_dir, f"{file_name}_radii.pdf"))
        fig_partition.savefig(os.path.join(args.figures_dir, f"{file_name}_partition.pdf"))
        
    plt.show()
