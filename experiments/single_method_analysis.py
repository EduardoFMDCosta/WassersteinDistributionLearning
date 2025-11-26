import os
import itertools
import matplotlib.pyplot as plt
import torch

from configs.handlers import parse_arguments, load_json, process_args
from experiments.utils import load_list_of_data_driven_radii


def main(args, M_options, N_options, random_seed_options = [0]):
    combinations = [(N, M) for N in N_options for M in M_options]

    data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
    
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(N_options) - 1)) for i in range(len(N_options))]
    for N, color in zip(N_options, colors):
        data_slice = data._slice(N_train=args.num_samples_training, N=N)
        M_options_plot = torch.as_tensor([key[2] for key in data_slice.keys()])
        idx = M_options_plot.argsort()

        ax.plot(M_options_plot[idx], data_slice.mean_radius[idx], label=str(N), color=color, marker="o")
        ax.fill_between(
            M_options_plot[idx],
            data_slice.mean_radius[idx] - data_slice.std_radius[idx],
            data_slice.mean_radius[idx] + data_slice.std_radius[idx],
            color=color,
            alpha=0.2,
        )

    ax.set_title(f"{args.num_dims}D {args.distribution} (setting={args.setting}) - {args.method}")
    ax.set_xlabel(f"Number of clusters (M)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=f"N", loc="upper right")

    if args.save:
        file_name = f"W{args.wasserstein_order}_N_train={args.num_samples_training}_method={args.method}"
        plt.savefig(os.path.join(args.figures_dir, f"{file_name}.png"))
        plt.close("all")


if __name__ == '__main__':
    args = parse_arguments( # Only parse arguments once, updated afterwards
        random_seed=0,
        distribution="Gaussian", # PLACEHOLDER
        num_dims=2, # PLACEHOLDER
        setting=0,  # PLACEHOLDER
        num_samples_training=5_000,
        method='diagonal_constrained_tp',
        save=True,
    )
    
    M_options = [5, 20, 30, 40, 50, 75]
    N_options = [1000, 2500, 5000, 7500, 10000, 25000, 50000, 100000, 500000, 1000000]
    random_seed_options = [0]

    params = load_json("parameters")
    settings = [(d, int(n), int(s)) for d in params.keys() for n in params[d]["num_dims"].keys() for s in params[d]["num_dims"][n]["settings"].keys()]

    # settings = [('Uniform', 2, 0)]  # TEMPORARY LIMITATION FOR DEBUGGING
    for (distribution, num_dims, setting), wasserstein_order in itertools.product(settings, [1,2]):
        args.distribution = distribution
        args.num_dims = num_dims
        args.setting = setting
        args.wasserstein_order = wasserstein_order
        args = process_args(args)

        try:
            main(args, M_options=M_options, N_options=N_options, random_seed_options=random_seed_options)
        except Exception as e:
            print(f"Failed for distribution={distribution}, num_dims={num_dims}, setting={setting} with error: {e}")

    if not args.save:
        plt.show()