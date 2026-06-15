import os
import itertools
import matplotlib.pyplot as plt
import torch

from configs.handlers import parse_arguments, load_json, process_args, num_samples_training_from_num_samples
from experiments.utils import load_list_of_data_driven_radii


def main(args, M_options, N_options, random_seed_options = [0]):
    combinations = [(num_samples_training_from_num_samples(N), N, M) for N in N_options for M in M_options]

    data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
    
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(N_options) - 1)) for i in range(len(N_options))]
    for N, color in zip(N_options, colors):
        data_slice = data._slice(N=N)
        if data_slice.keys() == []:
            continue

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
        file_name = f"W{args.wasserstein_order}_method={args.method}"
        plt.savefig(os.path.join(args.figures_dir, f"{file_name}.png"))
        plt.close("all")
    else:
        plt.show()


if __name__ == '__main__':
    args = parse_arguments( # Only parse arguments once, updated afterwards
        distribution="TruncatedGaussian", # PLACEHOLDER
        num_dims=2, # PLACEHOLDER
        setting=0,  # PLACEHOLDER
        method='joint_diagonal_milp',
        save=False,
    )
    
    M_options = [5, 20, 30, 40, 50, 75]
    N_options = [1000, 2500, 5000, 7500, 10000, 100000, 1000000]
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # [0, 1, 2, 3]

    params = load_json("parameters")
    settings = [(d, int(n), int(s)) for d in params.keys() for n in params[d]["num_dims"].keys() for s in params[d]["num_dims"][n]["settings"].keys()]
    settings = [elem for elem in settings if elem[0] == 'TruncatedGaussian']

    settings = [('Uniform', 2, 0)]  # TEMPORARY LIMITATION FOR DEBUGGING
    for (distribution, num_dims, setting), wasserstein_order in itertools.product(settings, [1]):
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