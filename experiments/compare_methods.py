import os
import itertools
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments, load_json, process_args
from experiments.utils import data_driven_radii_for_combinations, fournier_radii_for_combinations


def main(args, M_options, N_options, method_to_compare):
    method = args.method
    combinations = [(N, M) for N in N_options for M in M_options]

    data, _ = data_driven_radii_for_combinations(args, combinations=combinations, generate_partition_if_missing=False, generate_data_driven_radii_if_not_stored=False)
    
    if method_to_compare == 'fournier':
        data_to_compare = fournier_radii_for_combinations(args, combinations)
    else:
        args.method = method_to_compare
        args = process_args(args)
        data_to_compare, _ = data_driven_radii_for_combinations(args, combinations=combinations, generate_partition_if_missing=False, generate_data_driven_radii_if_not_stored=False)
        args.method = method
        args = process_args(args)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(N_options) - 1)) for i in range(len(N_options))]
    for N, color in zip(N_options, colors):
        M_options_non_empty = [key[2] for key in data.keys(N=N, N_train=args.num_samples_training)]
        M_options_to_compare_non_empty = [key[2] for key in data_to_compare.keys(N=N, N_train=args.num_samples_training)]
        M_options_plot = sorted(list(set(M_options_non_empty) & set(M_options_to_compare_non_empty)))

        ratios = list()
        for M in M_options_plot:
            key = (args.num_samples_training, N, M)
            radius = data.data[key].radius
            radius_to_compare = data_to_compare.data[key]
            if method_to_compare != 'fournier':
                radius_to_compare = radius_to_compare.radius
            
            ratios.append(radius / radius_to_compare)

        ax.plot(M_options_plot, ratios, label=str(N), color=color, marker="o")

    ax.set_title(f"{args.num_dims}D {args.distribution} (setting={args.setting}) {args.method} / {method_to_compare}")
    ax.set_xlabel(f"Number of clusters (M)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=f"N", loc="upper right")

    if args.save:
        file_name = f"W{args.wasserstein_order}_N_train={args.num_samples_training}_seed={args.random_seed}_{args.method}_vs_{method_to_compare}"
        plt.savefig(os.path.join(args.figures_dir, f"{file_name}.png"))
        plt.close("all")


if __name__ == '__main__':
    args = parse_arguments( # Only parse arguments once, updated afterwards
        random_seed=0,
        distribution="Gaussian", # PLACEHOLDER
        num_dims=2, # PLACEHOLDER
        setting=0,  # PLACEHOLDER
        num_samples_training=1_000,
        method='joint_diagonal_milp',
        save=True,
    )

    # method_to_compare = 'diagonal_constrained_tp'
    # method_to_compare = 'triangle_inequality_vertex'
    # method_to_compare = 'fournier'
    method_to_compare = 'joint_optimization_milp'
    # method_to_compare = 'joint_full_expansion_milp'

    M_options = [5, 20, 30, 40, 50, 75]
    N_options = [1000, 2500, 5000, 7500, 10000, 25000, 50000, 100000, 500000, 1000000]

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
            main(args, M_options=M_options, N_options=N_options, method_to_compare=method_to_compare)
        except Exception as e:
            print(f"Failed for distribution={distribution}, num_dims={num_dims}, setting={setting} with error: {e}")

    if not args.save:
        plt.show()