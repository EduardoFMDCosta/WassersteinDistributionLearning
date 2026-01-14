import os
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from configs.handlers import parse_arguments, num_samples_training_from_num_samples, process_args
from experiments.utils import load_list_of_data_driven_radii, fournier_radii_for_combinations
from experiments.generate_samples import SIZE

from plotting.utils_plot import set_style, convert_to_sci_notation

set_style()


def inflection_analysis(args) -> dict:
    # N = 10_000
    N = args.num_samples

    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500]
    if args.method != 'joint_diagonal_milp':
        M_options += [1000]
    
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    combinations = [(num_samples_training_from_num_samples(N), N, M) for M in M_options]

    data = load_list_of_data_driven_radii(args, combinations, random_seed_options)
    M_options_plot = torch.as_tensor([key[2] for key in data.keys()])
    idx = M_options_plot.argsort()

    fournier_data = fournier_radii_for_combinations(args, combinations=[(args.num_samples, int(M)) for M in M_options_plot])
    M_options_fornier_plots = torch.as_tensor([key[2] for key in fournier_data.keys(N=args.num_samples, N_train=args.num_samples_training)])
    idx_fournier = M_options_fornier_plots.argsort()

    if data.keys() == []:
        return dict(mean=torch.nan, std=torch.nan, M=torch.nan, fournier=torch.nan)

    color = (0.2298057, 0.298717966, 0.753683153)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(M_options_plot[idx], data.mean_radius[idx], label=rf"${convert_to_sci_notation(N)}$", color=color, marker="o")
    ax.fill_between(
        M_options_plot[idx],
        data.mean_radius[idx] - data.std_radius[idx],
        data.mean_radius[idx] + data.std_radius[idx],
        color=color,
        alpha=0.2,
    )

    ax.plot(M_options_fornier_plots[idx_fournier], fournier_data.radius[idx_fournier], color='grey', linestyle="--")

    our_legend = Line2D([], [], color=color, linestyle="-", label=rf"Ours")
    fournier_legend = Line2D([], [], color="grey", linestyle="--", label=rf"Fournier [13]")
    ax.legend(
        handles=[our_legend, fournier_legend],
        loc="best",
        bbox_transform=fig.transFigure,
        frameon=False
    )

    plt.tight_layout()
    fig.subplots_adjust(right=0.7)

    if args.save:
        file_name = f"inflection_incl_fournier_W{args.wasserstein_order}_{args.distribution.lower()}_dims_{args.num_dims}_setting_{args.setting}_{args.method}"
        folder = os.path.dirname(os.path.dirname(args.figures_dir))
        plt.savefig(os.path.join(folder, f"{file_name}.pdf"))  
    else:
        plt.show()

    idx_best = data.mean_radius[idx].argmin()

    return dict(
        mean=data.mean_radius[idx][idx_best].item(),
        std=data.std_radius[idx][idx_best].item(),
        M=M_options_plot[idx][idx_best].item(),
        fournier=fournier_data.radius[idx_fournier][idx_best].item()
    )


def main(args) -> dict:
    N = args.num_samples
    M_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    random_seed_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    combinations = [(num_samples_training_from_num_samples(N), N, M) for M in M_options]

    # joint_diagonal_milp
    args.method = 'joint_diagonal_milp'
    args = process_args(args)
    data_milp = load_list_of_data_driven_radii(args, combinations, random_seed_options)
    M_options_milp = torch.as_tensor([key[2] for key in data_milp.keys()])
    idx_milp = M_options_milp.argsort()

    # triangle_inequality_vertex
    args.method = 'triangle_inequality_vertex'
    args = process_args(args)
    data_triangle = load_list_of_data_driven_radii(args, combinations, random_seed_options)
    data_triangle = data_triangle._slice(M=M_options_milp.tolist())
    M_options_triangle = torch.as_tensor([key[2] for key in data_triangle.keys()])
    idx_triangle = M_options_triangle.argsort()

    if not torch.equal(M_options_triangle[idx_triangle], M_options_milp[idx_milp]):
        raise ValueError

    fournier_data = fournier_radii_for_combinations(args, combinations=[(args.num_samples, int(M)) for M in M_options_milp])
    M_options_fournier = torch.as_tensor([key[2] for key in fournier_data.keys(N=args.num_samples, N_train=args.num_samples_training)])
    idx_fournier = M_options_fournier.argsort()    

    idx_best = data_milp.mean_radius[idx_milp].argmin()

    return dict(
        mean_milp=data_milp.mean_radius[idx_milp][idx_best].item(),
        std_milp=data_milp.std_radius[idx_milp][idx_best].item(),
        mean_triangle=data_triangle.mean_radius[idx_triangle][idx_best].item(),
        std_triangle=data_triangle.std_radius[idx_triangle][idx_best].item(),
        M=M_options_milp[idx_milp][idx_best].item(),
        fournier=fournier_data.radius[idx_fournier][idx_best].item()
    )


if __name__ == '__main__':
    args = parse_arguments(
        distribution="UCI-Turbine",  # PLACEHOLDER
        num_dims=11,  # PLACEHOLDER
        setting=0,
        wasserstein_order=2,
        num_samples=10_000,
        beta=1e-6,
        method='joint_diagonal_milp', # PLACEHOLDER
        plot=True,
        save=True,
    )

    settings = [
        ("UCI-Turbine", 11),
        ("UCI-MiniBooNE", 50),
        ("OCTMNIST", 784),
    ]

    rows = []
    for distribution, num_dims in settings:
        args.distribution = distribution
        args.num_samples = SIZE[distribution] - args.num_samples_training
        args.num_dims = num_dims

        args = process_args(args)
        row = main(args)
        row["distribution"] = distribution
        row["num_dims"] = num_dims
        row["N"] = args.num_samples
        row["Ntrain"] = args.num_samples_training
        rows.append(row)

    df = pd.DataFrame(rows)
    if args.save:
        df.to_excel(os.path.join(args.tables_dir, "uci_mnist.xlsx"), index=False)
    else:
        print(df)
    

