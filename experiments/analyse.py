import os
import torch
import matplotlib.pyplot as plt
from typing import Optional

from configs.handlers import parse_arguments
from experiments.datastructures import Quantizations, DataDrivenRadii, FournierRadii, EmpiricalRadii
import plotting.plot as plot
from experiments.utils import load_data

def plot_slice(
    args,
    data_driven_radii: DataDrivenRadii, 
    quantizations: Quantizations, 
    num_samples: Optional[int] = None,
    num_clusters: Optional[int] = None,
):
    if (num_samples is None and num_clusters is None) or (num_samples is not None and num_clusters is not None):
        raise ValueError("Either num_samples or num_clusters must be provided.")

    fig, ax = plt.subplots(5, 1, figsize=(6, 12), constrained_layout=True)

    ax[0] = plot.plot_data_driven_radii_slice(ax[0], data_driven_radii, N=num_samples, M=num_clusters, cummulative=True)
    ax[1] = plot.plot_quantization_slice(ax[1], quantizations, stat='probs', N=num_samples, M=num_clusters)
    ax[2] = plot.plot_quantization_slice(ax[2], quantizations, stat='radii', N=num_samples, M=num_clusters)
    ax[3] = plot.plot_quantization_slice(ax[3], quantizations, stat='counts', N=num_samples, M=num_clusters)
    ax[4] = plot.plot_quantization_slice(ax[4], quantizations, stat='locs', N=num_samples, M=num_clusters)

    ax[0].set_title(f"Number of {'samples (N)' if num_samples is not None else 'clusters (M)'} = {num_samples if num_samples is not None else num_clusters}")
    ax[4].set_xlabel(f"Number of {'clusters (M)' if num_clusters is None else 'samples (N)'}")

    tag = f"slice_at_{'num_samples' if num_clusters is None else 'num_clusters'}={num_samples if num_samples is not None else num_clusters}"
    if args.save:
        plt.savefig(os.path.join(args.figures_dir, f"{tag}.png"))  # TODO save in pdf for paper
    else:
        plt.show()
        
def plot_grid(args, data_driven_radii: DataDrivenRadii):
    xlabel, ylabel = "Number of samples (N)", "Number of clusters (M)"

    fig, ax = plt.subplots(3, 1, figsize=(6, 12), constrained_layout=True)
    ax[0] = plot.plot_data_driven_radii(ax[0], data_driven_radii, field='moment_bound', ylabel=ylabel, title="Bound on Moment-Term (e1)")
    ax[1] = plot.plot_data_driven_radii(ax[1], data_driven_radii, field='discrete_bound', ylabel=ylabel, title="Bound on Discrete-Term Bound (e2)")
    ax[2] = plot.plot_data_driven_radii(ax[2], data_driven_radii, field='radius', xlabel=xlabel, ylabel=ylabel, title="Data-Driven Bound")

    if args.save:
        plt.savefig(os.path.join(args.figures_dir, f"grid.png"))  # TODO save in pdf for paper
    else:
        plt.show()

    # # Plot Computation Times
    # fig, ax = plt.subplots(3, 1, figsize=(6, 12), constrained_layout=True)
    # ax[0] = plot.plot_time_logger(ax[0], quantization_times, ylabel=ylabel, title="Quantization time")
    # ax[1] = plot.plot_time_logger(ax[1], radius_computation_times, ylabel=ylabel, title="Radius computation time")
    # ax[2] = plot.plot_time_logger(ax[2], computation_times, xlabel=xlabel, ylabel=ylabel, title="Total computation time")
    # plt.show()

def plot_quantization(args, quantizations: Quantizations):
    if args.num_dims == 2:
        fig, ax = plt.subplots(ncols=len(quantizations.keys()), nrows=1, figsize=(6 * len(quantizations.keys()), 6))

        for i, key in enumerate(quantizations.keys()):
            ax[i] = plot.plot_quantization(ax=ax[i], quantization=quantizations.at(key), title=f"M={key[1]}, N={key[0]}")

        if args.save:
            plt.savefig(os.path.join(args.figures_dir, f"quantizations.png"))
        else:
            plt.show()



if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Uniform",
        num_dims=2,
        setting=0,
        beta=1e-6,
        wasserstein_order=1,
        method='diagonal_constrained_tp',
        save=False,
    )

    data_driven_radii = load_data(args.data_driven_radii_file, DataDrivenRadii)
    fournier_radii = load_data(args.fournier_radii_file, FournierRadii)
    empirical_radii = load_data(args.empirical_radii_file, EmpiricalRadii)

    # Plot slice of statistics
    # plot_slice(args, data_driven_radii, quantizations, num_samples=1000)
    # plot_slice(args, data_driven_radii, quantizations, num_clusters=10)

    # # Plot grid
    # plot_grid(args, data_driven_radii)

    # # Plot Quantizations
    # plot_quantization(args, quantizations)
