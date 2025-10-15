import os
import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import run_combinations
import plotting.plot as plot


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_clusters=10,
        beta=1e-4,
        method='stochastic_vertice_ascent',
        plot=True, 
        save=False,
        compute_moment_bound=True,
        compute_discrete_bound=False,
    )
    results_dir = os.path.join(args.results_dir, args.distribution.lower())

    investigate_clusters = True

    # We assume num_samples_training = num_samples
    if investigate_clusters:
        N_options = [args.num_samples]
        M_options = [10, 25, 100]  # [10, 25, 75, 100, 200, 500, 1000]
    else:
        N_options = [1000, 2500] #  [1000, 2500, 5000, 7500, 10000]
        M_options = [args.num_clusters]

    (quantizations, data_driven_radii, fournier_radii), _ = run_combinations(args, M_options=M_options, N_options=N_options)

    # Illustrate Quantizations
    if args.num_dims == 2:
        fig, ax = plt.subplots(ncols=len(quantizations.keys()), nrows=1, figsize=(6 * len(quantizations.keys()), 6))
        for i, key in enumerate(quantizations.keys()):
            ax[i] = plot.plot_quantization(ax=ax[i], quantization=quantizations.at(key), title=f"M={key[1]}, N={key[0]}")

        if args.save:
            plt.savefig(os.path.join(results_dir, f"ndims={args.num_dims}_set={args.setting}_quantizations.png"))
        else:
            plt.show()

    # Plot Statistics
    fig, ax = plt.subplots(5, 1, figsize=(6, 12), constrained_layout=True)

    ax[0] = plot.plot_data_driven_radii_slice(ax[0], data_driven_radii, N=N_options[0], cummulative=True)
    ax[1] = plot.plot_quantization_slice(ax[1], quantizations, stat='probs', N=N_options[0])
    ax[2] = plot.plot_quantization_slice(ax[2], quantizations, stat='radii', N=N_options[0])
    ax[3] = plot.plot_quantization_slice(ax[3], quantizations, stat='counts', N=N_options[0])
    ax[4] = plot.plot_quantization_slice(ax[4], quantizations, stat='locs', N=N_options[0])

    ax[0].set_title(f"Number of {'samples (N)' if investigate_clusters else 'clusters (M)'} = {args.num_samples if investigate_clusters else args.num_clusters}")
    ax[4].set_xlabel(f"Number of {'clusters (M)' if investigate_clusters else 'samples (N)'}")

    tag = f"convergence_{args.distribution}_setting={args.setting}"
    if investigate_clusters:
        tag += f"_N={args.num_samples}_M={M_options}"
    else:
        tag += f"_N={N_options}_M={args.num_clusters}"

    if args.save:
        plt.savefig(os.path.join(results_dir, f"ndims={args.num_dims}_set={args.setting}_analysis.png"))
    else:
        plt.show()
        