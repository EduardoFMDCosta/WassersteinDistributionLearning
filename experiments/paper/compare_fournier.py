import torch
from sets import BoundedVoronoiPartition
from quantization import UncertainQuantization
from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from bound import DataDrivenRadius, fournier_radius
from configs.construct import get_support_assumption, get_distribution

import matplotlib.pyplot as plt
import seaborn as sns

colors = [
        "lightcoral",
        "olive",
        "mediumseagreen",
        "deepskyblue",
        "orchid"
    ]

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def compare_fournier(distribution, dimension, num_samples, num_clusters):
    args = parse_arguments(
        distribution=distribution,
        dimension=dimension,
        setting=0,
        num_samples_training=1000,
        num_samples=num_samples,
        num_clusters=num_clusters,
        beta=1e-4,
        plot=False
    )

    # Set parameters
    N_training = args.num_samples_training
    M = args.num_clusters
    N = args.num_samples
    beta = args.beta
    method = 'cutting_plane'
    support_assumption = get_support_assumption(**vars(args))

    # (Unknown) Generating probability
    distribution = get_distribution(**vars(args))

    # Generate Partitions
    samples_partition = distribution.sample((N_training,))
    partition = BoundedVoronoiPartition(
        support=support_assumption,
        samples=samples_partition,
        M=M,
        use_voronoi_radii=False # set to false to speed up
    )

    # Generate Quantization
    samples_quantization = distribution.sample((N,))
    quantization = UncertainQuantization(partition=partition, samples=samples_quantization, beta=beta)

    # Plot samples and clusterized distribution
    if args.plot:
        plot_quantization(quantization=quantization, title=f"M={M}, N={N}")

    # Compute bounds
    fournier_bound = fournier_radius(support=partition.support, nsamples=N, beta=beta)
    equiv_fournier_bound = fournier_radius(support=partition.support, nsamples=M, beta=beta)
    data_driven_output = DataDrivenRadius(quantization=quantization, method=method)

    return fournier_bound, equiv_fournier_bound, data_driven_output

if __name__ == '__main__':
    torch.manual_seed(0)

    results = {}

    distribution = "Gaussian"
    dimensions = [2, 3, 9]
    nums_samples = [1000, 5000, 10000]
    nums_clusters = [10, 50, 500]

    for num_samples in nums_samples:
        for num_clusters in nums_clusters:
            for dimension in dimensions:
                fournier_bound, equiv_fournier_bound, data_driven_output = compare_fournier(distribution, dimension, num_samples, num_clusters)

                results.setdefault(num_samples, {}).setdefault(num_clusters, {})[dimension] = {
                    "fournier_bound": fournier_bound,
                    "equiv_fournier_bound": equiv_fournier_bound,
                    "data_driven_output": data_driven_output.radius.item(),
                }

    num_samples_list = sorted(results.keys())
    # collect all dimensions across the dataset
    all_dims = sorted({dim for d in results.values() for M in d.values() for dim in M.keys()})

    sns.set_style("darkgrid")
    fig, axes = plt.subplots(len(num_samples_list), len(all_dims),
                             figsize=(5 * len(all_dims), 4 * len(num_samples_list)),
                             sharex=True, sharey=False)

    # ensure 2D array of axes
    if len(num_samples_list) == 1:
        axes = [axes]
    if len(all_dims) == 1:
        axes = [[ax] for ax in axes]

    for row, num_samples in enumerate(num_samples_list):
        data = results[num_samples]
        num_clusters_list = sorted(data.keys())

        for col, dim in enumerate(all_dims):
            dd_vals = [data[M][dim]['data_driven_output'] for M in num_clusters_list]
            fb_vals = [data[M][dim]['fournier_bound'] for M in num_clusters_list]
            efb_vals = [data[M][dim]['equiv_fournier_bound'] for M in num_clusters_list]

            ax = axes[row][col]
            ax.plot(num_clusters_list, dd_vals, label=r"Ours")
            ax.plot(num_clusters_list, fb_vals, label=rf"Fournier ($N$)")
            ax.plot(num_clusters_list, efb_vals, label=rf"Fournier ($M$)")

            ax.set_ylim(0.0, 3.5)

            if row == 0:
                ax.set_title(f"{distribution} distribution in {dim}d")
            if col == 0:
                ax.set_ylabel(rf"Number of samples $N = {num_samples}$")
            if row == len(num_samples_list) - 1:
                ax.set_xlabel(rf"Number of clusters ($M$)")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)

    plt.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    plt.show()



