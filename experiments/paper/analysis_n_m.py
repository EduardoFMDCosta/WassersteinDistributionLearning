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

def analysis_n_m(distribution, dimension, num_samples, num_clusters):
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
    data_driven_output = DataDrivenRadius(quantization=quantization, method=method)

    return data_driven_output

if __name__ == '__main__':
    torch.manual_seed(0)

    results = {}

    distribution = "Gaussian"
    dimension = 2
    nums_samples = [1000, 1500]

    options = {
        r"$M=20$": lambda n: 20,
        r"$M=50$": lambda n: 50,
        r"$M=N^{0.5}$": lambda n: int(n ** 0.5),
        #r"$M=N^{0.8}$": lambda n: int(n ** 0.8),
    }

    # Prepare storage for all options
    eps1_data = {opt: [] for opt in options}
    eps2_data = {opt: [] for opt in options}

    # Compute bounds for all num_samples
    for num_samples in nums_samples:
        for opt_name, clusters_fn in options.items():
            num_clusters = clusters_fn(num_samples)
            bound = analysis_n_m(distribution, dimension, num_samples, num_clusters)
            eps1_data[opt_name].append(bound.moment_bound.item())
            eps2_data[opt_name].append(bound.discrete_bound)

    # Plot side-by-side
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    lines = []
    labels = []

    # eps1
    for opt_name, values in eps1_data.items():
        line, = axes[0].plot(nums_samples, values)
        lines.append(line)
        labels.append(opt_name)
    axes[0].set_title(r'$\epsilon_1(D_N)$')
    axes[0].set_xlabel('$N$')
    axes[0].set_ylabel('Value')
    axes[0].grid(True)

    # eps2
    for opt_name, values in eps2_data.items():
        line, = axes[1].plot(nums_samples, values)
    axes[1].set_title(r'$\epsilon_2(D_N)$')
    axes[1].set_xlabel(r'$N$')
    axes[1].grid(True)

    # Single legend below figure
    fig.legend(lines, labels, loc='lower center', ncol=len(options), bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0.0, 0.15, 1.0, 1.0])
    plt.show()



