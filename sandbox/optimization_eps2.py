import torch
from matplotlib import pyplot as plt
import seaborn as sns

from solvers import get_discrete_solver
from quantization import UncertainQuantization
from sets import BoundedVoronoiPartition

from configs.construct import get_support_assumption, get_distribution
from configs.handlers import parse_arguments
from plotting.plot import plot_quantization

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

"""
This file plots the lower bound for epsilon_2 computed by each optimization method.
It shows that...
"""

def lower_bound_eps2(num_clusters):
    args = parse_arguments(
        random_seed=0,
        distribution="TruncatedGaussian",
        num_dims=2,
        setting=0,
        num_samples_training=1000,
        num_samples=1000,
        num_clusters=num_clusters,
        beta=1e-6,
        plot=False
    )

    support_assumption = get_support_assumption(**vars(args))

    # (Unknown) Generating probability
    distribution = get_distribution(**vars(args))

    # Generate Partitions
    samples_partition = distribution.sample((args.num_samples_training,))
    partition = BoundedVoronoiPartition.from_samples(
        support=support_assumption,
        samples=samples_partition,
        M=args.num_clusters,
    )

    # Generate Quantization
    samples_quantization = distribution.sample((args.num_samples,))
    quantization = UncertainQuantization(partition=partition, samples=samples_quantization, beta=args.beta)

    # Plot samples and clusterized distribution
    if args.plot:
        plot_quantization(quantization=quantization, title=f"M={args.num_clusters}, N={args.num_samples}")

    # Get variables
    store = dict()
    for name, Solver in get_discrete_solver.mapping.items():
        if name == 'max_oracle_gradient_descent':
            num_iters = 10
        else:
            num_iters = 1

        for i in range(num_iters):
            result = Solver().solve(
                cost=quantization.l2_distance_locs_to_locs ** 2,
                lower=quantization.lower_probs,
                upper=quantization.upper_probs,
                empirical_marginal=quantization.probs
            )
            if i == 0:
                store[name] = result
            else:
                if result.bound > store[name].objective_opt:
                    store[name] = result

    return store

if __name__ == '__main__':
    store = {}
    nums_clusters = [5, 10, 20, 30, 50, 100]

    for num_clusters in nums_clusters:
        store[num_clusters] = lower_bound_eps2(num_clusters)

    # Plot
    methods = list(next(iter(store.values())).keys())  # get method names from the first M
    data = {method: {'M': [], 'objective_opt': []} for method in methods}

    palette = sns.color_palette("colorblind", n_colors=len(data))
    method_colors = {method_name: palette[i] for i, method_name in enumerate(data.keys())}

    for M, method_dict in store.items():
        for method_name, results in method_dict.items():
            data[method_name]['M'].append(M)
            data[method_name]['objective_opt'].append(results.objective_opt)

    # Plot
    plt.figure(figsize=(8, 6))
    for method_name, values in data.items():
        plt.scatter(values['M'], values['objective_opt'], label=rf"{method_name}", color=method_colors[method_name], s=20)

    plt.xlabel(r"Number of clusters $M$")
    plt.xticks(nums_clusters)
    plt.ylabel(r"Lower bound for $\epsilon_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
