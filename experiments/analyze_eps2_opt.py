import torch
from matplotlib import pyplot as plt
import seaborn as sns

from configs.construct import get_support_assumption, get_distribution
from configs.handlers import parse_arguments
from optimization import ot_lp_solver, cutting_plane, full_search, diagonal_constrained_tp, max_oracle_gradient_descent, black_box
from plotting.plot import plot_quantization
from quantization import UncertainQuantization
from sets import BoundedVoronoiPartition

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
        distribution="Gaussian",
        dimension=2,
        setting=0,
        num_samples_training=1000,
        num_samples=1000,
        num_clusters=num_clusters,
        beta=1e-6,
        plot=False
    )

    # Set parameters
    N_training = args.num_samples_training
    M = args.num_clusters
    N = args.num_samples
    beta = args.beta
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

    # Get variables
    cost = quantization.partition.distance_locs ** 2
    lower = quantization.lower_probs
    upper = quantization.upper_probs
    empirical_marginal = quantization.probs

    store = {}

    # Cutting plane LP solver
    result = cutting_plane(cost=cost,
                           lower=lower,
                           upper=upper,
                           empirical_marginal=empirical_marginal,
                           num_steps=1000,
                           ot_solver=ot_lp_solver)

    store["Stochastic Vertice Ascent"] = {
        "w_opt": result['w_opt'],
        "objective_opt": result['objective_opt']
    }

    result = full_search(cost=cost,
                         lower=lower,
                         upper=upper,
                         empirical_marginal=empirical_marginal,
                         ot_solver=ot_lp_solver)

    store["Searching Vertices"] = {
        "w_opt": result['w_opt'],
        "objective_opt": result['objective_opt']
    }

    result = black_box(cost=cost,
                       lower=lower,
                       upper=upper,
                       empirical_marginal=empirical_marginal)

    store["Black Box (Gurobi)"] = {
        "w_opt": result['w_opt'],
        "objective_opt": result['objective_opt']
    }

    result = diagonal_constrained_tp(cost=cost,
                                     lower=lower,
                                     upper=upper,
                                     empirical_marginal=empirical_marginal)

    store["Diagonally Constrained TP"] = {
        "w_opt": result['w_opt'],
        "objective_opt": result['objective_opt']
    }

    max_value = 0
    for i in range(10):
        result = max_oracle_gradient_descent(cost=cost,
                                             lower=lower,
                                             upper=upper,
                                             empirical_marginal=empirical_marginal,
                                             plot=False)

        if result['objective_opt'] > max_value:
            max_value = result['objective_opt']
            w_opt = result['w_opt']

    store["Max Oracle Gradient Descent"] = {
        "w_opt": result['w_opt'],
        "objective_opt": result['objective_opt']
    }

    return store

if __name__ == '__main__':
    torch.manual_seed(0)

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
            data[method_name]['objective_opt'].append(results["objective_opt"])

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
