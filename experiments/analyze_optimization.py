import torch
import numpy as np
from matplotlib import pyplot as plt

from configs.construct import get_support_assumption, get_distribution
from configs.handlers import parse_arguments
from optimization import ot_lp_solver, get_omega_space_vertices, cutting_plane, full_search, plain_vanilla, \
    diagonal_constrained_tp, max_oracle_gradient_descent, black_box
from plotting.plot import plot_quantization
from quantization import UncertainQuantization
from sets import BoundedVoronoiPartition

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        dimension=2,
        setting=0,
        num_samples_training=1000,
        num_samples=1000,
        num_clusters=10,
        beta=1e-4,
        plot=False
    )

    # Set parameters
    N_training = args.num_samples_training
    M = args.num_clusters
    N = args.num_samples
    beta = args.beta
    method = 'max_oracle_gradient_descent'
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


    # Analysis 3: Compute optima for different starting points (if applicable)
    # Cutting plane LP solver
    result = cutting_plane(cost=cost,
                           lower=lower,
                           upper=upper,
                           empirical_marginal=empirical_marginal,
                           num_steps=1000,
                           ot_solver=ot_lp_solver)

    print(f"Final w (Cutting plane) = {result['w_opt']}")
    print(f"Value (Cutting plane) = {result['objective_opt']} \n")

    result = full_search(cost=cost,
                         lower=lower,
                         upper=upper,
                         empirical_marginal=empirical_marginal,
                         ot_solver=ot_lp_solver)

    print(f"Final w (Full search) = {result['w_opt']}")
    print(f"Value (Full search) = {result['objective_opt']} \n")

    result = black_box(cost=cost,
                       lower=lower,
                       upper=upper,
                       empirical_marginal=empirical_marginal)

    print(f"Final w (Black box) = {result['w_opt']}")
    print(f"Value (Black box) = {result['objective_opt']} \n")

    # result = plain_vanilla(cost=cost,
    #                        lower=lower,
    #                        upper=upper,
    #                        empirical_marginal=empirical_marginal)
    #
    # print(f"Final w (Plain vanilla) = {result['w_opt']}")
    # print(f"Value (Plain vanilla) = {result['objective_opt']} \n")
    #
    # result = diagonal_constrained_tp(cost=cost,
    #                                  lower=lower,
    #                                  upper=upper,
    #                                  empirical_marginal=empirical_marginal)
    #
    # print(f"Final w (Fixate TP) = {result['w_opt']}")
    # print(f"Value (Fixate TP) = {result['objective_opt']} \n")

    for i in range(10):
        result = max_oracle_gradient_descent(cost=cost,
                                       lower=lower,
                                       upper=upper,
                                       empirical_marginal=empirical_marginal)

        print(f"Final w (Oracle) = {result['w_opt']}")
        print(f"Value (Oracle) = {result['objective_opt']} \n")