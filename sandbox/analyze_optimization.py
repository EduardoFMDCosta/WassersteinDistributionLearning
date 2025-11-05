import torch
from matplotlib import pyplot as plt

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

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        num_samples_training=1000,
        num_samples=1000,
        num_clusters=10,
        beta=1e-4,
        method='max_oracle_gradient_descent',
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
        M=args.num_clusters
    )

    # Generate Quantization
    samples_quantization = distribution.sample((args.num_samples,))
    quantization = UncertainQuantization(partition=partition, samples=samples_quantization, beta=args.beta)

    # Plot samples and clusterized distribution
    if args.plot:
        plot_quantization(quantization=quantization, title=f"M={args.num_clusters}, N={args.num_samples}")


    # Analysis 3: Compute optima for different starting points (if applicable)
    for name, Solver in get_discrete_solver.mapping.items():
        if name == 'max_oracle_gradient_descent':
            num_iters = 10
        else:
            num_iters = 1

        for _ in range(num_iters):
            result = Solver().solve(
                cost=quantization.l2_distance_locs_to_locs ** 2,
                lower=quantization.lower_probs,
                upper=quantization.upper_probs,
                empirical_marginal=quantization.probs
            )

            print(f"Final w ({name}) = {result.w_opt}")
            print(f"Value ({name}) = {result.bound} \n")
