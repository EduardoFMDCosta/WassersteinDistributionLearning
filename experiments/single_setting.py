import torch

from sets import BoundedVoronoiPartition
from quantization import UncertainQuantization
from bound import DataDrivenRadius, fournier_radius
from solvers import get_solver

from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from configs.construct import get_support_assumption, get_distribution


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_samples_training=1000,
        num_clusters=5,
        beta=1e-4,
        method='diagonal_constrained_tp',
        plot=False, 
        compute_discrete_bound=False, 
        compute_moment_bound=True
    )

    solver = get_solver(method=args.method)

    if not args.compute_discrete_bound:
        solver.disable_discrete_bound_computation()
    if not args.compute_moment_bound:
        solver.disable_moment_bound_computation()

    support_assumption = get_support_assumption(**vars(args))

    # (Unknown) Generating probability
    distribution = get_distribution(**vars(args))

    # Generate Partitioning
    samples_partition = distribution.sample((args.num_samples_training,))
    partition = BoundedVoronoiPartition(
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

    # Compute bounds
    fournier_bound = fournier_radius(
        support=partition.support, 
        nsamples=args.num_samples + args.num_samples_training, 
        beta=args.beta
    )
    data_driven_output = DataDrivenRadius(quantization=quantization, solver=solver)

    print(f"Number of clusters (M) / num_samples (N): {args.num_clusters} / {args.num_samples} \n"
          f"\t Fournier: {fournier_bound:.4f} \n"
          f"\t Ours : {data_driven_output.radius:.4f} \n"
        )

    print("Process finished.")

