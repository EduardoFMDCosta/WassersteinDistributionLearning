import torch
import matplotlib.pyplot as plt

from sets import BoundedVoronoiPartition
from quantization import UncertainQuantization
from bound import DataDrivenRadius, FournierRadius
from solvers import get_solver

from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from configs.construct import get_support_assumption, get_distribution
from experiments.partitions import get_partition

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=3,
        setting=0,
        num_samples=10000,
        num_samples_training=1000,
        num_clusters=100,
        wasserstein_order=2,
        beta=1e-4,
        method='triangle_inequality_vertex',
        plot=False,
        save=False,
        compute_discrete_bound=False, 
        compute_moment_bound=True
    )

    solver = get_solver(method=args.method)

    support_assumption = get_support_assumption(**vars(args))

    # (Unknown) Generating probability
    distribution = get_distribution(**vars(args))

    # Generate Partitioning
    partition = get_partition(args=args, num_samples=args.num_samples_training, num_clusters=args.num_clusters)

    # Generate Quantization
    samples_quantization = distribution.sample((args.num_samples,))
    quantization = UncertainQuantization(partition=partition, samples=samples_quantization, beta=args.beta)

    # Plot samples and clusterized distribution
    if args.plot:
        plot_quantization(quantization=quantization, title=f"M={args.num_clusters}, N={args.num_samples}")
        plt.show()

    # Compute bounds
    samples = distribution.sample((args.num_samples + args.num_samples_training,))
    fournier_output = FournierRadius(
        samples=samples,
        support=partition.support,
        wasserstein_order=args.wasserstein_order,
        beta=args.beta,
        quantization_size=quantization.locs.shape[0],
    )
    data_driven_output = DataDrivenRadius(
        quantization=quantization, 
        solver=solver,
        wasserstein_order=args.wasserstein_order,
        compute_discrete_bound=args.compute_discrete_bound, 
        compute_moment_bound=args.compute_moment_bound
    )

    print(f"Number of clusters (M) / num_samples (N): {args.num_clusters} / {args.num_samples} \n"
          f"\t Fournier: {fournier_output.radius:.4f} \n"
          f"\t Ours : {data_driven_output.radius:.4f} \n"
          f"\t Fournier (comparable): {fournier_output.comparable_radius:.4f} \n"
        )

    print("Process finished.")

