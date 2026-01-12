import torch
import matplotlib.pyplot as plt

from quantization import UncertainQuantization
from bound import DataDrivenRadius, fournier_radius
from solvers import get_solver

from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from experiments.partitions import get_partition
from experiments.utils import load_quantization, load_quantization_samples

if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_samples_training=1000,
        num_clusters=20,
        wasserstein_order=1,
        beta=1e-6,
        method='full_search',
        plot=False,
        save=False,
        compute_discrete_bound=False, 
        compute_moment_bound=True
    )
    
    solver = get_solver(method=args.method)

    # Generate Partitioning
    partition = get_partition(args=args, num_samples=args.num_samples_training, num_clusters=args.num_clusters)

    # Generate Quantization
    quantization = load_quantization(args=args, partition=partition, N=args.num_samples)

    # Plot samples and clusterized distribution
    if args.plot:
        samples_quantization = load_quantization_samples(args, N=args.num_samples)
        plot_quantization(
            quantization=quantization, 
            samples=samples_quantization,
            title=f"M={args.num_clusters}, N={args.num_samples}"
        )
        plt.show()

    # Compute bounds
    fournier_bound = fournier_radius(
        support=partition.support, 
        nsamples=args.num_samples + args.num_samples_training,
        wasserstein_order=args.wasserstein_order,
        beta=args.beta
    )
    data_driven_output = DataDrivenRadius(
        quantization=quantization, 
        solver=solver,
        wasserstein_order=args.wasserstein_order,
        compute_discrete_bound=args.compute_discrete_bound, 
        compute_moment_bound=args.compute_moment_bound
    )

    print(f"Number of clusters (M) / num_samples (N): {args.num_clusters} / {args.num_samples} \n"
          f"\t Fournier: {fournier_bound:.4f} \n"
          f"\t Ours : {data_driven_output.radius:.4f} \n"
        )

    print("Process finished.")

