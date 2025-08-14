import torch
from sets import BoundedVoronoiPartition
from quantization import Quantization
from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from bound import data_driven_radius, fournier_radius
from configs.construct import get_support_assumption, get_distribution

from configs.handlers import parse_arguments

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        dimension=2,
        setting=0,
        num_samples=1000,
        num_clusters=10,
        beta=1e-4,
        plot=True
    )

    # Set parameters
    M = args.num_clusters
    N = args.num_samples
    beta = args.beta
    method = 'dual_sinkhorn'
    support_assumption = get_support_assumption(**vars(args))

    # (Unknown) Generating probability
    distribution = get_distribution(**vars(args))

    # Generate Partitions
    samples_partition = distribution.sample((N,))
    partition = BoundedVoronoiPartition(
        support=support_assumption, 
        samples=samples_partition, 
        M=M,
        use_voronoi_radii=False # set to false to speed up
    )

    # Generate Quantization
    samples_quantization = distribution.sample((N,))
    quantization = Quantization(partition=partition, samples=samples_quantization)

    # Plot samples and clusterized distribution
    if args.plot:
        plot_quantization(quantization=quantization, title=f"M={M}, N={N}")

    # Compute bounds
    fournier_bound = fournier_radius(support=partition.support, nsamples=N, beta=beta)
    data_driven_output = data_driven_radius(quantization=quantization, beta=beta, method=method)

    print(f"Number of clusters (M) / num_samples (N): {M} / {N} \n"
          f"\t Fournier: {fournier_bound:.4f} \n"
          f"\t Ours : {data_driven_output.radius:.4f} \n"
        )

