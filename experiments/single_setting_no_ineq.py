import torch

from sets import BoundedVoronoiPartition
from quantization import UncertainQuantization
from bound import DataDrivenRadiusNoIneq, fournier_radius, DataDrivenRadius
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
        plot=True,
        save=True,
    )

    full_search_solver = get_solver(method="full_search")
    sva_solver = get_solver(method="stochastic_vertice_ascent")

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
    fournier = fournier_radius(
        support=partition.support, 
        nsamples=args.num_samples + args.num_samples_training, 
        beta=args.beta
    )
    no_ineq = DataDrivenRadiusNoIneq(quantization=quantization)
    full_search = DataDrivenRadius(quantization=quantization, solver=full_search_solver)
    sva = DataDrivenRadius(quantization=quantization, solver=sva_solver)

    print(f"Number of clusters (M) / num_samples (N): {args.num_clusters} / {args.num_samples} \n"
          f"\t Fournier: {fournier:.4f} \n"
          f"\t Conditional TP: {no_ineq.radius:.4f} \n"
          f"\t Full Search : {full_search.radius:.4f} \n"
          f"\t SVA : {sva.radius:.4f} \n"
        )

