import torch
from sets import BoundedVoronoiPartition
from quantization import Quantization
from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from bound import data_driven_radius, fournier_radius
from configs.construct import get_support_assumption, get_distribution

from configs.handlers import parse_arguments

import matplotlib.pyplot as plt


def num_samples(args, M):
    support_assumption = get_support_assumption(**vars(args))

    distribution = get_distribution(**vars(args))
    
    data_driven_bounds, data_driven_lower_bounds, fournier_bounds = list(), list(), list()
    N_options = [1000, 5000, 10000, 50000]
    for N in N_options:
        print(f"Number of clusters (M) / num_samples (N): {M} / {N}")
        samples_partition = distribution.sample((N,))
        samples_quantization = distribution.sample((N,))

        # Clusterize samples (obtaining \hat{P}_M)
        partition = BoundedVoronoiPartition(support=support_assumption, samples=samples_partition, M=int(M))
        quantization = Quantization(partition=partition, samples=samples_quantization)

        # Plot samples and clusterized distribution
        if args.plot:
            plot_quantization(quantization=quantization)

        # Compute bounds
        data_driven_output = data_driven_radius(quantization=quantization, beta=args.beta, method=args.method)
        data_driven_bounds.append(data_driven_output.radius)
        data_driven_lower_bounds.append(data_driven_output.lower_bound)

        fournier_bounds.append(fournier_radius(support=partition.support, nsamples=N, beta=args.beta))

    return N_options, data_driven_bounds, fournier_bounds, data_driven_lower_bounds

def num_clusters(args, N):
    support_assumption = get_support_assumption(**vars(args))

    distribution = get_distribution(**vars(args))
    samples_partition = distribution.sample((N,))
    samples_quantization = distribution.sample((N,))

    data_driven_bounds, data_driven_lower_bounds, fournier_bounds = list(), list(), list()
    M_options = torch.arange(20, 100, 10).tolist()
    for M in M_options:
        print(f"Number of clusters (M) / num_samples (N): {M} / {N}")
        # Clusterize samples (obtaining \hat{P}_M)
        partition = BoundedVoronoiPartition(support=support_assumption, samples=samples_partition, M=int(M))
        quantization = Quantization(partition=partition, samples=samples_quantization)

        # Plot samples and clusterized distribution
        if args.plot:
            plot_quantization(quantization=quantization)

        # Compute bounds
        data_driven_output = data_driven_radius(quantization=quantization, beta=args.beta, method=args.method)
        data_driven_bounds.append(data_driven_output.radius)
        data_driven_lower_bounds.append(data_driven_output.lower_bound)

        fournier_bounds.append(fournier_radius(support=partition.support, nsamples=N, beta=args.beta))

    return M_options, data_driven_bounds, fournier_bounds, data_driven_lower_bounds

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Discrete",
        dimension=2,
        setting=0,
        num_samples=5000,
        num_clusters=40,
        beta=1e-4,
        plot=False
    )
    args.method = 'dual_sinkhorn'
    investigate_clusters = False

    if investigate_clusters:
        options, data_driven_bounds, fournier_bounds, data_driven_lower_bounds = num_clusters(args, N=args.num_samples)
    else:
        options, data_driven_bounds, fournier_bounds, data_driven_lower_bounds = num_samples(args, M=args.num_clusters)

    with torch.no_grad():
        plt.plot(options, torch.tensor(data_driven_bounds), label='Ours', marker='o')
        plt.plot(options, torch.tensor(fournier_bounds), label='Fournier', marker='x')
        plt.plot(options, torch.tensor(data_driven_lower_bounds), label='Ours Lower Bound', linestyle='--')
        plt.xlabel(f"Number of {'clusters (M)' if investigate_clusters else 'samples (N)'}")
        plt.title(f"Number of {'samples (N)' if investigate_clusters else 'clusters (M)'} = {args.num_samples if investigate_clusters else args.num_clusters}")
        plt.legend()
        plt.show()