import torch

from configs.construct import get_distribution
from configs.handlers import parse_arguments
from experiments.utils import DataDrivenRadii

from quantization import UncertainQuantization
from sets import BoundedVoronoiPartition
from configs.construct import get_support_assumption, get_distribution
from bound import DataDrivenRadius

from solvers import get_solver


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_clusters=10,
        beta=1e-6,
        plot=True, 
        save=False,
        compute_moment_bound=False,
        compute_discrete_bound=True,
    )
    investigate_clusters = True

    support_assumption = get_support_assumption(**vars(args))
    distribution = get_distribution(**vars(args))

    samples_partition = distribution.sample((args.num_samples_training,))
    samples_quantization = distribution.sample((args.num_samples,))

    N = args.num_samples
    M_options = [5]  # [10, 25, 75, 100, 200, 500, 1000]

    solvers = dict(
        worst= get_solver(method='scalar_strategy', strategy='worst'),
        exact= get_solver(method='scalar_strategy', strategy='exact'),
        bench= get_solver(method='full_search')
    )

    data_worst, data_exact, data_bench = DataDrivenRadii(), DataDrivenRadii(), DataDrivenRadii()
    for M in M_options:
        partition = BoundedVoronoiPartition(
            support=support_assumption, 
            samples=samples_partition, 
            M=M,
        )
        quantization = UncertainQuantization(partition=partition, samples=samples_quantization, beta=args.beta)

        # Compute bounds
        data_worst.append((N, M), DataDrivenRadius(quantization=quantization, solver=solvers['worst']))
        data_exact.append((N, M), DataDrivenRadius(quantization=quantization, solver=solvers['exact']))
        data_bench.append((N, M), DataDrivenRadius(quantization=quantization, solver=solvers['bench']))

    print(data_worst.discrete_bound)
    print(data_exact.discrete_bound)
    print(data_bench.discrete_bound)