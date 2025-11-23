import torch
import matplotlib.pyplot as plt

from quantization import UncertainQuantization
from bound import DataDrivenRadius
from solvers import IndependentSolver
from solvers.discrete_solvers import FullSearch, StochasticVerticeAscent

from configs.handlers import parse_arguments
from configs.construct import get_support_assumption, get_distribution
from experiments.partitions import get_partition
from experiments.datastructures import DataDrivenRadii

def num_vertices_omega_space_vertices(num_dims: int):
    return num_dims * (2 ** (num_dims - 1))

if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_samples_training=1000,
        compute_discrete_bound=True,
        compute_moment_bound=False,
    )
    
    torch.manual_seed(args.random_seed)

    support_assumption = get_support_assumption(**vars(args))
    distribution = get_distribution(**vars(args))
    samples_quantization = distribution.sample((args.num_samples,))

    fs_radii = DataDrivenRadii()
    sva_radii = DataDrivenRadii()

    num_steps = 5
    ratio = 0.5
    
    num_cluster_options = [5,6,7,8,9,10]
    for num_clusters in num_cluster_options:
        max_vertices = num_vertices_omega_space_vertices(num_clusters)
        max_inits = max(int(max_vertices * ratio / num_steps), 1)
        print(f"vertices = {max_vertices}, inits = {max_inits}")

        fs_solver = IndependentSolver(discrete_solver=FullSearch(max_vertices=max_vertices))
        sva_solver = IndependentSolver(discrete_solver=StochasticVerticeAscent(num_inits=max_inits, num_steps=num_steps))

        partition = get_partition(args=args, num_samples=args.num_samples_training, num_clusters=num_clusters)    
        quantization = UncertainQuantization(partition=partition, samples=samples_quantization, beta=args.beta)

        # # Compute bounds
        fs_radii.append((args.num_samples, num_clusters), DataDrivenRadius(quantization=quantization, solver=fs_solver))
        sva_radii.append((args.num_samples, num_clusters), DataDrivenRadius(quantization=quantization, solver=sva_solver))


    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(num_cluster_options, fs_radii.discrete_bound, label='Full Search')
    ax.plot(num_cluster_options, sva_radii.discrete_bound, label='Stochastic Vertice Ascent')
    ax.set_xlabel('Number of clusters M')
    ax.set_ylabel('e2')
    ax.set_title(f"sva underparametrized by factor {ratio} (N={args.num_samples})")
    ax.legend()

    plt.show()
    pass

