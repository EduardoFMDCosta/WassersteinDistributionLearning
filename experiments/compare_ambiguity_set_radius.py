import torch

from bound import compute_radius
from sets import HyperRectangle, Partition
from distributions import Uniform
from plotting.plot import plot_samples
from utils import generate_grid

if __name__ == '__main__':
    torch.manual_seed(0)

    M = 10
    N = 2000
    beta = 1e-4

    support_assumption = HyperRectangle(lower=torch.tensor([0.0, 0.0]), upper=torch.tensor([0.5, 0.5]))

    # (Unknown) Generating probability
    support = HyperRectangle(lower=torch.tensor([0.1, 0.1]), upper=torch.tensor([0.15, 0.15]))
    distribution = Uniform(support=support)
    samples = distribution(num_samples=N)

    # Clusterize samples (obtaining \hat{P}_M)
    locs = generate_grid(samples, int(M ** 0.5))
    partition = Partition(locs=locs, support=support_assumption)

    # Plot samples and clusterized distribution
    plot_samples(samples=samples, regions=partition.regions, support_assumption=support_assumption)

    bound = compute_radius(samples=samples, partition=partition, beta=beta)
    print(f"Bound Triangle Inequality: {bound}")

    partial_bound_fournier = 1.42 / (N ** 0.25)
    print(f"Partial bound Fournier: {partial_bound_fournier}")