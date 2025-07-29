import torch

from bound import data_driven_radius, fournier_radius
from sets import HyperRectangle, Partition
from distributions import Uniform
from plotting.plot import plot_samples
from utils import generate_grid_from_samples

if __name__ == '__main__':
    torch.manual_seed(0)

    M = 10
    N = 2000
    beta = 1e-4
    method = 'cvx_layers'

    support_assumption = HyperRectangle(lower=torch.tensor([-0.5, -0.5]), upper=torch.tensor([0.5, 0.5]))

    # (Unknown) Generating probability
    support = HyperRectangle(lower=torch.tensor([-0.15, -0.15]), upper=torch.tensor([0.15, 0.15]))
    distribution = Uniform(support=support)
    samples = distribution(num_samples=N)

    # Clusterize samples (obtaining \hat{P}_M)
    locs = generate_grid_from_samples(samples, int(M ** 0.5))
    partition = Partition(locs=locs, support=support_assumption)
    for i in range(2):
        partition = partition.refine(samples=samples, prob_thr=0.01, diam_thr=0.1)

    # Plot samples and clusterized distribution
    plot_samples(samples=samples, regions=partition.regions, support_assumption=support_assumption)

    data_driven_bound = data_driven_radius(samples=samples, partition=partition, beta=beta, method=method)
    print(f"Ours: {data_driven_bound}")

    fournier_bound = fournier_radius(samples=samples, partition=partition, beta=beta)
    print(f"Fournier: {fournier_bound}")