import torch
from sets import HyperRectangle
from distributions import Uniform
from plotting.plot import plot_samples
from bound import bound_wasserstein
from cluster import k_means_cluster, cluster_regions
from utils import in_set, subdivide_hyperrectangle

if __name__ == '__main__':
    torch.manual_seed(0)

    num_samples = 2000
    beta = 1e-4

    support_assumption = HyperRectangle(lower=torch.tensor([0.0, 0.0]), upper=torch.tensor([0.5, 0.5]))

    # (Unknown) Generating probability
    support = HyperRectangle(lower=torch.tensor([0.1, 0.1]), upper=torch.tensor([0.15, 0.15]))
    distribution = Uniform(support=support)
    samples = distribution(num_samples=num_samples)

    # Clusterize samples (obtaining \hat{P}_M)
    labels = k_means_cluster(samples, 1)
    regions = cluster_regions(samples, labels)
    regions = subdivide_hyperrectangle(regions, n=1)

    # Plot samples and clusterized distribution
    plot_samples(samples=samples, regions=regions, support_assumption=support_assumption)

    bound = bound_wasserstein(beta=beta, support_assumption=support_assumption, partition=regions, samples=samples)
    print(f"Bound Triangle Inequality: {bound}")

    partial_bound_fournier = 1.42 / (num_samples ** 0.25)
    print(f"Partial bound Fournier: {partial_bound_fournier}")