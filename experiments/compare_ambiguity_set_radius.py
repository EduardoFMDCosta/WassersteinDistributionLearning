import torch
from sets import HyperRectangle
from distributions import Uniform
from plotting.plot import plot_samples
from bound import bound_wasserstein_alternative, bound_wasserstein
from cluster import k_means_cluster, cluster_regions
from utils import in_set, subdivide_hyperrectangle

if __name__ == '__main__':
    torch.manual_seed(0)

    N = 2000
    beta = 1e-4

    shell = HyperRectangle(lower=torch.tensor([0.0, 0.0]), upper=torch.tensor([0.5, 0.5]))

    support = HyperRectangle(lower=torch.tensor([0.1, 0.1]), upper=torch.tensor([0.15, 0.15]))
    distribution = Uniform(support=support)
    samples = distribution(num_samples=N)

    # support = HyperRectangle(lower=torch.tensor([0.4, 0.4]), upper=torch.tensor([0.45, 0.45]))
    # distribution = Uniform(support=support)
    # samples = torch.cat([samples, distribution(num_samples=N)], dim=0)

    num_samples = samples.shape[0]

    labels = k_means_cluster(samples, 1)
    regions = cluster_regions(samples, labels)
    regions = subdivide_hyperrectangle(regions, n=1)

    plot_samples(samples, regions, shell)


    #bound = bound_wasserstein_alternative(beta=beta, shell=shell, samples=samples, plot=True)
    #print(f"Bound Alternative: {bound}")

    bound = bound_wasserstein(beta=beta, shell=shell, partition=regions, samples=samples)
    print(f"Bound Triangle Inequality: {bound}")

    partial_bound_fournier = 1.42 / (num_samples ** 0.25)
    print(f"Partial bound Fournier: {partial_bound_fournier}")