import torch
from cluster import k_means_cluster, cluster_regions
from confidence import ClopperPearsonConfidence
from optimization import solve_transport_lp, o_maximization, max_min_lp
from plotting import plot_samples
from sets import HyperRectangle
from utils import subdivide_hyperrectangle, in_set


def bound_wasserstein_alternative(beta: float,
                                  shell: HyperRectangle,
                                  samples: torch.Tensor,
                                  plot: bool = False):

    num_samples = samples.shape[0]

    labels = k_means_cluster(samples, 1)
    regions = cluster_regions(samples, labels)

    regions = subdivide_hyperrectangle(regions, n=4)

    if plot:
        plot_samples(samples, regions, shell)

    d = regions.distance_matrix(shell=shell, p=2)

    n_set = in_set(samples=samples, regions=regions, include_complement=True)
    empirical = n_set / num_samples

    pearson_confidence = ClopperPearsonConfidence(beta=beta, n_set=n_set, n=num_samples)

    p_lower = pearson_confidence.lower_proba
    p_upper = pearson_confidence.upper_proba

    Pi_opt, W_opt = solve_transport_lp(d, p_lower, p_upper, empirical)

    return W_opt ** 0.5

def bound_wasserstein(beta: float,
                      shell: HyperRectangle,
                      partition: HyperRectangle,
                      samples: torch.Tensor):

    num_samples = samples.shape[0]

    d = partition.distance_matrix(shell=shell, p=2) / 2 # TODO: Proxy for including a point in the center

    n_set = in_set(samples=samples, regions=partition, include_complement=True)
    empirical = n_set / num_samples

    pearson_confidence = ClopperPearsonConfidence(beta=beta, n_set=n_set, n=num_samples)

    p_lower = pearson_confidence.lower_proba
    p_upper = pearson_confidence.upper_proba

    bound1 = o_maximization(d.diag().double(), p_lower.double(), p_upper.double()) ** 0.5

    d_points = partition.distance_centers(shell=shell, p=2)
    bound2 = max_min_lp(d_points, p_lower.double(), p_upper.double(), empirical.double()) ** 0.5

    return bound1 + bound2