import torch
from sets import HyperRectangle
from utils import in_set
from confidence import ClopperPearsonConfidence
from optimization import o_maximization, max_min_lp

def bound_wasserstein(beta: float,
                      support_assumption: HyperRectangle,
                      partition: HyperRectangle,
                      samples: torch.Tensor):

    num_samples = samples.shape[0]

    d = partition.distance_matrix(support_assumption=support_assumption, p=2) / 2 # TODO: Proxy for including a point in the center

    n_set = in_set(samples=samples, regions=partition, include_complement=True)
    empirical = n_set / num_samples

    pearson_confidence = ClopperPearsonConfidence(beta=beta, n_set=n_set, n=num_samples)

    p_lower = pearson_confidence.lower_proba
    p_upper = pearson_confidence.upper_proba

    bound1 = o_maximization(d.diag().double(), p_lower.double(), p_upper.double()) ** 0.5

    d_points = partition.distance_centers(support_assumption=support_assumption, p=2)
    bound2 = max_min_lp(d_points, p_lower.double(), p_upper.double(), empirical.double()) ** 0.5

    return bound1 + bound2