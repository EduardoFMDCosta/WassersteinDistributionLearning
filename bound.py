import torch
from sets import Partition
from utils import in_set
from confidence import ClopperPearsonConfidence, Confidence
from optimization import o_maximization, max_min_lp

def bound_moment(partition: Partition,
                 confidence: Confidence):

    cost = partition.sup_distance_within_regions()
    bound = o_maximization(cost, confidence.lower_proba, confidence.upper_proba) ** 0.5

    return bound

def bound_discrete(partition: Partition,
                   confidence: Confidence,
                   empirical: torch.Tensor):

    cost = partition.distance_locs()
    bound = max_min_lp(cost, confidence.lower_proba, confidence.upper_proba, empirical) ** 0.5

    return bound


def compute_radius(samples: torch.Tensor,
                   partition: Partition,
                   beta: float):

    num_samples = samples.shape[0]

    n_set = in_set(samples=samples, regions=partition.regions, include_complement=False)
    empirical = n_set / num_samples

    assert empirical.sum() == 1.0, "Empirical distribution should sum to 1.0"

    pearson_confidence = ClopperPearsonConfidence(beta=beta, n_set=n_set, n=num_samples)

    moment_bound = bound_moment(partition=partition, confidence=pearson_confidence)
    discrete_bound = bound_discrete(partition=partition, confidence=pearson_confidence, empirical=empirical)

    return moment_bound + discrete_bound