import math
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


def data_driven_radius(samples: torch.Tensor,
                       partition: Partition,
                       beta: float):

    num_samples = samples.shape[0]

    n_set = in_set(samples=samples, regions=partition.regions, include_complement=False)
    empirical = n_set / num_samples

    assert torch.allclose(empirical.sum(), torch.tensor(1.0), atol=1e-8), "Empirical distribution should sum to 1.0"

    pearson_confidence = ClopperPearsonConfidence(beta=beta, n_set=n_set, n=num_samples)

    moment_bound = bound_moment(partition=partition, confidence=pearson_confidence)
    discrete_bound = bound_discrete(partition=partition, confidence=pearson_confidence, empirical=empirical)

    return moment_bound + discrete_bound

def fournier_radius(samples: torch.Tensor,
                    partition: Partition,
                    beta: float):

    #See Lemma 2 in Gracia et at, 2024 (https://proceedings.mlr.press/v242/gracia24a/gracia24a.pdf)
    num_samples = samples.shape[0]
    support_diameter = partition.support.width.max().item()
    log_inv_beta = math.log(1 / beta)
    tau = (2 * support_diameter ** 4 * log_inv_beta / num_samples) ** 0.25

    # See Table 2 in Fournier, 2023 (https://www.esaim-ps.org/articles/ps/pdf/2023/01/ps220050.pdf
    if support_diameter == 1.0 and samples.shape[-1] == 2: # TODO: Create a separate file with the bounds from the paper case by case
        moment_bound = 1.42 / (num_samples ** 0.25)
    else:
        raise NotImplementedError

    return moment_bound + tau