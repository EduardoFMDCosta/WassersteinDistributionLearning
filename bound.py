import math
from sets import KMeansPartition
from confidence import ClopperPearsonConfidence, Confidence
from optimization import o_maximization, max_min_lp


def bound_moment(
        partition: KMeansPartition,
        confidence: Confidence
):

    bound, _ = o_maximization(partition.diameters, confidence.lower_proba, confidence.upper_proba)

    return bound ** 0.5


def bound_discrete(
        partition: KMeansPartition,
        confidence: Confidence,
        method: str
):
    bound = max_min_lp(
        cost=partition.distance_locs, 
        lower=confidence.lower_proba, 
        upper=confidence.upper_proba, 
        empirical_marginal=partition.probs, 
        method=method
    )

    return bound ** 0.5


def data_driven_radius(
        partition: KMeansPartition,
        beta: float,
        method: str
    ):

    adjusted_beta = beta / partition.npartitions
    pearson_confidence = ClopperPearsonConfidence(beta=adjusted_beta, n_set=partition.counts, n=partition.nsamples)

    moment_bound = bound_moment(partition=partition, confidence=pearson_confidence)
    discrete_bound = bound_discrete(partition=partition, confidence=pearson_confidence, method=method)

    return moment_bound + discrete_bound


def fournier_radius(
        partition: KMeansPartition,
        beta: float
):
    #See Lemma 2 in Gracia et at, 2024 (https://proceedings.mlr.press/v242/gracia24a/gracia24a.pdf)
    support_diameter = partition.support.width.max().item()
    log_inv_beta = math.log(1 / beta)
    tau = (2 * support_diameter ** 4 * log_inv_beta / partition.nsamples) ** 0.25

    # See Table 2 in Fournier, 2023 (https://www.esaim-ps.org/articles/ps/pdf/2023/01/ps220050.pdf)
    if support_diameter == 1.0:
        if partition.ndim == 1:
            moment_bound = 1.05 / (partition.nsamples ** (1 / 4))
        elif partition.ndim == 2:
            moment_bound = 1.42 / (partition.nsamples ** (1 / 4))
        elif partition.ndim == 3:
            moment_bound = 2.20 / (partition.nsamples ** (1 / 4))
        elif partition.ndim == 4:
            moment_bound = math.sqrt(0.73 * math.log(partition.nsamples) + 1.26) / (partition.nsamples ** (1 / 4))
        elif partition.ndim == 5:
            moment_bound = 2.75 / (partition.nsamples ** (1 / 5))
        elif partition.ndim == 6:
            moment_bound = 2.20 / (partition.nsamples ** (1 / 6))
        elif partition.ndim == 7:
            moment_bound = 2.01 / (partition.nsamples ** (1 / 7))
        elif partition.ndim == 8:
            moment_bound = 1.92 / (partition.nsamples ** (1 / 8))
        elif partition.ndim == 9:
            moment_bound = 1.87 / (partition.nsamples ** (1 / 9))
        else:
            raise NotImplementedError

        moment_bound = moment_bound * math.sqrt(partition.ndim) # Adjustment for 2-Wasserstein

    else:
        raise NotImplementedError

    return moment_bound + tau