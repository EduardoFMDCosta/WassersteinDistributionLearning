import math
from quantization import UncertainQuantization
from sets import Partition, HyperRectangle
from quantization import UncertainQuantization
from optimization import o_maximization, max_min_lp

def bound_moment(
        quantization: UncertainQuantization,
):
    bound, _ = o_maximization(quantization.partition.radii, quantization.lower_probs, quantization.upper_probs)
    return bound ** 0.5


def bound_discrete(
        quantization: UncertainQuantization,
        method: str
):
    cost_matrix = quantization.partition.distance_locs ** 2

    bound = max_min_lp(
        cost=cost_matrix,
        lower=quantization.lower_probs,
        upper=quantization.upper_probs,
        empirical_marginal=quantization.probs,
        method=method
    )

    return bound ** 0.5


class DataDrivenRadius:
    def __init__(self, quantization: UncertainQuantization, method: str):
        self.moment_bound = bound_moment(quantization=quantization)
        self.discrete_bound = bound_discrete(quantization=quantization, method=method)

        self.lower_bound = (quantization.upper_probs[-1] * (quantization.partition.support.width.norm() / 2).pow(2)).sqrt()

    @property
    def radius(self):
        return self.moment_bound + self.discrete_bound
    
    def lower_bound(self):
        return self.moment_bound
    
    def __repr__(self):
        return self.radius


def fournier_radius(
        support: HyperRectangle,
        nsamples: int,
        beta: float
):
    #See Lemma 2 in Gracia et at, 2024 (https://proceedings.mlr.press/v242/gracia24a/gracia24a.pdf)
    support_diameter = support.width.max().item()
    log_inv_beta = math.log(1 / beta)
    tau = (2 * support_diameter ** 4 * log_inv_beta / nsamples) ** 0.25

    # See Table 2 in Fournier, 2023 (https://www.esaim-ps.org/articles/ps/pdf/2023/01/ps220050.pdf)
    if support_diameter == 1.0:
        if support.ndim == 1:
            moment_bound = 1.05 / (nsamples ** (1 / 4))
        elif support.ndim == 2:
            moment_bound = 1.42 / (nsamples ** (1 / 4))
        elif support.ndim == 3:
            moment_bound = 2.20 / (nsamples ** (1 / 4))
        elif support.ndim == 4:
            moment_bound = math.sqrt(0.73 * math.log(nsamples) + 1.26) / (nsamples ** (1 / 4))
        elif support.ndim == 5:
            moment_bound = 2.75 / (nsamples ** (1 / 5))
        elif support.ndim == 6:
            moment_bound = 2.20 / (nsamples ** (1 / 6))
        elif support.ndim == 7:
            moment_bound = 2.01 / (nsamples ** (1 / 7))
        elif support.ndim == 8:
            moment_bound = 1.92 / (nsamples ** (1 / 8))
        elif support.ndim == 9:
            moment_bound = 1.87 / (nsamples ** (1 / 9))
        else:
            raise NotImplementedError

        moment_bound = moment_bound * math.sqrt(support.ndim) # Adjustment for 2-Wasserstein

    else:
        raise NotImplementedError

    return moment_bound + tau