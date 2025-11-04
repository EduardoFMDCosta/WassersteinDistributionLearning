import torch
import math
import ot

from sets import HyperRectangle
from quantization import UncertainQuantization
from optimization_utils import euclidean_projection_to_vertex, ot_lp_solver, sample_vertices
from solvers import MaxMinLP


def bound(
        quantization: UncertainQuantization,
        solver: MaxMinLP,
):
    bounds = solver.solve(quantization=quantization)
    return bounds


class DataDrivenRadius:
    _radius = torch.tensor(torch.nan)
    _moment_bound = torch.tensor(torch.nan)
    _discrete_bound = torch.tensor(torch.nan)

    def __init__(
            self, 
            quantization: UncertainQuantization, 
            solver: MaxMinLP,
        ):
        bounds = bound(quantization=quantization, solver=solver)
        self._radius = bounds.bound
        self._moment_bound = bounds.moment_bound
        self._discrete_bound = bounds.discrete_bound

        self._lower_bound = (quantization.upper_probs[-1] * (quantization.partition.support.width.norm() / 2).pow(2)).sqrt()

    @property
    def moment_bound(self) -> torch.Tensor:
        return self._moment_bound
    
    @property
    def discrete_bound(self) -> torch.Tensor:
        return self._discrete_bound

    @property
    def radius(self) -> torch.Tensor:
        return self._radius
    
    @property
    def lower_bound(self):
        return self._lower_bound
    
    def __repr__(self):
        return self.radius


def fournier_radius(
        support: HyperRectangle,
        nsamples: int,
        beta: float
) -> float:
    # See Proposition A.2. in Boissard and Le Gouic (2014)
    support_euclidean_diameter = support.width.norm(p=2).item()
    log_inv_beta = math.log(1 / beta)
    tau = support_euclidean_diameter * (2 * log_inv_beta / nsamples) ** 0.25

    constants = {
        5: 2.75,
        6: 2.20,
        7: 2.01,
        8: 1.92,
        9: 1.87,
        10: 1.85, # for 10 <= d <= 75, we divide 3rd line of Table 4 by sqrt(d)
        12: 1.83,
        15: 1.84,
        20: 1.87,
        25: 1.89,
        50: 1.95,
        75: 1.96,
        100: 1.98,
        500: 2.00
    }

    # See Table 2 in Fournier, 2023 (https://www.esaim-ps.org/articles/ps/pdf/2023/01/ps220050.pdf)
    if isinstance(support, HyperRectangle): # for infinite norm ball
        if support.ndim == 1:
            moment_bound = 1.05 / (nsamples ** (1 / 4))
        elif support.ndim == 2:
            moment_bound = 1.42 / (nsamples ** (1 / 4))
        elif support.ndim == 3:
            moment_bound = 2.20 / (nsamples ** (1 / 4))
        elif support.ndim == 4:
            moment_bound = math.sqrt(0.73 * math.log(nsamples) + 1.26) / (nsamples ** (1 / 4))
        else:
            moment_bound = constants[support.ndim] / (nsamples ** (1 / support.ndim))

        moment_bound = moment_bound * math.sqrt(support.ndim) # Adjustment for 2-Wasserstein

    else:
        raise NotImplementedError

    return moment_bound + tau


class EmpiricalRadius:
    def __init__(
        self,
        quantization: UncertainQuantization,
        dist: torch.distributions.Distribution,
    ):

        emp_dist = dist.sample((10 * quantization.nsamples,))

        self._radius_samples = ot.solve_sample(X_a=emp_dist, X_b=quantization.samples).value.sqrt().item()
        self._radius_quantization = ot.solve_sample(X_a=emp_dist, X_b=quantization.locs, b=quantization.probs).value.sqrt().item()

    @property
    def radius_samples(self) -> float:
        return self._radius_samples

    @property
    def radius_quantization(self) -> float:
        return self._radius_quantization