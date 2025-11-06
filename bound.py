from typing import Optional
import torch
import math
import ot

from sets import HyperRectangle
from quantization import UncertainQuantization
from solvers import Solver, Result


class DataDrivenRadius:
    _radius = torch.tensor(torch.nan)
    _moment_bound = torch.tensor(torch.nan)
    _discrete_bound = torch.tensor(torch.nan)

    def __init__(
            self, 
            quantization: UncertainQuantization, 
            solver: Solver,
            wasserstein_order: int,
            compute_discrete_bound: bool = True,
            compute_moment_bound: bool = True,
            time_limit: Optional[float] = None
        ):
        solver.wasserstein_order = wasserstein_order
        solver.compute_discrete_bound = compute_discrete_bound
        solver.compute_moment_bound = compute_moment_bound
        solver.time_limit = time_limit

        self._result =  solver.solve(quantization=quantization)
        
        self._lower_bound = (quantization.upper_probs[-1] * (quantization.support.width.norm() / 2).pow(2)).sqrt()

    @property
    def moment_bound(self) -> torch.Tensor:
        return self._result.moment_bound
    
    @property
    def discrete_bound(self) -> torch.Tensor:
        return self._result.discrete_bound

    @property
    def radius(self) -> torch.Tensor:
        return self._result.bound
    
    @property
    def lower_bound(self):
        return self._lower_bound


def fournier_radius(
    support: HyperRectangle,
    nsamples: int,
    wasserstein_order: int,
    beta: float
) -> float:
    # See Proposition A.2. in Boissard and Le Gouic (2014)
    support_euclidean_diameter = support.width.norm(p=2).item()
    log_inv_beta = math.log(1 / beta)
    tau = support_euclidean_diameter * (2 * log_inv_beta / nsamples) ** (1 / (2 * wasserstein_order))

    if wasserstein_order == 1:
        constants = {
            3: 3.72,
            4: 2.45,
            5: 2.09,
            6: 1.94,
            7: 1.87,
            8: 1.84,
            9: 1.82,
            10: 1.81,  # for 10 <= d <= 75, we divide 3rd line of Table 3 by sqrt(d)
            12: 1.82,
            15: 1.84,
            20: 1.87,
            25: 1.89,
            50: 1.95,
            75: 1.96,
            100: 1.98,
            500: 2.00
        }

        # See Table 1 in Fournier, 2023 (https://hal.science/hal-03768963/)
        if isinstance(support, HyperRectangle):  # for infinite norm ball
            if support.ndim == 1:
                moment_bound = 2.42 / (nsamples ** (1 / 2))
            elif support.ndim == 2:
                moment_bound = math.sqrt(0.73 * math.log(nsamples) + 1.0) / (nsamples ** (1 / 2))
            else:
                moment_bound = constants[support.ndim] / (nsamples ** (1 / support.ndim))

            moment_bound = moment_bound * math.sqrt(support.ndim) # Adjustment for L2 norm
        else:
            raise NotImplementedError

    elif wasserstein_order == 2:
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

        # See Table 2 in Fournier, 2023 (https://hal.science/hal-03768963/)
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

            moment_bound = moment_bound * math.sqrt(support.ndim) # Adjustment for L2 norm
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return moment_bound + tau

class EmpiricalRadius:
    def __init__(
        self,
        quantization: UncertainQuantization,
        dist: torch.distributions.Distribution,
        wasserstein_order: int
    ):

        emp_dist = dist.sample((10 * quantization.nsamples,))

        metric = {1: "euclidean", 2: "sqeuclidean"}
        assert wasserstein_order in metric, "Empirical computation not available for this Wasserstein order."

        self._radius_samples = ot.solve_sample(X_a=emp_dist, X_b=quantization.samples, metric=metric[wasserstein_order]).value.pow(1 / wasserstein_order).item()
        self._radius_quantization = ot.solve_sample(X_a=emp_dist, X_b=quantization.locs, b=quantization.probs, metric=metric[wasserstein_order]).value.pow(1 / wasserstein_order).item()

    @property
    def radius_samples(self) -> float:
        return self._radius_samples

    @property
    def radius_quantization(self) -> float:
        return self._radius_quantization