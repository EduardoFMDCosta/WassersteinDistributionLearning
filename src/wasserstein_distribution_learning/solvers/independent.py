from typing import Optional
import torch

from ..optimization_utils import o_maximization
from ..quantization import UncertainQuantization

from .templates import Solver, Result, DiscreteSolver


class IndependentSolver(Solver):
    def __init__(
        self, 
        discrete_solver: DiscreteSolver,
    ) -> None:
        self.discrete_solver = discrete_solver
        self.wasserstein_order = discrete_solver.wasserstein_order

    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:

        if self.compute_moment_bound:
            moment_bound, _ = o_maximization(
                quantization.l2_radii.pow(self.wasserstein_order),
                quantization.interval.lower,
                quantization.interval.upper,
            )
            moment_bound = moment_bound.pow(1 / self.wasserstein_order)
        else:
            moment_bound = torch.tensor(torch.nan)

        if self.compute_discrete_bound:
            cost_matrix = quantization.l2_distance_locs_to_locs.pow(self.wasserstein_order)

            discrete_bound = self.discrete_solver.solve(
                cost=cost_matrix.detach(),
                lower=quantization.interval.lower,
                upper=quantization.interval.upper,
                empirical_marginal=quantization.probs 
            ).bound
        else:
            discrete_bound = torch.tensor(torch.nan)

        return Result(moment_bound=moment_bound, discrete_bound=discrete_bound)

    @property
    def wasserstein_order(self) -> int:
        return self._wasserstein_order

    @wasserstein_order.setter
    def wasserstein_order(self, value: int) -> None:
        self._wasserstein_order = value
        self.discrete_solver.wasserstein_order = value

    @property
    def time_limit(self) -> Optional[float]:
        return self._time_limit
    
    @time_limit.setter
    def time_limit(self, value: Optional[float]) -> None:
        self._time_limit = value
        self.discrete_solver.time_limit = value