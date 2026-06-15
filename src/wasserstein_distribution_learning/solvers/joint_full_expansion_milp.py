from typing import Optional

from ..optimization_utils import o_maximization
from ..quantization import UncertainQuantization
from .discrete_solvers import DiagonalConstrainedTP
from .joint_optimization_milp import JointOptimizationMilp
from .templates import Solver, Result

class JointFullExpansionMilp(Solver):
    def __init__(
        self,
        use_gurobi: bool = True
    ):
        super().__init__()
        self.use_gurobi = use_gurobi
        self.joint_optim_milp_solver = JointOptimizationMilp(use_gurobi=self.use_gurobi)
        self.diagonal_constrained_tp_solver = DiagonalConstrainedTP(use_gurobi=self.use_gurobi)

    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:

        inside_region_cost = quantization.l2_radii.pow(self.wasserstein_order)
        cross_location_cost = quantization.l2_distance_locs_to_locs.pow(self.wasserstein_order)

        factor = 2 ** (self.wasserstein_order - 1)
        
        sum_of_power_rho = self.joint_optim_milp_solver.solve(quantization=quantization).bound.pow(self.wasserstein_order) / factor # we need to adjust for the factor penalty

        moment, _ = o_maximization(
            cost=inside_region_cost,
            lower=quantization.interval.lower,
            upper=quantization.interval.upper,
        )
        moment = moment.pow(1 / self.wasserstein_order)

        discrete = self.diagonal_constrained_tp_solver.solve(
            cost=cross_location_cost,
            lower=quantization.interval.lower,
            upper=quantization.interval.upper,
            empirical_marginal=quantization.probs,
        ).bound

        obj = (sum_of_power_rho + 2 * moment * discrete).pow(1 / self.wasserstein_order)
        return Result(bound=obj)
    
    @property
    def time_limit(self) -> Optional[float]:
        return self._time_limit
    
    @time_limit.setter
    def time_limit(self, value: Optional[float]) -> None:
        self._time_limit = value
        self.joint_optim_milp_solver.time_limit = value
        self.diagonal_constrained_tp_solver.time_limit = value