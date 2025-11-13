import torch
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from typing import Optional

from optimization_utils import o_maximization
from quantization import UncertainQuantization
from solvers.discrete_solvers import DiagonalConstrainedTP
from solvers.joint_optimization_milp import solve_milp_cvxpy, solve_milp_gurobi, JointOptimizationMilp
from solvers.templates import Solver, Result

class JointFullExpansionMilp(Solver):
    def __init__(
        self,
        use_gurobi: bool = True
    ):
        super().__init__()
        self.use_gurobi = use_gurobi

    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:

        inside_region_cost = quantization.l2_radii.pow(self.wasserstein_order)
        cross_location_cost = quantization.l2_distance_locs_to_locs.pow(self.wasserstein_order)

        factor = 2 ** (self.wasserstein_order - 1)
        joint_optim_milp_solver = JointOptimizationMilp(use_gurobi=self.use_gurobi)
        sum_of_power_rho = joint_optim_milp_solver.solve(quantization=quantization).bound.pow(self.wasserstein_order) / factor # we need to adjust for the factor penalty

        moment, _ = o_maximization(cost=inside_region_cost, lower=quantization.lower_probs, upper=quantization.upper_probs)
        moment = moment.pow(1 / self.wasserstein_order)

        diagonal_constrained_tp_solver = DiagonalConstrainedTP(use_gurobi=self.use_gurobi)
        discrete = diagonal_constrained_tp_solver.solve(cost=cross_location_cost, lower=quantization.lower_probs, upper=quantization.upper_probs, empirical_marginal=quantization.probs).bound

        obj = (sum_of_power_rho + 2 * moment * discrete).pow(1 / self.wasserstein_order)
        return Result(bound=obj)