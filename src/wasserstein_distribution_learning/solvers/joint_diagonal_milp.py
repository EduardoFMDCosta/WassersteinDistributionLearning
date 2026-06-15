import torch
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from typing import Optional

from ..quantization import UncertainQuantization
from .templates import Solver, Result

def solve_milp_gurobi(
    cost: torch.Tensor,
    empirical_distribution: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    time_limit: Optional[float] = None,
    **kwargs
):
    n = len(empirical_distribution)
    cost = cost.detach().cpu().numpy()
    empirical_distribution = empirical_distribution.detach().cpu().numpy()
    lower = lower.detach().cpu().numpy()
    upper = upper.detach().cpu().numpy()

    M = upper - lower  # tight big-M constants

    # Create model
    model = gp.Model("MILP")

    # Decision variables
    Pi = model.addVars(n, n, lb=0.0, name="Pi")
    w = model.addVars(n, lb=0.0, name="w")
    m = model.addVars(n, lb=0.0, name="m")
    b = model.addVars(n, vtype=GRB.BINARY, name="b")

    # Objective
    objective = gp.quicksum(cost[i, j] * Pi[i, j] for i in range(n) for j in range(n))
    model.setObjective(objective, GRB.MAXIMIZE)

    # Column sums
    for j in range(n):
        model.addConstr(gp.quicksum(Pi[i, j] for i in range(n)) == empirical_distribution[j])

    # Row sums
    for i in range(n):
        model.addConstr(gp.quicksum(Pi[i, j] for j in range(n)) == w[i])

    # Bounds on w
    for i in range(n):
        model.addConstr(w[i] >= lower[i])
        model.addConstr(w[i] <= upper[i])

    # Big-M linearization constraints
    for i in range(n):
        model.addConstr(m[i] <= w[i])
        model.addConstr(m[i] <= empirical_distribution[i])
        model.addConstr(m[i] >= w[i] - M[i] * (1 - b[i]))
        model.addConstr(m[i] >= empirical_distribution[i] - M[i] * b[i])

        # Diagonal constraint
        model.addConstr(Pi[i, i] >= m[i])

    # w sums to 1
    model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1)

    # Set optimization params
    if n >= 500:
        model.setParam("FeasibilityTol", 1e-7)
        model.setParam("IntFeasTol", 1e-7)
        model.setParam("Presolve", 1)

    # Optimize
    model.setParam("OutputFlag", kwargs.get("verbose", False))
    model.setParam("TimeLimit", time_limit if time_limit is not None else GRB.INFINITY)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution within {time_limit} seconds (status {model.Status})")

    # Extract results
    obj_value = model.objVal
    w_opt = torch.tensor([w[i].X for i in range(n)], dtype=torch.float64)

    return obj_value, w_opt


def solve_milp_cvxpy(
    cost: torch.Tensor,
    empirical_distribution: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    **kwargs
):
    raise NotImplementedError

class JointDiagonalMilp(Solver):
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

        locs_to_region_cost = quantization.l2_distance_locs_to_region.pow(self.wasserstein_order)

        if not self.use_gurobi:
            objective, w_opt = solve_milp_cvxpy(
                cost=locs_to_region_cost,
                empirical_distribution=quantization.probs,
                lower=quantization.interval.lower,
                upper=quantization.interval.upper,
            )
        else:
            objective, w_opt = solve_milp_gurobi(
                cost=locs_to_region_cost,
                empirical_distribution=quantization.probs,
                lower=quantization.interval.lower,
                upper=quantization.interval.upper,
                time_limit=self.time_limit,
            )

        bound = torch.as_tensor(objective).pow(1 / self.wasserstein_order)
        w_opt = torch.as_tensor(w_opt)

        return Result(bound=bound, w_opt=w_opt)