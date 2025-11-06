import torch
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from typing import Optional

from quantization import UncertainQuantization
from solvers.templates import Solver, Result

def solve_milp_gurobi(
    inside_region_cost: torch.Tensor,
    cross_location_cost: torch.Tensor,
    empirical_distribution: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    time_limit: Optional[float] = None,
    **kwargs
):
    n = len(empirical_distribution)
    inside_region_cost = inside_region_cost.detach().cpu().numpy()
    cross_location_cost = cross_location_cost.detach().cpu().numpy()
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
    term_diag = gp.quicksum(inside_region_cost[i] * w[i] for i in range(n))
    term_transport = gp.quicksum(cross_location_cost[i, j] * Pi[i, j] for i in range(n) for j in range(n))
    model.setObjective(term_diag + term_transport, GRB.MAXIMIZE)

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

    # Optimize
    model.setParam("OutputFlag", kwargs.get("verbose", False))
    model.optimize()

    # Extract results
    total_value = model.objVal
    w_opt = torch.tensor([w[i].X for i in range(n)], dtype=torch.float64)
    diag_term_value = sum(inside_region_cost[i] * w_opt[i].item() for i in range(n))
    transport_term_value = sum(
        cross_location_cost[i, j] * Pi[i, j].X for i in range(n) for j in range(n)
    )

    return total_value, w_opt, diag_term_value, transport_term_value


def solve_milp_cvxpy(
    inside_region_cost: torch.Tensor,
    cross_location_cost: torch.Tensor,
    empirical_distribution: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    **kwargs
):
    n = len(empirical_distribution)

    # Decision variables
    Pi = cp.Variable((n, n), nonneg=True)
    w = cp.Variable(n)
    m = cp.Variable(n)
    b = cp.Variable(n, boolean=True)

    # Objective
    term_diag = cp.sum(cp.multiply(inside_region_cost, w))
    term_transport = cp.sum(cp.multiply(cross_location_cost, Pi))
    objective = cp.Maximize(term_diag + term_transport)

    constraints = []

    # Column sums
    for j in range(n):
        constraints.append(cp.sum(Pi[:, j]) == empirical_distribution[j])

    # Row sums
    for i in range(n):
        constraints.append(cp.sum(Pi[i, :]) == w[i])

    # Bounds on w
    constraints += [w >= lower, w <= upper]

    # Big-M linearization for min(w[i], empirical_distribution[i]) # TODO use m.addGenConstrMin
    M = upper - lower  # tight big-M
    for i in range(n):
        constraints.append(m[i] <= w[i])
        constraints.append(m[i] <= empirical_distribution[i])
        constraints.append(m[i] >= w[i] - M[i] * (1 - b[i]))
        constraints.append(m[i] >= empirical_distribution[i] - M[i] * b[i])

        # Pi[i,i] constraint
        constraints.append(Pi[i, i] >= m[i])

    # w sums to 1
    constraints.append(cp.sum(w) == 1)

    # Solve MILP
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GLPK_MI)

    # Extract optimal values
    total_value = prob.value
    w_opt = w.value
    diag_term_value = term_diag.value
    transport_term_value = term_transport.value

    return total_value, w_opt, diag_term_value, transport_term_value

class JointOptimizationMilp(Solver):
    def __init__(
        self,
        time_limit: Optional[float] = None,
        use_gurobi: bool = True
    ):
        super().__init__()

        self.time_limit = time_limit
        self.use_gurobi = use_gurobi

    def solve(
        self,
        quantization: UncertainQuantization,
        wasserstein_order: int,
    ) -> Result:

        inside_region_cost = quantization.l2_radii.pow(wasserstein_order)
        cross_location_cost = quantization.l2_distance_locs_to_locs.pow(wasserstein_order)

        if not self.use_gurobi:
            total_value, w_opt, diag_term_value, transport_term_value = solve_milp_cvxpy(
                inside_region_cost=inside_region_cost,
                cross_location_cost=cross_location_cost,
                empirical_distribution=quantization.probs,
                lower=quantization.lower_probs,
                upper=quantization.upper_probs
            )
        else:
            total_value, w_opt, diag_term_value, transport_term_value = solve_milp_gurobi(
                inside_region_cost=inside_region_cost,
                cross_location_cost=cross_location_cost,
                empirical_distribution=quantization.probs,
                lower=quantization.lower_probs,
                upper=quantization.upper_probs,
                time_limit=self.time_limit,
            )

        factor = 2 ** ((wasserstein_order - 1) / wasserstein_order)
        obj = factor * torch.as_tensor(total_value).pow(1 / wasserstein_order)
        obj_moment = factor * torch.as_tensor(diag_term_value).pow(1 / wasserstein_order)
        obj_discrete = factor * torch.as_tensor(transport_term_value).pow(1 / wasserstein_order)
        w_opt = torch.as_tensor(w_opt)

        return Result(bound=obj, moment_bound=obj_moment, discrete_bound=obj_discrete, w_opt=w_opt)