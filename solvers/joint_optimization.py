import torch
import cvxpy as cp

from quantization import UncertainQuantization
from .templates import MaxMinLP, MaxMinLPResult

def solve_milp(
    cost: torch.Tensor,
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
    term_transport = cp.sum(cp.multiply(cost, Pi))
    term_diag = cp.sum(cp.multiply(cp.diag(cost), w))
    objective = cp.Maximize(term_transport + term_diag)

    constraints = []

    # Column sums
    for j in range(n):
        constraints.append(cp.sum(Pi[:, j]) == empirical_distribution[j])

    # Row sums
    for i in range(n):
        constraints.append(cp.sum(Pi[i, :]) == w[i])

    # Bounds on w
    constraints += [w >= lower, w <= upper]

    # Big-M linearization for min(w[i], empirical_distribution[i])
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

class JointOptimization(MaxMinLP):
    def __init__(self):
        super().__init__()

    def solve(
        self,
        quantization: UncertainQuantization,
        tol=1e-7,
    ) -> MaxMinLPResult:

        cost_matrix = (quantization.partition.distance_locs + quantization.partition.radii.unsqueeze(-1)).pow(2).T  # j,i # TODO: CHECK IF TRANSPOSE OR NOT

        total_value, w_opt, diag_term_value, transport_term_value = solve_milp(cost=cost_matrix, empirical_distribution=quantization.probs, lower=quantization.lower_probs, upper=quantization.upper_probs)

        factor = 2
        obj = torch.as_tensor(factor * total_value).pow(0.5)
        obj_moment = torch.as_tensor(diag_term_value).pow(0.5)
        obj_discrete = torch.as_tensor(transport_term_value).pow(0.5)
        w_opt = torch.as_tensor(w_opt)

        return MaxMinLPResult(bound=obj, moment_bound=obj_moment, discrete_bound=obj_discrete, w_opt=w_opt)