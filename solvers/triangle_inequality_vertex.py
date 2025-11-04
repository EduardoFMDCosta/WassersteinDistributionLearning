import torch
import cvxpy as cp
import numpy as np

from optimization_utils import euclidean_projection_to_vertex, ot_lp_solver
from quantization import UncertainQuantization
from .templates import MaxMinLP, MaxMinLPResult

def lifted_lp_from_vertex(cost: torch.Tensor,
              p: torch.Tensor,
              lower: torch.Tensor,
              upper: torch.Tensor,
              I: list,
              J: list,
              method: str = "cvxopt",
              tol: float = 1e-8):
    # Convert to numpy
    cost_np = cost.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    l_np = lower.detach().cpu().numpy()
    u_np = upper.detach().cpu().numpy()

    n = len(p_np)
    mI, mJ = len(I), len(J)

    # Variables
    w = cp.Variable(n)
    s = cp.Variable(mI, nonneg=True)
    r = cp.Variable(mJ, nonneg=True)
    t = cp.Variable((mI, mJ), nonneg=True)

    # Constraints
    cons = []
    cons += [w >= l_np, w <= u_np]

    # s_i = w_i - p_i  for i in I
    for k, i in enumerate(I):
        cons.append(s[k] == w[i] - p_np[i])
    # r_j = p_j - w_j for j in J
    for k, j in enumerate(J):
        cons.append(r[k] == p_np[j] - w[j])

    # transport marginal constraints
    cons += [cp.sum(t, axis=1) == s,  # row sums = s_i
             cp.sum(t, axis=0) == r]  # col sums = r_j

    # Objective
    obj_expr = 0
    # diagonal terms for I
    for i in I:
        obj_expr += cost_np[i, i] * w[i]
    # constant diagonal terms for J
    const_term = sum(cost_np[j, j] * p_np[j] for j in J)
    # transport term
    if mI > 0 and mJ > 0:
        C_IJ = cost_np[np.ix_(I, J)]
        obj_expr += cp.sum(cp.multiply(C_IJ, t))

    objective = cp.Maximize(obj_expr + const_term)

    # Problem
    prob = cp.Problem(objective, cons)
    prob.solve(solver=method, verbose=False, feastol=tol, reltol=tol, abstol=tol)

    result = {
        "w": torch.tensor(w.value, dtype=torch.float64),
        "t": torch.tensor(t.value, dtype=torch.float64),
        "objective": float(prob.value),
        "status": prob.status,
    }
    return result

class TriangleInequalityFromVertex(MaxMinLP):
    def __init__(self):
        super().__init__()

    def solve(
        self,
        quantization: UncertainQuantization,
        tol=1e-7,
    ) -> MaxMinLPResult:

        lower = quantization.lower_probs
        upper = quantization.upper_probs

        # Get nearest vertex to empirical
        vertex = euclidean_projection_to_vertex(w=quantization.probs, lower=quantization.lower_probs, upper=quantization.upper_probs)

        n = vertex.shape[0]

        # Identify fixed indices
        I_fixed = torch.nonzero(vertex <= lower + tol).flatten().tolist()
        J_fixed = torch.nonzero(vertex >= upper - tol).flatten().tolist()
        free = list(set(range(n)) - set(I_fixed) - set(J_fixed))
        if len(free) != 1:
            raise ValueError("There must be exactly one free index")
        free_idx = free[0]

        # Run the LP twice, once assuming free belongs to I, once to J
        best_obj = -float('inf')
        best_w = None
        for free_assignment in ["I", "J"]:
            if free_assignment == "I":
                I = I_fixed + [free_idx]
                J = J_fixed
            else:
                I = I_fixed
                J = J_fixed + [free_idx]

            # Convert to lists
            I = list(I)
            J = list(J)
            if len(I) == 0 or len(J) == 0:
                continue

            cost_matrix = (quantization.partition.distance_locs + quantization.partition.radii.unsqueeze(-1)).pow(2).T  # j,i # TODO: CHECK IF TRANSPOSE OR NOT
            result = lifted_lp_from_vertex(cost=cost_matrix, p=vertex, lower=lower, upper=upper, I=I, J=J)
            obj_val = result["objective"]
            w_opt = result["w"]

            if obj_val > best_obj:
                best_obj = obj_val
                best_w = w_opt.clone()

        moment_bound = torch.as_tensor(best_obj).pow(0.5)

        # Compute second term
        cost_matrix = quantization.partition.distance_locs.pow(2)
        _, discrete_bound, _ = ot_lp_solver(cost=cost_matrix, w=vertex, empirical_distribution=quantization.probs)

        discrete_bound = torch.as_tensor(discrete_bound).pow(0.5)

        objective_opt = moment_bound + discrete_bound

        return MaxMinLPResult(bound=objective_opt, moment_bound=moment_bound, discrete_bound=discrete_bound)