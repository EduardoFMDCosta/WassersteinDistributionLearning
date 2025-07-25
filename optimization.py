import torch
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

def o_maximization(cost: torch.Tensor,
                   lower: torch.Tensor,
                   upper: torch.Tensor):

    sorted_idx = torch.argsort(cost, descending=True)
    p = torch.zeros_like(cost)
    total = 1.0

    for j in sorted_idx:
        lo = lower[j].item()
        hi = upper[j].item()
        alloc = min(hi, max(lo, total))
        p[j] = alloc
        total -= alloc
        if total <= 1e-8:
            break

    result = torch.einsum('i,i->', cost, p)
    return result

def solve_inner_lp(d, w, p):
    """
    Solves min_{Pi} sum d_{ij} Pi_{ij}
    s.t. sum_j Pi_{ij} = w_i
         sum_i Pi_{ij} = p_j
         Pi >= 0
    """
    n = d.shape[0]
    Pi = cp.Variable((n, n))
    objective = cp.Minimize(cp.sum(cp.multiply(d, Pi)))
    constraints = [
        cp.sum(Pi, axis=1) == w,
        cp.sum(Pi, axis=0) == p,
        Pi >= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Inner LP did not solve properly: {prob.status}")
    return prob.value  # this is the minimum cost for given w

def max_min_lp(d: torch.Tensor, a: torch.Tensor, b: torch.Tensor, p: torch.Tensor):
    """
    Outer maximization over w in [a, b], sum w = 1
    Inner minimization over Pi as LP
    """
    n = d.shape[0]
    d_np = d.numpy()
    a_np = a.numpy()
    b_np = b.numpy()
    p_np = p.numpy()

    def objective(w_np):
        w = np.clip(w_np, a_np, b_np)
        w = w / w.sum()  # enforce sum w = 1
        return solve_inner_lp(d_np, w, p_np)

    def neg_objective(w_np):
        return -objective(w_np)

    # Initial guess
    w0 = (a_np + b_np) / 2
    w0 = w0 / w0.sum()  # normalize

    # Bounds and constraint
    bounds = [(ai, bi) for ai, bi in zip(a_np, b_np)]
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    res = minimize(neg_objective, w0, method='SLSQP', bounds=bounds, constraints=[cons])

    if not res.success:
        raise RuntimeError(f"Outer optimization failed: {res.message}")

    w_opt = torch.tensor(res.x, dtype=torch.float32)
    min_cost = -res.fun
    return min_cost


def solve_transport_lp(d, p_lower, p_upper, P_R):

    M = d.shape[0]
    Pi = cp.Variable((M, M), nonneg=True)

    # Objective
    objective = cp.Maximize(cp.sum(cp.multiply(d, Pi)))

    # Row sum constraints: for each m, the sum over columns in [p_lower, p_upper]
    row_sums = cp.sum(Pi, axis=1)
    constraints = [row_sums >= p_lower,
                   row_sums <= p_upper]

    # Column sum constraints: for each l, the sum over rows equals P_R
    col_sums = cp.sum(Pi, axis=0)
    constraints += [col_sums == P_R]

    # Solve the LP
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return Pi.value, prob.value