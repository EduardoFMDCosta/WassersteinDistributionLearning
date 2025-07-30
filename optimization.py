import torch
import numpy as np
import cvxpy as cp
from copy import deepcopy
from scipy.optimize import minimize
from cvxpylayers.torch import CvxpyLayer


def o_maximization(cost: torch.Tensor,
                   lower: torch.Tensor,
                   upper: torch.Tensor):

    # Inspired from https://www.baymler.com/IntervalMDP.jl/dev/algorithms/#Efficient-value-iteration
    order = torch.argsort(-cost)
    p = lower.clone()
    rem = 1 - p.sum()
    gap = upper - p
    cumgap = torch.cumsum(gap[order], dim=0)
    for idx, o in enumerate(order):
        rem_state = max(rem - cumgap[idx] + gap[o], 0)
        if gap[o] < rem_state:
            p[o] += gap[o]
        else:
            p[o] += rem_state
            break

    result = torch.einsum('i,i->', cost.double(), p.double())
    return result, p

def max_min_lp(cost: torch.Tensor,
               lower: torch.Tensor,
               upper: torch.Tensor,
               empirical_marginal: torch.Tensor,
               method: str,
               num_steps=1000,
               lr=1e-2,
               tol=1e-6):

    if method == 'cvx_layers':
        return max_min_lp_cvxlayers(cost=cost,
                                    lower=lower,
                                    upper=upper,
                                    empirical_marginal=empirical_marginal,
                                    num_steps=num_steps,
                                    lr=lr,
                                    tol=tol)
    elif method == 'cvxpy':
        return max_min_lp_cvx(cost=cost,
                              lower=lower,
                              upper=upper,
                              empirical_marginal=empirical_marginal,
                              num_steps=num_steps,
                              lr=lr,
                              tol=tol)
    elif method == 'dual':
        return max_min_lp_dual(cost=cost,
                              lower=lower,
                              upper=upper,
                              empirical_marginal=empirical_marginal,
                              num_steps=num_steps,
                              lr=lr,
                              tol=tol)
    else:
        raise ValueError('Unknown optimization method')

# ============================================================
#                      CVXPyLayers-based
# ============================================================

def lp_layer(d: torch.Tensor, p: torch.Tensor, n: int):

    w = cp.Parameter(n)
    Pi = cp.Variable((n, n))       # Transport plan

    objective = cp.Minimize(cp.sum(cp.multiply(d, Pi)))
    constraints = [
        cp.sum(Pi, axis=1) == w, # Row marginals
        cp.sum(Pi, axis=0) == p, # Column marginals
        Pi >= 0
    ]

    problem = cp.Problem(objective, constraints)
    layer = CvxpyLayer(problem, parameters=[w], variables=[Pi])
    return layer

def max_min_lp_cvxlayers(cost: torch.Tensor,
                         lower: torch.Tensor,
                         upper: torch.Tensor,
                         empirical_marginal: torch.Tensor,
                         num_steps: int,
                         lr: float,
                         tol: float):

    n = cost.shape[0]
    layer = lp_layer(d=cost, p=empirical_marginal, n=n)

    # Initialize w as a learnable parameter, normalized within [a, b] and sum=1
    w = torch.nn.Parameter(torch.randn(n), requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr)

    prev_obj = None

    for step in range(num_steps):

        with torch.no_grad():
            w.data = torch.clamp(w.data, lower, upper)
            w.data /= w.data.sum()

        # Solve inner LP
        Pi_star, = layer(w)

        # Compute objective
        wasserstein_squared = torch.sum(cost * Pi_star)
        loss = -wasserstein_squared  # max-min → maximize -min

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Early stopping condition
        current_obj = wasserstein_squared.item()
        if prev_obj is not None and abs(current_obj - prev_obj) < tol:
            print(f"Early stopping at step {step} with change {abs(current_obj - prev_obj):.2e}")
            break
        prev_obj = current_obj

        # Print every 100 steps
        if step % 1 == 20 or step == num_steps - 1:
            print(f"Step {step}: objective = {current_obj:.8f}")

    return wasserstein_squared.item()


# ============================================================
#           CVXPy-based (to be eventually removed)
# ============================================================

def solve_lp_cvx(d, w, p):
    """
    min_{Pi} sum d_{ij} Pi_{ij}
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
    return prob.value

def max_min_lp_cvx(cost: torch.Tensor,
                   lower: torch.Tensor,
                   upper: torch.Tensor,
                   empirical_marginal: torch.Tensor,
                   num_steps: int,
                   lr: float,
                   tol: float):

    d_np = cost.numpy()
    a_np = lower.numpy()
    b_np = upper.numpy()
    p_np = empirical_marginal.numpy()

    def objective(w_np):
        w = np.clip(w_np, a_np, b_np)
        w = w / w.sum()  # enforce sum w = 1
        return solve_lp_cvx(d_np, w, p_np)

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


# ============================================================
#                      Solving the dual
# ============================================================

def solve_dual_with_cvxpy(d, w, p):

    n = d.shape[0]
    d_np = d.detach().cpu().numpy()
    w_np = w.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()

    # Define CVXPY variables
    alpha = cp.Variable(n)
    beta = cp.Variable(n)

    # Objective
    objective = cp.Maximize(alpha @ w_np + beta @ p_np)

    # Constraints: α_i + β_j ≤ d_ij for all i, j
    constraints = [cp.reshape(alpha, (n, 1), order="C") + cp.reshape(beta, (1, n), order="C") <= d_np]

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"CVXPY solve failed: {problem.status}")

    alpha_val = torch.tensor(alpha.value, dtype=torch.float32)
    beta_val = torch.tensor(beta.value, dtype=torch.float32)
    return alpha_val, beta_val

def max_min_lp_dual(cost: torch.Tensor,
                         lower: torch.Tensor,
                         upper: torch.Tensor,
                         empirical_marginal: torch.Tensor,
                    num_steps: int,
                    lr: float,
                    tol: float):
    n = cost.shape[0]

    # Initialize dual variables
    alpha = torch.rand(n)
    beta = torch.rand(n)

    prev_obj = None

    for step in range(num_steps):

        # Solve maximization for w
        _, w = o_maximization(alpha, lower, upper)

        # Solve maximization for dual variables
        alpha, beta = solve_dual_with_cvxpy(cost, w, empirical_marginal)

        wasserstein_squared = torch.dot(alpha.double(), w.double()) + torch.dot(beta.double(), empirical_marginal.double())

        # Early stopping condition
        current_obj = wasserstein_squared.item()
        if prev_obj is not None and abs(current_obj - prev_obj) < tol:
            print(f"Early stopping at step {step} with change {abs(current_obj - prev_obj):.2e}")
            break
        prev_obj = current_obj

        # Print every 100 steps
        if step % 50 == 0 or step == num_steps - 1:
            print(f"Step {step}: objective = {current_obj:.8f}")

    # Compute objective
    wasserstein_squared = torch.dot(alpha.double(), w.double()) + torch.dot(beta.double(), empirical_marginal.double())

    return wasserstein_squared.item()
