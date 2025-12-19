from typing import Tuple, Optional

import torch
import numpy as np

import cvxpy as cp
from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB


def o_maximization(
    cost: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor, 
    tol: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Inspired from https://www.baymler.com/IntervalMDP.jl/dev/algorithms/#Efficient-value-iteration
    order = torch.argsort(-cost)
    p = lower.clone()
    rem = 1 - p.sum()
    gap = upper - p
    cumgap = torch.cumsum(gap[order], dim=0)
    for idx, o in enumerate(order):
        rem_state = rem - cumgap[idx] + gap[o]
        if rem_state <= 0:
            continue
        if gap[o] < rem_state:
            p[o] += gap[o]
        else:
            p[o] += rem_state
            break

    assert (p.sum() - 1.0).abs() <= tol
    assert (p >= lower - tol).all() & (p <= upper + tol).all()

    result = torch.einsum('i,i->', cost, p)
    return result, p


def ot_lp_solver(
    cost: torch.Tensor,
    w: torch.Tensor,
    empirical_distribution: torch.Tensor,
    tol: float = 1e-8,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

    device = cost.device
    dtype = cost.dtype
    n = cost.shape[0]

    C = cost.detach().cpu().double().numpy()
    p = w.detach().cpu().double().numpy()
    q = empirical_distribution.detach().cpu().double().numpy()

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    model = gp.Model("optimal_transport")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.FeasibilityTol = tol
    model.Params.OptimalityTol = tol

    # Decision variables: T[i,j] >= 0
    T = model.addVars(
        n, n,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="T"
    )

    # Objective
    model.setObjective(
        gp.quicksum(C[i, j] * T[i, j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE
    )

    # Row constraints: sum_j T[i,j] = p[i]
    row_constrs = []
    for i in range(n):
        c = model.addConstr(
            gp.quicksum(T[i, j] for j in range(n)) == p[i],
            name=f"row_{i}"
        )
        row_constrs.append(c)

    # Column constraints: sum_i T[i,j] = q[j]
    col_constrs = []
    for j in range(n):
        c = model.addConstr(
            gp.quicksum(T[i, j] for i in range(n)) == q[j],
            name=f"col_{j}"
        )
        col_constrs.append(c)

    # Solve
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed with status {model.Status}")

    T_np = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            T_np[i, j] = T[i, j].X

    T_torch = torch.tensor(T_np, device=device, dtype=dtype)
    obj = (cost * T_torch).sum()

    # Extract dual variables
    try:
        u = torch.tensor(
            np.array([c.Pi for c in row_constrs]),
            device=device,
            dtype=dtype
        )
        v = torch.tensor(
            np.array([c.Pi for c in col_constrs]),
            device=device,
            dtype=dtype
        )
        duals = (u, v)
    except Exception:
        duals = None

    return T_torch, obj, duals


def lp_maximization( # TODO depreciate
    cost: torch.Tensor,
    empirical_distribution: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor
):
    n = cost.shape[0]

    # Decision variables
    Pi = cp.Variable((n, n), nonneg=True)
    w = cp.Variable(n)

    objective = cp.Maximize(cp.sum(cp.multiply(cost, Pi)))

    constraints = [
        cp.sum(Pi, axis=0) == empirical_distribution,
        cp.sum(Pi, axis=1) == w,
        w >= lower, w <= upper,
        cp.sum(w) == 1
    ]

    for i in range(n):
        constraints.append(Pi[i, i] >= lower[i])

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver status: {prob.status}")

    return prob.value, w.value


def sample_vertex(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor: # TODO robustify, kill after N attempts
    n = lower.numel()

    while True:
        # Choose one free variable index
        free_idx = torch.randint(0, n, (1,)).item()

        # Randomly assign lower or upper bounds to others
        x = torch.where(torch.rand(n) < 0.5, lower, upper)

        # Compute value needed for sum(x)=1
        residual = 1.0 - (x.sum() - x[free_idx])

        # Set free variable
        x[free_idx] = residual

        # Check feasibility
        if lower[free_idx] <= x[free_idx] <= upper[free_idx]:
            return x


def euclidean_projection_to_vertex(
    w: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 1000
):

    # Start with all lower bounds
    w_proj = lower.clone()
    total = lower.sum()
    remaining = 1.0 - total

    if remaining < -tol:
        raise ValueError("Lower bounds already sum to more than 1, thus infeasible.")

    if abs(remaining) < tol:
        return w_proj

    # Sort indices by how much w prefers to go up
    order = torch.argsort(w - lower, descending=True)

    # Incrementally raise values toward upper bounds
    for k in order:
        cap = (upper[k] - lower[k]).item()
        if remaining > cap + tol:
            # Fill this variable fully to its upper bound
            w_proj[k] = upper[k]
            remaining -= cap
        else:
            w_proj[k] = lower[k] + remaining
            break

    # Clip for numerical safety
    w_proj = torch.clamp(w_proj, lower, upper)

    assert (w_proj >= 0.0 - tol).all()
    assert 1.0 - tol <= w_proj.sum() <= 1.0 + tol
    assert (w_proj - lower >= -tol).all()
    assert (w_proj - upper <= tol).all()

    return w_proj