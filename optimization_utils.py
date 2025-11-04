from typing import Tuple, Optional

import torch
import numpy as np

import cvxpy as cp
from scipy.optimize import linprog


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
    method: str = "highs",
    tol: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    n = cost.shape[0]

    # Move to CPU/NumPy for the solver
    C_np = cost.detach().cpu().double().numpy()
    p_np = w.detach().cpu().double().numpy()
    q_np = empirical_distribution.detach().cpu().double().numpy()

    # normalize
    p_np /= p_np.sum()
    q_np /= q_np.sum()

    # Decision variable is vec(T) of length n*n in row-major order
    c = C_np.reshape(-1)

    # Equality constraints: A_eq x = b_eq
    # Row sums: for each i, sum_j T[i,j] = p[i]
    A_eq_rows = []
    b_eq = []

    # Row constraints
    for i in range(n):
        row = np.zeros(n*n, dtype=float)
        row[i*n:(i+1)*n] = 1.0
        A_eq_rows.append(row)
        b_eq.append(p_np[i])

    # Column constraints
    for j in range(n):
        col = np.zeros(n*n, dtype=float)
        col[j::n] = 1.0
        A_eq_rows.append(col)
        b_eq.append(q_np[j])

    A_eq = np.stack(A_eq_rows, axis=0)
    b_eq = np.array(b_eq, dtype=float)

    # Bounds: T[i,j] >= 0 (no upper bound)
    bounds = [(-tol, None)] * (n*n)

    # Solve LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)

    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    x = res.x  # optimal vec(T)
    T_np = x.reshape(n, n)

    # Convert back to torch on original device/dtype
    T = torch.tensor(T_np)
    obj = (cost * T).sum()

    # Try to return dual potentials (u for rows, v for cols) if provided
    u = v = None
    try:
        # SciPy (HiGHS) exposes equality marginals; the first n are row constraints, next n columns
        eq_duals = res.eqlin.marginals  # length 2n
        u = torch.tensor(eq_duals[:n])
        v = torch.tensor(eq_duals[n:])
    except Exception:
        pass

    return T, obj, (u, v) if (u is not None and v is not None) else None


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


def sample_vertices(lower: torch.Tensor, upper: torch.Tensor, num_vertices: int) -> torch.Tensor:
    n = lower.numel()
    assert torch.all(lower <= upper), "Each lower bound must be <= upper bound"

    # Expand bounds
    lower_b = lower.unsqueeze(0).expand(num_vertices, n)
    upper_b = upper.unsqueeze(0).expand(num_vertices, n)

    while True:
        # Randomly pick free index for each vertex
        free_idx = torch.randint(0, n, (num_vertices,))

        # Randomly select lower or upper for all coords
        use_upper = (torch.rand(num_vertices, n) < 0.5)
        x = torch.where(use_upper, upper_b, lower_b)

        # Compute required residuals for sum = 1
        row_sum = x.sum(dim=1)
        residuals = 1.0 - (row_sum - x[torch.arange(num_vertices), free_idx])

        # Set the free variable
        x[torch.arange(num_vertices), free_idx] = residuals

        # Check feasibility (vectorized)
        ok = (x >= lower_b).all(dim=1) & (x <= upper_b).all(dim=1)

        if ok.all():
            return x
        else:
            # Resample only failed rows
            failed = ~ok
            n_failed = failed.sum().item()
            if n_failed > 0:
                x[failed] = sample_vertices(lower, upper, n_failed)
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