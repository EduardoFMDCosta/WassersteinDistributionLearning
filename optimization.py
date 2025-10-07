import math
import collections
from typing import Callable, Tuple, Optional

import torch
import ot
from tqdm import tqdm
import warnings
import itertools
import numpy as np

import cvxpy as cp
from gurobipy import GRB, QuadExpr
from scipy.optimize import linprog

from plotting.plot import plot_optimization_curves

try:
    import gurobipy as gp
except:
    gp = None


def o_maximization(
    cost: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor, 
    tol: float = 1e-6
):
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

def project_to_omega_subspace(
    w: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 1000
):
    """Project a vector onto the capped probability simplex.

    Solve:  minimize ||y - w||_2  subject to  lower <= y <= upper (elementwise), sum(y)=1.

    The KKT conditions imply the solution has the form
        y_i = clip(w_i - λ, lower_i, upper_i)
    for a single scalar Lagrange multiplier λ enforcing the sum constraint. Define
        S(λ) = sum_i clip(w_i - λ, lower_i, upper_i).
    S is continuous and strictly decreasing in λ, so we locate λ* with bisection.

    Parameters
    ---------
    w : torch.Tensor (n,)
        Point to project.
    lower, upper : torch.Tensor (n,)
        Elementwise bounds (must satisfy lower <= upper and
        sum(lower) <= 1 <= sum(upper) for feasibility).
    tol : float
        Absolute tolerance on the sum constraint.
    max_iter : int
        Maximum bisection iterations.
    """

    # First clamp into the box; if the sum hits 1 already we're done.
    y = torch.clamp(w, min=lower, max=upper)
    s = y.sum().item()
    if abs(s - 1.0) <= tol:
        final_y = y
    else:
        # Establish bisection interval [low, high] for λ.
        # Let low = min_i (w_i - upper_i). Then for every i: w_i - low >= upper_i, so after clipping y = upper and S(low)=sum(upper).
        # Let high = max_i (w_i - lower_i). Then for every i: w_i - high <= lower_i, so after clipping y = lower and S(high)=sum(lower).
        # Feasibility (sum(lower) <= 1 <= sum(upper)) guarantees the root λ* with S(λ*)=1 lies in [low, high].
        low = float(torch.min(w - upper).item())    # S(low) = sum(upper) ≥ 1
        high = float(torch.max(w - lower).item())   # S(high) = sum(lower) ≤ 1

        # Adaptive effective tolerance: don't demand more than floating precision permits.
        effective_tol = max(tol, torch.finfo(w.dtype).eps * w.numel())

        final_y = None
        best_y = None
        best_res = float('inf')

        # Bisection on S(λ) - 1 = 0. S decreases with λ.
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            y = torch.clamp(w - mid, min=lower, max=upper)
            s = y.sum().item()
            res = abs(s - 1.0)

            # Track best iterate always
            if res < best_res:
                best_res = res
                best_y = y

            # Terminate on residual or bracket size.
            if res <= effective_tol:
                final_y = y
                break

            if s > 1.0:
                # Sum too large ⇒ λ too small ⇒ increase lower bound
                low = mid
            else:
                # Sum too small ⇒ λ too large ⇒ decrease upper bound
                high = mid

        if final_y is None:
            # Fall back to best iterate obtained
            final_y = best_y
            # Only raise if we are *far* outside user-requested tolerance.
            if best_res > effective_tol:
                raise RuntimeError(
                    f"Bisection did not reach tolerance: residual={best_res:.3e}, interval=({low:.3e},{high:.3e}), tol={tol:.1e}, effective_tol={effective_tol:.1e}"
                )

        # Use effective_tol for internal assertion; still ensure within a modest multiple of user tol
        assert abs(final_y.sum() - 1.0) <= effective_tol
        assert (final_y >= lower).all() & (final_y <= upper).all() 

    return final_y

def ot_lp_solver(
    cost: torch.Tensor,
    w: torch.Tensor,
    empirical_distribution: torch.Tensor,
    method: str = "highs",
    tol: float = 1e-8
):
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
    obj = float((cost * T).sum().item())

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

def ot_sinkhorn_solver(
    cost: torch.Tensor,
    w: torch.Tensor,
    empirical_distribution: torch.Tensor,
    epsilon: float = 1e-3,
    max_iter: int = 1000,
    tol: float = 1e-5,
    method: str = 'sinkhorn_stabilized'
) -> Tuple[torch.Tensor, float, Tuple[torch.Tensor, torch.Tensor]]:
    """Entropic OT solver (POT stabilized Sinkhorn) matching solve_lin_prog interface.

    Returns transport plan T, objective = (-cost * T).sum() for consistency
    with solve_lin_prog, and dual-like potentials (alpha, beta) derived from
    scaling vectors. Alpha/beta are epsilon * log(u/v) and defined up to an
    additive constant.
    """
    assert method in ['sinkhorn_stabilized', 'sinkhorn_log']
    assert cost.dim() == 2 and cost.shape[0] == cost.shape[1], "cost must be square"
    n = cost.shape[0]
    assert w.shape == (n,) and empirical_distribution.shape == (n,), "marginals must match cost dimension"

    C_np = cost.detach().cpu().double().numpy()
    a_np = w.detach().cpu().double().numpy()
    b_np = empirical_distribution.detach().cpu().double().numpy()

    # Stabilized log-domain sinkhorn
    T_np, log = ot.sinkhorn(a_np, b_np, C_np, reg=epsilon, numItermax=max_iter, stopThr=tol, method=method, log=True)

    T = torch.from_numpy(T_np).to(device=cost.device, dtype=cost.dtype)
    logu = torch.from_numpy(log[f"log{'_' if 'log' in method else ''}u"]).to(cost.device, cost.dtype)
    logv = torch.from_numpy(log[f"log{'_' if 'log' in method else ''}v"]).to(cost.device, cost.dtype)

    alpha = epsilon * logu
    beta = epsilon * logv

    objective = float((cost * T).sum().item())

    # assert not alpha.isinf().any() and not alpha.isnan().any() and not beta.isinf().any() and not beta.isnan().any()

    return T, objective, (alpha, beta)

def get_vertices(
    lower: torch.Tensor,
    upper: torch.Tensor,
    max_vertices: int = 1000,
    tol: float = 1e-7
):
    M = lower.shape[0]
    vertices = []

    # Loop: choose which coordinate is solved from the sum constraint
    for free_idx in range(M):
        # All others are clamped to either lower or upper
        fixed_indices = [i for i in range(M) if i != free_idx]
        for pattern in itertools.product([0, 1], repeat=M - 1):  # 0 -> lower, 1 -> upper
            w = torch.empty(M, dtype=lower.dtype)

            # Assign bounds to fixed coords
            for idx, choice in zip(fixed_indices, pattern):
                w[idx] = lower[idx] if choice == 0 else upper[idx]

            # Solve for the free coordinate
            remaining = 1.0 - w[fixed_indices].sum()
            w[free_idx] = remaining

            # Check feasibility
            if lower[free_idx] <= w[free_idx] <= upper[free_idx]:
                if abs(w.sum() - 1.0) <= tol and (w >= lower - tol).all() and (w <= upper + tol).all():
                    vertices.append(w)
                if len(vertices) > max_vertices:
                    warnings.warn("Maximum number of vertices reached. Full Search will return a lower bound.", UserWarning)
                    return vertices

    return vertices

def get_omega_space_vertices(
    lower: torch.Tensor,
    upper: torch.Tensor,
    max_vertices: int = 1000,
    tol: float = 1e-7
):
    vertices = get_vertices(lower, upper, max_vertices, tol)
    if vertices:
        return torch.stack(vertices, dim=0)
    else:
        return torch.empty((0, lower.shape[0]), dtype=lower.dtype)

def full_search(
    cost: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    empirical_marginal: torch.Tensor,
    ot_solver: Callable
):
    # Store quantities of interest
    result = {}

    vertices = get_omega_space_vertices(lower=lower, upper=upper)

    objective_opt = -float("inf")
    w_opt = None

    for w in vertices:
        Pi, objective, duals = ot_solver(cost=cost, w=w, empirical_distribution=empirical_marginal)

        # Update highest objective
        if objective_opt < objective:
            objective_opt = objective
            w_opt = w

    result["w_opt"] = w_opt
    result["objective_opt"] = objective_opt
    return result


def cutting_plane(
    cost: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    empirical_marginal: torch.Tensor,
    num_steps: int,
    ot_solver: Callable
):
    M = cost.shape[0]
    delta = 1e-3

    objective_opt = -float("inf")
    w_opt = None

    pbar = tqdm(total=num_steps, desc="Cutting Plane Outer Loop")

    for d in range(num_steps):
        msg = f"[CuttingPlane] iteration={d}"
        pbar.set_postfix_str(msg)

        # Initialize w
        w = empirical_marginal.clone() + torch.rand_like(empirical_marginal) * delta
        w = project_to_omega_subspace(w=w, lower=lower, upper=upper, max_iter=10_000)

        for step in range(num_steps):
            # Solve for primal (w^{(k)}) and dual (alpha^{(k)}, beta^{(k)})
            Pi, objective, duals = ot_solver(cost=cost, w=w, empirical_distribution=empirical_marginal)
            alpha, beta = duals

            alpha = alpha.float()
            beta = beta.float()

            # O-maximization (w^{(k+1)})
            _, w_next = o_maximization(cost=alpha, lower=lower, upper=upper)

            epsilon = torch.einsum('i,i->', alpha, w_next - w)
            msg_inner = f"[Inner] step={step+1}, epsilon={epsilon:.2e}, objective={objective:.4f}"
            pbar.set_postfix_str(msg + ", " + msg_inner)

            if epsilon < 1e-5:
                pbar.write(f"{msg} converged after {step + 1} iterations. Final objective: {objective:.4f}")
                break

            # Update w
            w = w_next

        #Update highest objective
        if objective_opt < objective:
            objective_opt = objective
            w_opt = w

        pbar.update(1)

    pbar.close()

    return dict(
        w_opt=w_opt,
        objective_opt=objective_opt,
        alpha=alpha,
        beta=beta
    )

def plain_vanilla(
    cost: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    empirical_marginal: torch.Tensor
):
    # See Corollary 6.2 in

    upper_diff = upper - empirical_marginal
    lower_diff = empirical_marginal - lower

    max_prob_diff = torch.max(upper_diff, lower_diff)
    max_dist, _ = torch.max(cost, dim=1)

    return dict(
        w_opt=None, 
        objective_opt=torch.einsum('i,i->', max_dist, max_prob_diff)
    )

def lp_maximization(
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


def solve_milp_min_diagonal_cvxpy(
    cost: torch.Tensor, 
    empirical_distribution: torch.Tensor, 
    lower: torch.Tensor, 
    upper: torch.Tensor
):
    n = len(empirical_distribution)

    # Decision variables
    Pi = cp.Variable((n, n), nonneg=True)
    w = cp.Variable(n)
    m = cp.Variable(n)
    b = cp.Variable(n, boolean=True)

    objective = cp.Maximize(cp.sum(cp.multiply(cost, Pi)))
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

    return prob.value, w.value

def solve_milp_min_diagonal_gurobi(
    cost: torch.Tensor,
    empirical_distribution: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[float, torch.Tensor]:
    device, dtype = cost.device, cost.dtype

    # Convert to NumPy for Gurobi
    c = cost.detach().cpu().numpy()
    p = empirical_distribution.detach().cpu().numpy()
    lo = lower.detach().cpu().numpy()
    up = upper.detach().cpu().numpy()

    n = int(p.size)
    if c.shape != (n, n):
        raise ValueError(f"cost must be (n,n); got {c.shape}")
    if lo.shape != (n,) or up.shape != (n,):
        raise ValueError("lower/upper must be 1D of length n")
    M = up - lo
    if (M < 0).any():
        raise ValueError("upper must be >= lower componentwise")

    m = gp.Model("min_diagonal_milp")
    if not verbose:
        m.setParam("OutputFlag", 0)
    if time_limit is not None:
        m.setParam("TimeLimit", float(time_limit))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))

    # Variables
    Pi = m.addVars(n, n, lb=0.0, vtype=GRB.CONTINUOUS, name="Pi")
    w  = m.addVars(n, lb=lo, ub=up, vtype=GRB.CONTINUOUS, name="w")
    mm = m.addVars(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="m")
    b  = m.addVars(n, vtype=GRB.BINARY, name="b")

    # Objective: sum_{i,j} c[i,j] * Pi[i,j]
    m.setObjective(gp.quicksum(c[i, j] * Pi[i, j] for i in range(n) for j in range(n)), GRB.MAXIMIZE)

    # Column sums: sum_i Pi[i,j] == p[j]
    m.addConstrs(
        (gp.quicksum(Pi[i, j] for i in range(n)) == float(p[j]) for j in range(n)),
        name="col_sums"
    )

    # Row sums: sum_j Pi[i,j] == w[i]
    m.addConstrs(
        (gp.quicksum(Pi[i, j] for j in range(n)) == w[i] for i in range(n)),
        name="row_sums"
    )

    # Big-M min linearization (elementwise)
    m.addConstrs((mm[i] <= w[i] for i in range(n)), name="m_le_w")
    m.addConstrs((mm[i] <= float(p[i]) for i in range(n)), name="m_le_p")
    m.addConstrs((mm[i] >= w[i] - float(M[i]) * (1 - b[i]) for i in range(n)), name="m_ge_w_minus_M")
    m.addConstrs((mm[i] >= float(p[i]) - float(M[i]) * b[i] for i in range(n)), name="m_ge_p_minus_Mb")

    # Diagonal constraint: Pi[i,i] >= mm[i]
    m.addConstrs((Pi[i, i] >= mm[i] for i in range(n)), name="diag_ge_m")

    # Sum w == 1
    m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0, name="sum_w_eq_1")

    # Optimize
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi ended with status {m.Status}")

    obj_val = float(m.ObjVal)

    # Extract w as a torch tensor on the original device/dtype
    w_sol_np = [w[i].X for i in range(n)]
    w_sol = torch.tensor(w_sol_np, device=device, dtype=dtype)

    return obj_val, w_sol

def solve_milp_min_diagonal(
    cost: torch.Tensor, 
    empirical_distribution: torch.Tensor, 
    lower: torch.Tensor, 
    upper: torch.Tensor, 
    **kwargs
):
    if gp is None:
        return solve_milp_min_diagonal_cvxpy(cost, empirical_distribution, lower, upper)
    else:
        return solve_milp_min_diagonal_gurobi(cost, empirical_distribution, lower, upper, **kwargs)
    

def diagonal_constrained_tp(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor, 
        **kwargs
):
    # See Section 6.1. in

    objective, w = solve_milp_min_diagonal(cost=cost, empirical_distribution=empirical_marginal, lower=lower, upper=upper, **kwargs)

    return dict(
        w_opt=w,
        objective_opt=objective
    )

def anchor(alpha, beta):
    return alpha + beta[0], beta - beta[0]

def project_alpha_beta(alpha0, beta0, C, verbose=False):
    alpha0 = np.asarray(alpha0)
    beta0 = np.asarray(beta0)
    C = np.asarray(C)

    n, m = C.shape
    assert alpha0.shape == (n,)
    assert beta0.shape == (m,)

    # Variables
    alpha = cp.Variable(n)
    beta = cp.Variable(m)

    # Objective: minimize squared distance
    obj = 0.5 * cp.sum_squares(alpha - alpha0) + 0.5 * cp.sum_squares(beta - beta0)

    # Constraints: alpha_i + beta_j <= C_ij for all (i,j)
    constraints = [alpha[i] + beta[j] <= C[i, j] for i in range(n) for j in range(m)]

    # Problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.GUROBI, verbose=verbose)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Projection failed, solver status: {prob.status}")

    alpha = torch.tensor(alpha.value, dtype=torch.float32)
    beta = torch.tensor(beta.value, dtype=torch.float32)

    return alpha, beta

def inner_lp_maximization(
    alpha: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor
):
    M = lower.shape[0]

    # Variables
    lam = cp.Variable()  # scalar λ
    mu = cp.Variable(M, nonneg=True)  # vector μ ≥ 0
    nu = cp.Variable(M, nonneg=True)  # vector ν ≥ 0

    a = lower.detach().cpu().numpy()
    b = upper.detach().cpu().numpy()

    # Constraint λ*1 + μ - ν ≥ α
    ones = np.ones(M)
    constraint = alpha - lam * ones - mu + nu <= 0

    # Problem
    objective = cp.Maximize(-lam - b @ mu + a @ nu)
    prob = cp.Problem(objective, [constraint, mu >= 0, nu >= 0])
    prob.solve(solver=cp.GUROBI, verbose=False)

    y = (torch.tensor(lam.value, dtype=torch.float32), torch.tensor(mu.value, dtype=torch.float32), torch.tensor(nu.value, dtype=torch.float32))
    dual_vector = torch.tensor(constraint.dual_value, dtype=torch.float32)

    return y, dual_vector

def max_oracle_gradient_descent(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
        num_steps: int = 5000,
        tol: float = 1e-3,
        plot: bool = True,
        **kwargs
):

    # See Algorithm 1 in Goktas, Greenwald (2021): https://proceedings.neurips.cc/paper/2021/hash/174a61b0b3eab8c94e0a9e78b912307f-Abstract.html

    M = cost.shape[0]

    def c_transform(alpha):
        return (cost - alpha.unsqueeze(1)).min(dim=0).values

    # Initialize x = (alpha, beta)
    alpha_0 = torch.randn(cost.shape[0])
    alpha = alpha_0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.SGD([alpha], lr=0.5, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    best = float("inf")
    history_len = 10
    recent_values = collections.deque(maxlen=history_len)

    # Logging
    values = []
    best_values = []
    grad_norms = []
    lr_sizes = []

    # Gradient noise settings
    base_noise = 0.0
    max_noise = 0.0 * math.sqrt(M)
    noise_scale = base_noise
    decay_factor = 0.95

    for step in tqdm(range(num_steps)):

        # Solve for primal and dual (lines 2 and 3)
        y, dual_vector = inner_lp_maximization(alpha.clone().detach(), lower, upper)
        lam, mu, nu = y

        def f(alpha):
            beta = c_transform(alpha)
            return -lam - (mu * upper).sum() + (nu * lower).sum() - (beta * empirical_marginal).sum()

        def g(alpha):
            return -(alpha - lam * torch.ones(alpha.shape[0]) - mu + nu)

        def lagrangian(alpha):
            return f(alpha) + torch.dot(dual_vector, g(alpha))

        # Compute current value and store best value
        value = f(alpha).detach().item()
        values.append(value)
        recent_values.append(value)

        if value < best:
            best = value
        best_values.append(best)

        # Gradient step update for x
        optimizer.zero_grad()
        lagrange = lagrangian(alpha)
        lagrange.backward()


        # Log gradient norm and lr
        grad_norm = alpha.grad.detach().norm().item()
        grad_norms.append(grad_norm)

        eta = optimizer.param_groups[0]['lr']
        lr_sizes.append(eta)

        # Randomize gradients
        alpha.grad += noise_scale * torch.randn_like(alpha.grad)
        optimizer.step()
        scheduler.step()

        # Detect stagnation
        if len(recent_values) == history_len:
            if abs(recent_values[0] - recent_values[-1]) < tol:
                # Stuck -> increase gradient noise
                noise_scale = min(noise_scale * 2, max_noise)
            else:
                # Progress -> decay back to base noise
                noise_scale = max(noise_scale * decay_factor, base_noise)

    _, w = o_maximization(alpha, lower, upper)
    objective_value = -best

    if plot:
        plot_optimization_curves(values, best_values, grad_norms, lr_sizes)

    return dict(
        w_opt=w,
        objective_opt=objective_value
    )

def black_box(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
        time_limit: int = 3000,
        **kwargs
):
    M = cost.shape[0]

    # Create model
    model = gp.Model("dual_transport")
    model.setParam("NonConvex", 2)
    model.setParam("TimeLimit", time_limit)

    # Decision variables
    alpha = model.addVars(M, lb=-GRB.INFINITY, name="alpha")
    beta = model.addVars(M, lb=-GRB.INFINITY, name="beta")
    w = model.addVars(M, lb=0.0, name="w")

    # Bounds on w
    for i in range(M):
        w[i].lb = lower[i]
        w[i].ub = upper[i]
    model.addConstr(gp.quicksum(w[i] for i in range(M)) == 1, name="sum_w")

    # Constraints
    model.addConstrs((alpha[i] + beta[j] <= cost[i, j]
                      for i in range(M) for j in range(M)), name="dual_constr")

    model.addConstr(alpha[0] == 0, name="alpha_anchor")

    # Objective
    obj = QuadExpr()
    for i in range(M):
        obj.addTerms(1.0, alpha[i], w[i])
    for j in range(M):
        obj.add(beta[j] * empirical_marginal[j])
    model.setObjective(obj, GRB.MAXIMIZE)

    # Solve
    model.optimize()

    # Extract solution
    if model.status == GRB.OPTIMAL  or model.status == GRB.SUBOPTIMAL or model.status == GRB.TIME_LIMIT:
        w_opt = torch.tensor([w[i].X for i in range(M)])
        objective_value = model.ObjVal
        return dict(
            w_opt=w_opt,
            objective_opt=objective_value
        )
    else:
        return None

def max_min_lp(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
        method: str,
        num_steps=1000,
        lr=1e-3
):
    if method == 'full_search':
        result = full_search(
            cost=cost,
            lower=lower,
            upper=upper,
            empirical_marginal=empirical_marginal,
            ot_solver=ot_lp_solver
        )
        return result["objective_opt"]
    elif method == 'cutting_plane':
        result = cutting_plane(
            cost=cost,
            lower=lower,
            upper=upper,
            empirical_marginal=empirical_marginal,
            num_steps=num_steps,
            ot_solver=ot_lp_solver
        )
        return result["objective_opt"]
    elif method == 'plain_vanilla':
        result = plain_vanilla(
            cost=cost,
            lower=lower,
            upper=upper,
            empirical_marginal=empirical_marginal
        )
        return result["objective_opt"]
    elif method == 'diagonal_constrained_tp':
        result = diagonal_constrained_tp(
            cost=cost,
            lower=lower,
            upper=upper,
            empirical_marginal=empirical_marginal
        )
        return result["objective_opt"]
    elif method == 'max_oracle_gradient_descent':
        result = max_oracle_gradient_descent(
            cost=cost,
            lower=lower,
            upper=upper,
            empirical_marginal=empirical_marginal
        )
        return result["objective_opt"]
    elif method == 'black_box':
        result = black_box(
            cost=cost,
            lower=lower,
            upper=upper,
            empirical_marginal=empirical_marginal
        )
        return result["objective_opt"]
    else:
        raise ValueError('Unknown optimization method.')