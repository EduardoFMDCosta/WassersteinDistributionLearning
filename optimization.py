from typing import Callable
import torch
import numpy as np
from scipy.optimize import linprog
import ot

TOL = 1e-6

def o_maximization(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor
):

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

    assert 1.0 - TOL <= p.sum() <= 1.0 + TOL
    assert (p >= lower - TOL).all() & (p <= upper + TOL).all()

    result = torch.einsum('i,i->', cost, p)
    return result, p

def project_to_omega_subspace(
        w: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        tol: float = 1e-10,
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
        return y

    # Establish bisection interval [low, high] for λ.
    # Let low = min_i (w_i - upper_i). Then for every i: w_i - low >= upper_i, so after clipping y = upper and S(low)=sum(upper).
    # Let high = max_i (w_i - lower_i). Then for every i: w_i - high <= lower_i, so after clipping y = lower and S(high)=sum(lower).
    # Feasibility (sum(lower) <= 1 <= sum(upper)) guarantees the root λ* with S(λ*)=1 lies in [low, high].
    low = float(torch.min(w - upper).item())   # yields S(low) = sum(upper)
    high = float(torch.max(w - lower).item())  # yields S(high) = sum(lower)

    # Bisection on S(λ) - 1 = 0. S decreases with λ.
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        y = torch.clamp(w - mid, min=lower, max=upper)
        s = y.sum().item()

        if abs(s - 1.0) <= tol:
            return y
        if s > 1.0:
            # Current sum too large ⇒ λ too small (need larger λ) ⇒ move lower bound up.
            low = mid
        else:
            # Current sum too small ⇒ λ too large ⇒ move upper bound down.
            high = mid

    # Fallback (max_iter reached): return last approximation.
    return torch.clamp(w - 0.5 * (low + high), min=lower, max=upper)

def project_to_gamma_subspace(Pi: torch.Tensor, empirical_marginal: torch.Tensor):
    n = Pi.size(0)

    # sort each column descending
    U, _ = torch.sort(Pi, dim=0, descending=True)  # (n, n), per-column sort
    cssv = U.cumsum(dim=0) - empirical_marginal.unsqueeze(0)  # (n, n)
    ks = torch.arange(1, n + 1).unsqueeze(1)  # (n,1)
    thetas = cssv / ks  # (n, n)
    cond = (U - thetas) > 0  # (n, n) booleans
    rho = cond.sum(dim=0).clamp(min=1)  # (n,) number of positives per col
    # gather theta at rho-1 for each column
    idx = rho - 1  # (n,)
    theta = thetas.gather(0, idx.unsqueeze(0).expand(1, n)).squeeze(0)  # (n,)
    # project
    return (Pi - theta.unsqueeze(0)).clamp(min=0)


def gradient_step(
        w: torch.Tensor,
        alpha: torch.Tensor,
        iteration: int,
        lr: float,
        **kwargs
):

    #learning_rate = lr / (iteration + 1) # square-summable but not summable
    learning_rate = lr # TODO: IMPROVE GRADIENT STEP SIZE

    return w + learning_rate * alpha

def ot_lp_solver(
        cost: torch.Tensor,
        w: torch.Tensor,
        empirical_distribution: torch.Tensor,
        method: str = "highs",
        tol: float = 1e-9
):

    n = cost.shape[0]

    # Move to CPU/NumPy for the solver
    C_np = cost.detach().cpu().double().numpy()
    p_np = w.detach().cpu().double().numpy()
    q_np = empirical_distribution.detach().cpu().double().numpy()

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
    bounds = [(0.0, None)] * (n*n)

    # Solve LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)

    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    x = res.x  # optimal vec(T)
    T_np = x.reshape(n, n)

    # Convert back to torch on original device/dtype
    T = torch.tensor(T_np)
    obj = float((-cost * T).sum().item())

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
        max_iter: int = 100,
        tol: float = 1e-5,
):
    """Entropic OT solver (POT stabilized Sinkhorn) matching solve_lin_prog interface.

    Returns transport plan T, objective = (-cost * T).sum() for consistency
    with solve_lin_prog, and dual-like potentials (alpha, beta) derived from
    scaling vectors. Alpha/beta are epsilon * log(u/v) and defined up to an
    additive constant.
    """

    assert cost.dim() == 2 and cost.shape[0] == cost.shape[1], "cost must be square"
    n = cost.shape[0]
    assert w.shape == (n,) and empirical_distribution.shape == (n,), "marginals must match cost dimension"

    C_np = cost.detach().cpu().double().numpy()
    a_np = w.detach().cpu().double().numpy()
    b_np = empirical_distribution.detach().cpu().double().numpy()

    # Stabilized log-domain sinkhorn
    T_np, log = ot.sinkhorn(a_np, b_np, C_np, reg=epsilon, numItermax=max_iter,
                             stopThr=tol, method='sinkhorn_log', log=True)

    T = torch.from_numpy(T_np).to(device=cost.device, dtype=cost.dtype)
    # POT's log dict gives scaling factors 'u','v' (not log) for stabilized method
    u = torch.from_numpy(log['u']).to(cost.device, cost.dtype)
    v = torch.from_numpy(log['v']).to(cost.device, cost.dtype)
    alpha = epsilon * torch.log(u.clamp_min(1e-300))
    beta = epsilon * torch.log(v.clamp_min(1e-300))

    objective = float((-cost * T).sum().item())
    return T, objective, (alpha, beta)

def max_oracle_gradient_descent(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
        num_steps: int,
        lr: float,
        tol: float,
        ot_solver: Callable
):
    # See Algorithm 1 in Goktas, Greenwald (2021): https://proceedings.neurips.cc/paper/2021/hash/174a61b0b3eab8c94e0a9e78b912307f-Abstract.html

    # Store quantities of interest
    result = {}

    # Initialize w
    w = torch.distributions.Dirichlet(torch.ones(cost.shape[0])).sample()
    result["initial_w"] = w

    best_objective, best_w = None, None

    for step in range(num_steps):
        # Solve for primal and dual (lines 2 and 3)
        Pi, objective, duals = ot_solver(cost, w, empirical_marginal)
        alpha, beta = duals

        # Check for best value
        if step >= 1 and (best_objective is None or objective < best_objective):
            best_objective = objective
            best_w = w.clone()

        # Gradient step (line 4)
        w = gradient_step(w=w, alpha=alpha, iteration=step, lr=lr)
        w = project_to_omega_subspace(w=w, lower=lower, upper=upper)

        assert 1.0 - TOL <= w.sum() <= 1.0 + TOL
        assert (w>=lower).all() & (w<=upper).all()

    result["final_w"] = best_w
    result["objective_value"] = best_objective * (-1) # as we solve for f = -h

    return result

def f(Pi: torch.Tensor, cost: torch.Tensor):
    return (cost * Pi).sum()

def g(w: torch.Tensor, Pi: torch.Tensor):
    return Pi.sum(dim=1) - w

def lagrangian(w: torch.Tensor, Pi: torch.Tensor, lambd: torch.Tensor, cost: torch.Tensor):
    return f(Pi=Pi, cost=cost) + torch.dot(lambd, g(w=w, Pi=Pi))

def nested_gradient_descent(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
        num_steps: int,
        lr: float,
        tol: float
):
    # See Algorithm 2 in Goktas, Greenwald (2021): https://proceedings.neurips.cc/paper/2021/hash/174a61b0b3eab8c94e0a9e78b912307f-Abstract.html

    # Store quantities of interest
    result = {}
    M = cost.shape[0]

    # Initialize w and Pi
    w = torch.distributions.Dirichlet(torch.ones(M)).sample()
    result["initial_w"] = w
    Pi = torch.outer(w, empirical_marginal).clone().requires_grad_(True)
    lambd = torch.randn(M, requires_grad=True)

    # Optimizers for each variable
    Pi_optimizer = torch.optim.Adam([Pi], lr=lr)
    lambd_optimizer = torch.optim.Adam([lambd], lr=lr)
    w_optimizer = torch.optim.Adam([w], lr=lr)

    for step in range(num_steps):

        prev_L = None
        for _ in range(10):
            Pi_optimizer.zero_grad()
            loss = f(Pi=Pi, cost=cost)
            loss.backward()
            Pi_optimizer.step()

            with torch.no_grad():
                Pi.copy_(project_to_gamma_subspace(Pi=Pi, empirical_marginal=empirical_marginal))

                assert (Pi>=0).all()
                assert 1.0 - TOL <= Pi.sum() <= 1.0 + TOL
                assert (abs(Pi.sum(dim=0) - empirical_marginal) <= TOL).all()

                # Convergence check
                if prev_L is not None and abs((loss.item() - prev_L)) < tol:
                    break
                prev_L = loss.item()

        prev_L = None
        for _ in range(10):
            lambd_optimizer.zero_grad()
            L = lagrangian(w=w, Pi=Pi, lambd=lambd, cost=cost)
            (-L).backward()
            lambd_optimizer.step()

            with torch.no_grad():
                if prev_L is not None and abs((L.item() - prev_L)) < 1e-6:
                    break
                prev_L = L.item()

        w_optimizer.zero_grad()
        L = lagrangian(w=w, Pi=Pi, lambd=lambd, cost=cost)
        (-L).backward()
        w_optimizer.step()

        with torch.no_grad():
            w = project_to_omega_subspace(w=w, lower=lower, upper=upper)

            assert 1.0 - TOL <= w.sum() <= 1.0 + TOL
            assert (w>=lower).all() & (w<=upper).all()


    Pi, objective, duals = ot_lp_solver(cost=cost, w=w, empirical_distribution=empirical_marginal)
    result["final_w"] = w
    result["objective_value"] = objective * (-1) # as we solve for f = -h

    return result

def max_min_lp(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
        method: str,
        num_steps=1000,
        lr=1e-3,
        tol=1e-8
):
    if method == 'stackelberg_equilibrium':
        result = max_oracle_gradient_descent(cost=cost,
                                           lower=lower,
                                           upper=upper,
                                           empirical_marginal=empirical_marginal,
                                           num_steps=num_steps,
                                           lr=lr,
                                           tol=tol,
                                           ot_solver=ot_lp_solver)
        return result["objective_value"]
    elif method == 'nested_gradient_descent':
        result = nested_gradient_descent(cost=cost,
                                         lower=lower,
                                         upper=upper,
                                         empirical_marginal=empirical_marginal,
                                         num_steps=num_steps,
                                         lr=lr,
                                         tol=tol)
        return result["objective_value"]
    elif method == 'sinkhorn':
        result = max_oracle_gradient_descent(cost=cost,
                                           lower=lower,
                                           upper=upper,
                                           empirical_marginal=empirical_marginal,
                                           num_steps=num_steps,
                                           lr=lr,
                                           tol=tol,
                                           ot_solver=ot_sinkhorn_solver)
        return result["objective_value"]
    else:
        raise ValueError('Unknown optimization method.')