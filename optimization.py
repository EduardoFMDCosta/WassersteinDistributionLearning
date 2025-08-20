from typing import Callable, Optional, Tuple
import torch
import numpy as np
from scipy.optimize import linprog
import ot


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

def project_to_gamma_subspace(
        Pi: torch.Tensor,
        w: torch.Tensor,
        empirical_marginal: torch.Tensor,
        max_iters: int = 100,
        tol: float = 1e-8
) -> torch.Tensor:

    n = Pi.shape[0]

    # Ensure strictly positive entries to avoid division by 0
    K = Pi.clamp_min(1e-12)

    u = torch.ones(n)
    v = torch.ones(n)

    for _ in range(max_iters):
        u_prev = u
        u = w / (K @ v)
        v = empirical_marginal / (K.t() @ u)

        # check convergence on rows
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    Pi = torch.diag(u) @ K @ torch.diag(v)

    assert (Pi >= 0).all(), "Projection failed: negative entries found."
    assert abs(Pi.sum() - 1.0) <= tol, "Projection failed: total proability mass not equal to one"
    assert torch.allclose(Pi.sum(dim=0), empirical_marginal, atol=tol), "Projection failed: marginal mismatch."

    return Pi


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

    objective = float((-cost * T).sum().item())

    # assert not alpha.isinf().any() and not alpha.isnan().any() and not beta.isinf().any() and not beta.isnan().any()

    return T, objective, (alpha, beta)

def max_oracle_gradient_descent(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
        num_steps: int,
        lr: float,
        ot_solver: Callable
):
    # See Algorithm 1 in Goktas, Greenwald (2021): https://proceedings.neurips.cc/paper/2021/hash/174a61b0b3eab8c94e0a9e78b912307f-Abstract.html

    # Store quantities of interest
    result = {}

    # Initialize w
    w = torch.distributions.Dirichlet(torch.ones(cost.shape[0])).sample()
    w = project_to_omega_subspace(w=w, lower=lower, upper=upper, max_iter=10_000)

    # Initialize trackers for best solution (objective is minimized here before sign flip)
    best_objective = float('inf')
    best_w = w.clone()

    for step in range(num_steps):
        # Solve for primal and dual (lines 2 and 3)
        Pi, objective, duals = ot_solver(cost, w, empirical_marginal)
        alpha, beta = duals

        # Check for best value
        if objective < best_objective:
            best_objective = objective
            best_w = w.clone()

        # Gradient step (line 4)
        w = gradient_step(w=w, alpha=alpha, iteration=step, lr=lr)
        w = project_to_omega_subspace(w=w, lower=lower, upper=upper)

    result["final_w"] = best_w
    result["objective_value"] = best_objective * (-1) # as we solve for f = -h

    return result

def f(Pi: torch.Tensor, cost: torch.Tensor):
    return -(cost * Pi).sum()

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
        tol: float = 1e-8
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

    previous_loss = None  # For stopping criterion based on successive loss differences

    for step in range(num_steps):

        # Phase 1: Gradient ascent for Pi
        for _ in range(10):
            Pi_optimizer.zero_grad()
            loss = f(Pi=Pi, cost=cost)
            (-loss).backward()
            Pi_optimizer.step()

            # Projection
            with torch.no_grad():
                projected = project_to_gamma_subspace(Pi=Pi, w=w, empirical_marginal=empirical_marginal)
                Pi.copy_(projected)

        # Phase 2: Gradient descent for lambd
        for _ in range(10):
            lambd_optimizer.zero_grad()
            loss = lagrangian(w=w, Pi=Pi, lambd=lambd, cost=cost)
            loss.backward()
            lambd_optimizer.step()

        # Phase 3: Gradient descent for w
        w_optimizer.zero_grad()
        loss = lagrangian(w=w, Pi=Pi, lambd=lambd, cost=cost)
        loss.backward()
        w_optimizer.step()

        # Projection
        with torch.no_grad():
            w = project_to_omega_subspace(w=w, lower=lower, upper=upper)

        # Check convergence
        with torch.no_grad():
            # Stopping criterion: absolute change in objective below threshold
            if previous_loss is not None and abs((loss.item() - previous_loss)) < tol:
                print(f"Gradient ascent-descent converged after {step + 1} iterations.")
                break
            previous_loss = loss.item()

    for _ in range(1000):
        Pi_optimizer.zero_grad()
        loss = f(Pi=Pi, cost=cost)
        (-loss).backward()
        Pi_optimizer.step()

        # Projection
        with torch.no_grad():
            projected = project_to_gamma_subspace(Pi=Pi, w=w, empirical_marginal=empirical_marginal)
            Pi.copy_(projected)

    # Compute exact value
    result["final_w"] = w
    result["objective_value"] = -f(Pi=Pi, cost=cost) # as we solve for f = -h

    return result

def max_min_lp(
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
        method: str,
        num_steps=1000,
        lr=1e-3
):
    if method == 'stackelberg_equilibrium':
        result = max_oracle_gradient_descent(
            cost=cost,
            lower=lower,
            upper=upper,
            empirical_marginal=empirical_marginal,
            num_steps=num_steps,
            lr=lr,
            ot_solver=ot_lp_solver
        )
        return result["objective_value"]
    elif method == 'nested_gradient_descent':
        result = nested_gradient_descent(
            cost=cost,
            lower=lower,
            upper=upper,
            empirical_marginal=empirical_marginal,
            num_steps=num_steps,
            lr=lr
        )
        return result["objective_value"]
    elif method == 'sinkhorn':
        result = max_oracle_gradient_descent(
            cost=cost,
            lower=lower,
            upper=upper,
            empirical_marginal=empirical_marginal,
            num_steps=num_steps,
            lr=lr,
            ot_solver=ot_sinkhorn_solver
        )
        return result["objective_value"]
    else:
        raise ValueError('Unknown optimization method.')