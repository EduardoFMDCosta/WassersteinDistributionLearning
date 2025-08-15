import torch
import numpy as np
from scipy.optimize import linprog

TOL = 1e-6

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

    assert 1.0 - TOL <= p.sum() <= 1.0 + TOL
    assert (p >= lower - TOL).all() & (p <= upper + TOL).all()

    result = torch.einsum('i,i->', cost, p)
    return result, p

def project_to_subspace(w: torch.Tensor,
                        lower: torch.Tensor,
                        upper: torch.Tensor,
                        tol: float = 1e-10,
                        max_iter: int = 1000):

    y = torch.clamp(w, min=lower, max=upper)
    s = y.sum().item()
    if abs(s - 1.0) <= tol:
        return y

    # lower/upper bounds for lambda such that S(low)=sum(b) and S(high)=sum(a)
    # Using scalars (float) for the bisection endpoints.
    low = float(torch.min(w - upper).item())   # yields S(low) = sum(b)
    high = float(torch.max(w - lower).item())  # yields S(high) = sum(a)

    # bisection loop
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        # compute clipped values and their sum
        y = torch.clamp(w - mid, min=lower, max=upper)
        s = y.sum().item()

        if abs(s - 1.0) <= tol:
            return y
        if s > 1.0:
            # need to reduce lambda to make sum smaller? note: S(lambda) is decreasing in lambda
            # if s > 1 => mid is too small (y too big) => move low up
            low = mid
        else:
            high = mid

    # fallback
    return torch.clamp(w - mid, min=lower, max=upper)

def gradient_step(
        w: torch.Tensor,
        alpha: torch.Tensor,
        iteration: int,
        lr: float,
        **kwargs):

    learning_rate = lr / (iteration + 1) # square-summable but not summable

    if iteration < 100:
        learning_rate = lr # TODO: IMPROVE LR SCHEDULING

    return w + learning_rate * alpha # TODO: CHECK IF IT SHOULD BE MINUS OR PLUS

def solve_lin_prog(
    cost: torch.Tensor,
    w: torch.Tensor,
    empirical_distribution: torch.Tensor,
    method: str = "highs",
    tol: float = 1e-9):

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

def max_oracle_gradient_descent(cost: torch.Tensor,
                                lower: torch.Tensor,
                                upper: torch.Tensor,
                                empirical_marginal: torch.Tensor,
                                num_steps: int,
                                lr: float,
                                tol: float):
    # See Algorithm 1 in Goktas, Greenwald (2021): https://proceedings.neurips.cc/paper/2021/hash/174a61b0b3eab8c94e0a9e78b912307f-Abstract.html

    # Initialize w
    w = torch.distributions.Dirichlet(torch.ones(cost.shape[0])).sample()

    previous_obj = None

    for step in range(num_steps):
        # Solve for primal and dual (lines 2 and 3)
        Pi, objective, duals = solve_lin_prog(cost, w, empirical_marginal)
        alpha, beta = duals

        # Gradient step (line 4)
        w = gradient_step(w=w, alpha=alpha, iteration=step, lr=lr)
        w = project_to_subspace(w=w, lower=lower, upper=upper)

        # Check for convergence
        if previous_obj is not None and abs(objective - previous_obj) < tol:
            print(f'Stackelberg equilibrium found after {step + 1} iterations.')
            break
        previous_obj = objective

    assert 1.0 - TOL <= w.sum() <= 1.0 + TOL
    assert (w>=lower).all() & (w<=upper).all()

    Pi, objective, duals = solve_lin_prog(cost, w, empirical_marginal)

    return objective

def max_min_lp(cost: torch.Tensor,
               lower: torch.Tensor,
               upper: torch.Tensor,
               empirical_marginal: torch.Tensor,
               method: str,
               num_steps=1000,
               lr=1e-2,
               tol=1e-6):
    if method == 'stackelberg_equilibrium':
        return max_oracle_gradient_descent(cost=cost,
                                           lower=lower,
                                           upper=upper,
                                           empirical_marginal=empirical_marginal,
                                           num_steps=num_steps,
                                           lr=lr,
                                           tol=tol)
    else:
        raise ValueError('Unknown optimization method.')