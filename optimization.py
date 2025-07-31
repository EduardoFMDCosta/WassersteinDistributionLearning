import torch


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

    result = torch.einsum('i,i->', cost, p)
    return result, p

def max_min_lp(cost: torch.Tensor,
               lower: torch.Tensor,
               upper: torch.Tensor,
               empirical_marginal: torch.Tensor,
               method: str,
               num_steps=1000,
               lr=1e-2,
               tol=1e-6):
    if method == 'dual_sinkhorn':
        return max_min_lp_dual(cost=cost,
                              lower=lower,
                              upper=upper,
                              empirical_marginal=empirical_marginal,
                              num_steps=num_steps,
                              lr=lr,
                              tol=tol)
    else:
        raise ValueError('Unknown optimization method.')

def sinkhorn(cost, a, b, reg=1e-8, max_iter=1000, tol=1e-9):

    n, m = cost.shape

    # Compute the Gibbs kernel: K = exp(-cost/ε)
    K = torch.exp(-cost / reg)

    # Initialize scaling factors u and v
    u = torch.ones(n, device=cost.device, dtype=cost.dtype) / n
    v = torch.ones(m, device=cost.device, dtype=cost.dtype) / m

    # Sinkhorn iterations
    for i in range(max_iter):
        u_prev = u.clone()
        # Update v such that K^T * u * v = b
        v = b / (K.t() @ u + 1e-16)
        # Update u such that K * v * u = a
        u = a / (K @ v + 1e-16)

        # Check for convergence
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    # Recover dual potentials (up to an additive constant)
    # Note: log(·) is computed element-wise
    f = reg * torch.log(u + 1e-16)
    g = reg * torch.log(v + 1e-16)

    return f, g

def max_min_lp_dual(cost: torch.Tensor,
                    lower: torch.Tensor,
                    upper: torch.Tensor,
                    empirical_marginal: torch.Tensor,
                    num_steps: int,
                    lr: float,
                    tol: float):

    n = cost.shape[0]
    prev_obj = None

    # Initialize dual variables
    alpha = torch.rand(n)

    for step in range(num_steps):

        # Solve maximization for w
        _, w = o_maximization(alpha, lower, upper)

        # Solve maximization for dual variables
        alpha, beta = sinkhorn(cost, w, empirical_marginal)

        objective = torch.dot(alpha, w) + torch.dot(beta, empirical_marginal)
        objective = objective.item()

        # Early stopping condition
        if prev_obj is not None and abs(objective - prev_obj) < tol:
            print(f"Early stopping at step {step} with change {abs(objective - prev_obj):.2e}")
            break
        prev_obj = objective

        # Monitor every 100 steps
        if step % 50 == 0 or step == num_steps - 1:
            print(f"Step {step}: objective = {objective:.8f}")

    return objective
