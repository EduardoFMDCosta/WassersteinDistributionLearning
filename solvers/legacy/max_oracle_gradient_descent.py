import torch
import cvxpy as cp
import collections
from tqdm import tqdm

from solvers.templates import MaxMinLP, MaxMinLPResult
from optimization_utils import o_maximization


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
    constraint = alpha - lam * torch.ones(M) - mu + nu <= 0

    # Problem
    objective = cp.Maximize(-lam - b @ mu + a @ nu)
    prob = cp.Problem(objective, [constraint, mu >= 0, nu >= 0])
    prob.solve(solver=cp.GUROBI, verbose=False)

    y = (torch.tensor(lam.value, dtype=torch.float32), torch.tensor(mu.value, dtype=torch.float32), torch.tensor(nu.value, dtype=torch.float32))
    dual_vector = torch.tensor(constraint.dual_value, dtype=torch.float32)

    return y, dual_vector

class MaxOracleGradientDescent(MaxMinLP):
    def __init__(self, num_steps: int = 1000):
        super().__init__()
        self.num_steps = num_steps

    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
    ) -> MaxMinLPResult:

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

        for step in tqdm(range(self.num_steps)):

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
            optimizer.step()
            scheduler.step()

        _, w = o_maximization(alpha, lower, upper)
        objective_value = -best

        return MaxMinLPResult(objective_opt=objective_value, w_opt=w)