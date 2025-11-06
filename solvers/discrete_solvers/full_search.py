import torch
import warnings
import itertools

from solvers.templates import DiscreteSolver, DiscreteResult
from optimization_utils import ot_lp_solver

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


class FullSearch(DiscreteSolver):
    def __init__(self, max_vertices: int = 1000):
        super().__init__()
        self.max_vertices = max_vertices
    
    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
    ) -> DiscreteResult:
        vertices = get_omega_space_vertices(lower=lower, upper=upper, max_vertices=self.max_vertices)

        objective_opt = -float("inf")
        w_opt = None

        for w in vertices:
            Pi, objective, duals = ot_lp_solver(cost=cost, w=w, empirical_distribution=empirical_marginal)

            # Update highest objective
            if objective_opt < objective:
                objective_opt = objective
                w_opt = w

        return DiscreteResult(bound=torch.as_tensor(objective_opt).pow(1 / self.wasserstein_order), w_opt=w_opt)