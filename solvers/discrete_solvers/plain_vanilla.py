import torch

from solvers.templates import DiscreteResult, DiscreteSolver


class PlainVanilla(DiscreteSolver):
    def __init__(self):
        super().__init__()

    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor
    ) -> DiscreteResult:
        # See Corollary 6.2 in

        upper_diff = upper - empirical_marginal
        lower_diff = empirical_marginal - lower

        max_prob_diff = torch.max(upper_diff, lower_diff)
        max_dist, _ = torch.max(cost, dim=1)

        return DiscreteResult(bound=torch.einsum('i,i->', max_dist, max_prob_diff).pow(1 / self.wasserstein_order))