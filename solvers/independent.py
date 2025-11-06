import torch

from optimization_utils import o_maximization
from quantization import UncertainQuantization

from solvers.templates import Solver, Result, DiscreteSolver


class IndependentSolver(Solver):
    def __init__(
        self, 
        discrete_solver: DiscreteSolver,
        
    ) -> None:
        self.discrete_solver = discrete_solver

    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:
        wasserstein_order = self.wasserstein_order

        if self.compute_moment_bound:
            moment_bound, _ = o_maximization(quantization.l2_radii.pow(wasserstein_order), quantization.lower_probs, quantization.upper_probs)
            moment_bound = moment_bound.pow(1 / wasserstein_order)
        else:
            moment_bound = torch.tensor(torch.nan)

        if self.compute_discrete_bound:
            cost_matrix = quantization.l2_distance_locs_to_locs.pow(wasserstein_order)

            discrete_bound = self.discrete_solver.solve(
                cost=cost_matrix.detach(),
                lower=quantization.lower_probs,
                upper=quantization.upper_probs,
                empirical_marginal=quantization.probs 
            ).bound.pow(1 / wasserstein_order)
        else:
            discrete_bound = torch.tensor(torch.nan)

        return Result(moment_bound=moment_bound, discrete_bound=discrete_bound)