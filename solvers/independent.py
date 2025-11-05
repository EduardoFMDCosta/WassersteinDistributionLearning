import torch

from optimization_utils import o_maximization
from quantization import UncertainQuantization

from solvers.templates import Solver, Result, DiscreteSolver


class IndependentSolver(Solver):
    def __init__(self, discrete_solver: DiscreteSolver) -> None:
        super().__init__()
        self.discrete_solver = discrete_solver

    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:

        # Compute moment bound
        moment_bound, _ = o_maximization(quantization.partition.radii.pow(2), quantization.lower_probs, quantization.upper_probs)
        moment_bound = moment_bound.pow(0.5)

        # Compute discrete bound
        cost_matrix = quantization.partition.distance_locs.pow(2)

        discrete_result = self.discrete_solver.solve(
            cost=cost_matrix.detach(),
            lower=quantization.lower_probs,
            upper=quantization.upper_probs,
            empirical_marginal=quantization.probs 
        )

        return Result(moment_bound=moment_bound, discrete_bound=discrete_result.bound)