from typing import Callable, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from quantization import UncertainQuantization


@dataclass
class Result:
    bound: torch.Tensor
    moment_bound: Optional[torch.Tensor] = None
    discrete_bound: Optional[torch.Tensor] = None
    w_opt: Optional[torch.Tensor] = None

class Solver(ABC):
    @abstractmethod
    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:
        pass

@dataclass
class DiscreteResult:
    objective_opt: float
    w_opt: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None

class DiscreteSolver(ABC):
    @abstractmethod
    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor
    ) -> DiscreteResult:
        pass
