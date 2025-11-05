from typing import Any, Callable, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from quantization import UncertainQuantization


class Result:
    def __init__(
        self,
        bound: Optional[torch.Tensor] = None,
        moment_bound: Optional[torch.Tensor] = None,
        discrete_bound: Optional[torch.Tensor] = None,
        w_opt: Optional[torch.Tensor] = None        
    ):
        if bound is None and moment_bound is not None and discrete_bound is not None:
            self.bound = moment_bound + discrete_bound
        else:
            self.bound = bound

        self.moment_bound = moment_bound
        self.discrete_bound = discrete_bound
        self.w_opt = w_opt
    

class Solver(ABC):
    @abstractmethod
    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> Result:
        pass

@dataclass
class DiscreteResult:
    bound: torch.Tensor
    w_opt: Optional[torch.Tensor] = None

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
