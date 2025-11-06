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
            bound = moment_bound + discrete_bound
    
        self.bound = torch.tensor(torch.nan) if bound is None else bound
        self.moment_bound = torch.tensor(torch.nan) if moment_bound is None else moment_bound
        self.discrete_bound = torch.tensor(torch.nan) if discrete_bound is None else discrete_bound
        self.w_opt = w_opt
    

class Solver(ABC):
    _compute_moment_bound = True
    _compute_discrete_bound = True

    @property
    def compute_moment_bound(self) -> bool:
        return self._compute_moment_bound
    
    @compute_moment_bound.setter
    def compute_moment_bound(self, value: bool) -> None:
        self._compute_moment_bound = value

    @property
    def compute_discrete_bound(self) -> bool:
        return self._compute_discrete_bound

    @compute_discrete_bound.setter
    def compute_discrete_bound(self, value: bool) -> None:
        self._compute_discrete_bound = value

    def enable_moment_bound_computation(self) -> None:
        self.compute_moment_bound = True

    def disable_moment_bound_computation(self) -> None:
        self.compute_moment_bound = False

    def enable_discrete_bound_computation(self) -> None:
        self.compute_discrete_bound = True
    
    def disable_discrete_bound_computation(self) -> None:
        self.compute_discrete_bound = False

    @abstractmethod
    def solve(
        self,
        quantization: UncertainQuantization,
        wasserstein_order: int,
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
