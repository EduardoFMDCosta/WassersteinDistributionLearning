from typing import Callable, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class MaxMinLPResult:
    objective_opt: float
    w_opt: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None

class MaxMinLP(ABC):
    @abstractmethod
    def solve(
        self,
        cost: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        empirical_marginal: torch.Tensor,
    ) -> MaxMinLPResult:
        pass