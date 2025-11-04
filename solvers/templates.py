from typing import Callable, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from quantization import UncertainQuantization


@dataclass
class MaxMinLPResult:
    bound: torch.Tensor
    moment_bound: Optional[torch.Tensor] = None
    discrete_bound: Optional[torch.Tensor] = None
    w_opt: Optional[torch.Tensor] = None

class MaxMinLP(ABC):
    @abstractmethod
    def solve(
        self,
        quantization: UncertainQuantization,
    ) -> MaxMinLPResult:
        pass