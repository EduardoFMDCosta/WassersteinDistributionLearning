from dataclasses import dataclass
import torch


@dataclass
class ProbabilityInterval:
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass
class AmbiguitySet:
    center: object
    radius: torch.Tensor
