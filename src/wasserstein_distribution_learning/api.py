from dataclasses import dataclass
from typing import Optional, Union

import torch

from .sets import HyperRectangle, BoundedVoronoiPartition, HyperRectanglePartition, Partition
from .quantization import Quantization, FullLearningQuantization, ConditionalLearningQuantization
from .confidence import ClopperPearsonConfidence
from .bound import DataDrivenRadius, fournier_radius as _fournier_radius
from .solvers import get_solver


_LEARNING_TYPE_MAP = {
    'full':               FullLearningQuantization,
    'full_learning':      FullLearningQuantization,
    'conditional':        ConditionalLearningQuantization,
    'conditional_learning': ConditionalLearningQuantization,
}

_PARTITION_TYPE_MAP = {
    'voronoi':        BoundedVoronoiPartition,
    'hyperrectangle': HyperRectanglePartition,
}


@dataclass
class ProbabilityInterval:
    """Clopper-Pearson probability interval [lower, upper] for the complement set."""
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass
class AmbiguitySet:
    center: Quantization
    radius: torch.Tensor


class EmpiricalPartition:

    def __init__(
        self,
        pretraining_samples: torch.Tensor,
        num_clusters: int = 100,
        support: Optional[torch.Tensor] = None,
        partition_type: str = 'voronoi',
    ):
        if partition_type not in _PARTITION_TYPE_MAP:
            raise ValueError(f"partition_type must be one of {list(_PARTITION_TYPE_MAP)}, got {partition_type!r}")
        _support = HyperRectangle(support[0], support[1]) if support is not None else None
        self.partition: Partition = _PARTITION_TYPE_MAP[partition_type].from_samples(
            support=_support, samples=pretraining_samples, M=num_clusters,
        )

    @property
    def locs(self) -> torch.Tensor:
        return self.partition.region_locs

    @property
    def num_regions(self) -> int:
        return self.partition.region_locs.shape[0]


class AmbiguitySetLearner:

    def __init__(
        self,
        partition: Union['EmpiricalPartition', Partition],
        samples: torch.Tensor,
        beta: float,
        learning_type: str = 'full',
        method: str = 'triangle_inequality_vertex',
        wasserstein_order: int = 2,
        ConfidenceClass: type = ClopperPearsonConfidence,
        time_limit: Optional[float] = None,
    ):
        _partition = partition.partition if isinstance(partition, EmpiricalPartition) else partition
        if learning_type not in _LEARNING_TYPE_MAP:
            raise ValueError(f"learning_type must be one of {list(_LEARNING_TYPE_MAP)}, got {learning_type!r}")
        LearningType = _LEARNING_TYPE_MAP[learning_type]

        quantization = LearningType(
            partition=_partition,
            samples=samples,
            beta=beta,
            ConfidenceClass=ConfidenceClass,
        )

        solver = get_solver(method=method)
        result = DataDrivenRadius(
            quantization=quantization,
            solver=solver,
            wasserstein_order=wasserstein_order,
            time_limit=time_limit,
        )

        self.ambiguity_set = AmbiguitySet(
            center=quantization,
            radius=result.radius,
        )
        self.complement_interval = (
            ProbabilityInterval(
                lower=result.lb_complement_prob,
                upper=result.ub_complement_prob,
            )
            if isinstance(quantization, ConditionalLearningQuantization)
            else None
        )
        self.fournier_radius: float = _fournier_radius(
            support=_partition.support,
            nsamples=samples.shape[0],
            wasserstein_order=wasserstein_order,
            beta=beta,
        )
        self._result = result



