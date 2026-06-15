from dataclasses import dataclass
from typing import Optional

import torch

from .sets import HyperRectangle, BoundedVoronoiPartition, HyperRectanglePartition, Partition
from .quantization import Quantization, FullLearningQuantization, ConditionalLearningQuantization
from .confidence import ClopperPearsonConfidence
from .bound import DataDrivenRadius, fournier_radius as _fournier_radius
from .solvers import get_solver


_LEARNING_TYPE_MAP = {
    'full':               FullLearningQuantization,
    'full_learning':      FullLearningQuantization,       # handlers.py alias
    'conditional':        ConditionalLearningQuantization,
    'conditional_learning': ConditionalLearningQuantization,  # handlers.py alias
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
    """Wasserstein ambiguity ball of given radius around a discrete centre distribution.

    Attributes
    ----------
    center : Quantization
        The discrete distribution at the centre of the ball.  Access
        ``center.locs`` (support points) and ``center.probs`` (weights).
    radius : torch.Tensor
        Data-driven Wasserstein radius of the ambiguity ball.
    """
    center: Quantization
    radius: torch.Tensor


class WassersteinDistributionLearning:
    """Data-driven Wasserstein ambiguity set learning.

    Parameters
    ----------
    pretraining_samples : torch.Tensor, shape (N_pre, d)
        Samples used to build the partition (cluster centres).
    samples : torch.Tensor, shape (N, d)
        Samples used to estimate the probability weights and compute the
        data-driven radius.
    beta : float
        Overall confidence level.  The returned ambiguity set contains the
        true distribution with probability at least 1 − beta.
    support : torch.Tensor of shape (2, d), optional
        Known (or assumed) bounded support, given as
        ``torch.stack([lower_bounds, upper_bounds])``.  Pass ``None`` to
        indicate an unbounded support; the radius will naturally be infinite.
    learning_type : {'full', 'conditional'}
        ``'full'``: treat all N samples as unconditional observations
        (complement mass becomes the (M+1)-th atom of the centre distribution).
        ``'conditional'``: N_pre samples define the partition, N samples are
        drawn conditional on lying in the bounded region; complement probability
        is reported separately via ``complement_interval``.
    partition_type : {'voronoi', 'hyperrectangle'}
        Geometry used to build the partition of the support.
    num_clusters : int
        Number of bounded partition regions M.
    method : str
        Solver method name (e.g. ``'triangle_inequality_vertex'``).
    wasserstein_order : int
        Wasserstein order p (1 or 2).
    ConfidenceClass : type
        Confidence interval class (default: ``ClopperPearsonConfidence``).
    time_limit : float, optional
        Solver time limit in seconds.

    Attributes
    ----------
    ambiguity_set : AmbiguitySet
        The learned ambiguity set (``center`` + ``radius``).
    complement_interval : ProbabilityInterval or None
        Clopper-Pearson bounds for P(X outside bounded region).
        ``None`` for ``learning_type='full'``; set for ``'conditional'``.
    fournier_radius : float
        Minimax-optimal Fournier–Guillin radius for reference.
    """

    def __init__(
        self,
        pretraining_samples: torch.Tensor,
        samples: torch.Tensor,
        beta: float,
        support: Optional[torch.Tensor] = None,
        learning_type: str = 'full',
        partition_type: str = 'voronoi',
        num_clusters: int = 100,
        method: str = 'triangle_inequality_vertex',
        wasserstein_order: int = 2,
        ConfidenceClass: type = ClopperPearsonConfidence,
        time_limit: Optional[float] = None,
    ):
        # Convert support tensor (2, d) → HyperRectangle
        if support is not None:
            _support = HyperRectangle(lower=support[0], upper=support[1])
        else:
            _support = None

        # Resolve type strings
        if learning_type not in _LEARNING_TYPE_MAP:
            raise ValueError(
                f"learning_type must be one of {list(_LEARNING_TYPE_MAP)}, got {learning_type!r}"
            )
        LearningType = _LEARNING_TYPE_MAP[learning_type]

        if partition_type not in _PARTITION_TYPE_MAP:
            raise ValueError(
                f"partition_type must be one of {list(_PARTITION_TYPE_MAP)}, got {partition_type!r}"
            )
        PartitionType = _PARTITION_TYPE_MAP[partition_type]

        partition: Partition = PartitionType.from_samples(
            support=_support,
            samples=pretraining_samples,
            M=num_clusters,
        )

        quantization = LearningType(
            partition=partition,
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
            support=_support,
            nsamples=pretraining_samples.shape[0] + samples.shape[0],
            wasserstein_order=wasserstein_order,
            beta=beta,
        )

        self._result = result

