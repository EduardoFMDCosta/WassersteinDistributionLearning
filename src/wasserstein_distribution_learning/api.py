from dataclasses import dataclass
from typing import Optional

import torch

from .sets import HyperRectangle, BoundedVoronoiPartition, HyperRectanglePartition, Partition
from .quantization import Quantization, FullLearningQuantization, ConditionalLearningQuantization
from .confidence import ClopperPearsonConfidence
from .bound import DataDrivenRadius
from .solvers import get_solver


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
    support : HyperRectangle, optional
        Known (or assumed) bounded support.  Pass ``None`` to indicate an
        unbounded support; the radius will naturally be infinite.
    LearningClass : type
        Either ``FullLearningQuantization`` (default) or
        ``ConditionalLearningQuantization``.
    PartitionClass : type
        Either ``BoundedVoronoiPartition`` (default) or
        ``HyperRectanglePartition``.
    num_clusters : int
        Number of bounded partition regions M.
    method : str
        Solver method name (e.g. ``'triangle_inequality_vertex'``).
    wasserstein_order : int
        Wasserstein order p (1 or 2).
    ConfidenceClass : type
        Confidence interval class (default: ``ClopperPearsonConfidence``).
    compute_moment_bound : bool
        Whether to compute the moment term of the radius.
    compute_discrete_bound : bool
        Whether to compute the discrete term of the radius.
    time_limit : float, optional
        Solver time limit in seconds.

    Attributes
    ----------
    ambiguity_set : AmbiguitySet
        The learned ambiguity set.
    complement_interval : ProbabilityInterval or None
        Clopper-Pearson bounds for P(X in complement of bounded sets).
        Always ``None`` for ``FullLearningQuantization`` (the complement is
        the last element of the ambiguity set's centre distribution).
        Set to a ``ProbabilityInterval`` for ``ConditionalLearningQuantization``.
    """

    def __init__(
        self,
        pretraining_samples: torch.Tensor,
        samples: torch.Tensor,
        beta: float,
        support: Optional[HyperRectangle] = None,
        LearningClass: type = FullLearningQuantization,
        PartitionClass: type = BoundedVoronoiPartition,
        num_clusters: int = 100,
        method: str = 'triangle_inequality_vertex',
        wasserstein_order: int = 2,
        ConfidenceClass: type = ClopperPearsonConfidence,
        compute_moment_bound: bool = True,
        compute_discrete_bound: bool = True,
        time_limit: Optional[float] = None,
    ):
        partition: Partition = PartitionClass.from_samples(
            support=support,
            samples=pretraining_samples,
            M=num_clusters,
        )

        quantization = LearningClass(
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
            compute_moment_bound=compute_moment_bound,
            compute_discrete_bound=compute_discrete_bound,
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

        self._result = result
