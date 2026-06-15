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


# ---------------------------------------------------------------------------
# API 1 — partition step
# ---------------------------------------------------------------------------

class EmpiricalPartition:
    """Build a data-driven partition of the sample space from pretraining data.

    This is the *geometry* step: given a cloud of pretraining samples it
    produces M bounded regions that tile the relevant part of the space.
    The result can be inspected on its own, or passed directly to
    :class:`AmbiguitySetLearner` to learn a Wasserstein ambiguity set from a
    separate set of evaluation samples.

    Parameters
    ----------
    pretraining_samples : torch.Tensor, shape (N_pre, d)
        Samples used to determine the partition geometry (cluster centres and
        region extents).  These are *not* used for probability estimation.
    num_clusters : int
        Number of bounded regions M.
    support : torch.Tensor of shape (2, d), optional
        Known bounded support given as
        ``torch.stack([lower_bounds, upper_bounds])``.
        Pass ``None`` for an unbounded (or unknown) support.
    partition_type : {'voronoi', 'hyperrectangle'}
        Geometry of the regions.
        ``'voronoi'`` uses K-means centres with bounded Voronoi cells.
        ``'hyperrectangle'`` uses a GMM-seeded binary space partition into
        axis-aligned boxes.

    Attributes
    ----------
    partition : Partition
        The underlying partition object (``BoundedVoronoiPartition`` or
        ``HyperRectanglePartition``).  Pass this to
        :class:`AmbiguitySetLearner` when you want to handle the two steps
        separately.
    locs : torch.Tensor, shape (M, d)
        Centroid of each bounded region.
    num_regions : int
        Number of bounded regions M.

    Examples
    --------
    >>> ep = EmpiricalPartition(X_pre, num_clusters=50,
    ...                         support=torch.stack([lo, hi]))
    >>> ep.locs          # (50, d) centroid tensor
    >>> ep.num_regions   # 50
    >>> # pass to the learner:
    >>> learner = AmbiguitySetLearner(ep, samples=X, beta=1e-6)
    """

    def __init__(
        self,
        pretraining_samples: torch.Tensor,
        num_clusters: int = 100,
        support: Optional[torch.Tensor] = None,
        partition_type: str = 'voronoi',
    ):
        if support is not None:
            _support = HyperRectangle(lower=support[0], upper=support[1])
        else:
            _support = None

        if partition_type not in _PARTITION_TYPE_MAP:
            raise ValueError(
                f"partition_type must be one of {list(_PARTITION_TYPE_MAP)}, "
                f"got {partition_type!r}"
            )
        PartitionType = _PARTITION_TYPE_MAP[partition_type]

        self.partition: Partition = PartitionType.from_samples(
            support=_support,
            samples=pretraining_samples,
            M=num_clusters,
        )

    @property
    def locs(self) -> torch.Tensor:
        """Centroid of each bounded region, shape (M, d)."""
        return self.partition.region_locs

    @property
    def num_regions(self) -> int:
        """Number of bounded regions M."""
        return self.partition.region_locs.shape[0]


# ---------------------------------------------------------------------------
# API 2 — learning step
# ---------------------------------------------------------------------------

class AmbiguitySetLearner:
    """Learn a data-driven Wasserstein ambiguity set from samples given a partition.

    This is the *statistical* step: given a pre-built partition and a set of
    evaluation samples it estimates empirical probabilities with Clopper-Pearson
    confidence intervals and then solves the optimisation problem that yields the
    data-driven Wasserstein radius.

    The partition can come from :class:`EmpiricalPartition` (pass the instance
    directly, not ``instance.partition``) or be any :class:`Partition` object
    supplied by the user.

    Parameters
    ----------
    partition : EmpiricalPartition or Partition
        The partition that defines the geometry.  If an
        :class:`EmpiricalPartition` is passed its ``partition`` attribute is
        used automatically.
    samples : torch.Tensor, shape (N, d)
        Evaluation samples used to estimate probability weights and compute
        the data-driven radius.
    beta : float
        Overall confidence level.  The returned ambiguity set contains the
        true distribution with probability at least 1 − beta.
    learning_type : {'full', 'conditional'}
        ``'full'``: all N samples are treated as unconditional observations;
        the complement mass becomes the (M+1)-th atom of the centre
        distribution (appropriate for bounded support).
        ``'conditional'``: samples are treated as conditional on lying in the
        bounded regions; complement probability is reported separately via
        ``complement_interval`` (appropriate for unbounded support).
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
        The learned ambiguity set (``center`` discrete distribution + ``radius``).
    complement_interval : ProbabilityInterval or None
        Clopper-Pearson bounds on P(X outside bounded region).
        ``None`` for ``learning_type='full'``; set for ``'conditional'``.
    fournier_radius : float
        Minimax-optimal Fournier–Guillin radius computed from ``samples``.
        ``inf`` when ``support`` is unbounded.

    Examples
    --------
    >>> ep = EmpiricalPartition(X_pre, num_clusters=50)
    >>> learner = AmbiguitySetLearner(ep, samples=X, beta=1e-6,
    ...                               learning_type='conditional')
    >>> learner.ambiguity_set.radius
    >>> learner.complement_interval
    """

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
        # Accept either an EmpiricalPartition wrapper or a raw Partition
        _partition = partition.partition if isinstance(partition, EmpiricalPartition) else partition

        if learning_type not in _LEARNING_TYPE_MAP:
            raise ValueError(
                f"learning_type must be one of {list(_LEARNING_TYPE_MAP)}, "
                f"got {learning_type!r}"
            )
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



