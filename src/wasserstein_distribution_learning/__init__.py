"""WassersteinDistributionLearning — data-driven Wasserstein ambiguity sets.

Typical usage
-------------
>>> from wasserstein_distribution_learning import WassersteinDistributionLearning
>>> wdl = WassersteinDistributionLearning(
...     pretraining_samples=X_pre,
...     samples=X,
...     beta=1e-6,
...     support=support,          # HyperRectangle or None
... )
>>> wdl.ambiguity_set.radius      # Wasserstein radius (tensor)
>>> wdl.ambiguity_set.center.locs # discrete support points
>>> wdl.ambiguity_set.center.probs
>>> wdl.complement_interval       # ProbabilityInterval or None
"""

from .api import WassersteinDistributionLearning, AmbiguitySet, ProbabilityInterval
from .sets import HyperRectangle, BoundedVoronoiPartition, HyperRectanglePartition
from .quantization import FullLearningQuantization, ConditionalLearningQuantization
from .confidence import ClopperPearsonConfidence, HoeffdingConfidence, DuchiConfidence
from .bound import fournier_radius

__all__ = [
    "WassersteinDistributionLearning",
    "AmbiguitySet",
    "ProbabilityInterval",
    "HyperRectangle",
    "BoundedVoronoiPartition",
    "HyperRectanglePartition",
    "FullLearningQuantization",
    "ConditionalLearningQuantization",
    "ClopperPearsonConfidence",
    "HoeffdingConfidence",
    "DuchiConfidence",
    "fournier_radius",
]
