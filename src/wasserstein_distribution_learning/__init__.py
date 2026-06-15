"""WassersteinDistributionLearning — data-driven Wasserstein ambiguity sets.

Two-step API
------------
1. **EmpiricalPartition** — build a data-driven partition from pretraining samples::

       ep = EmpiricalPartition(X_pre, num_clusters=50,
                               support=torch.stack([lo, hi]))
       ep.locs          # (50, d) centroids
       ep.partition     # raw Partition object

2. **AmbiguitySetLearner** — given any partition, learn the ambiguity set from
   evaluation samples::

       learner = AmbiguitySetLearner(ep, samples=X, beta=1e-6,
                                     learning_type='conditional')
       learner.ambiguity_set.radius
       learner.complement_interval
"""

from .api import (
    EmpiricalPartition,
    AmbiguitySetLearner,
    AmbiguitySet,
    ProbabilityInterval,
)
from .sets import HyperRectangle, BoundedVoronoiPartition, HyperRectanglePartition
from .quantization import FullLearningQuantization, ConditionalLearningQuantization
from .confidence import ClopperPearsonConfidence
from .bound import fournier_radius

__all__ = [
    # Step-by-step APIs
    "EmpiricalPartition",
    "AmbiguitySetLearner",
    # Return types
    "AmbiguitySet",
    "ProbabilityInterval",
    # Support / geometry
    "HyperRectangle",
    "BoundedVoronoiPartition",
    "HyperRectanglePartition",
    # Internals exposed for advanced users
    "FullLearningQuantization",
    "ConditionalLearningQuantization",
    "ClopperPearsonConfidence",
    "fournier_radius",
]
