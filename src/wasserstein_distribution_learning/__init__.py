"""WassersteinDistributionLearning — data-driven Wasserstein ambiguity sets.

Three levels of API
-------------------
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

3. **WassersteinDistributionLearning** — full pipeline in one call::

       wdl = WassersteinDistributionLearning(
           pretraining_samples=X_pre, samples=X, beta=1e-6)
       wdl.ambiguity_set.radius
       wdl.fournier_radius
"""

from .api import (
    WassersteinDistributionLearning,
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
    # Full pipeline
    "WassersteinDistributionLearning",
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
