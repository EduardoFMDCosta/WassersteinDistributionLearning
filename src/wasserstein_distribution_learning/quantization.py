import torch

from .confidence import ClopperPearsonConfidence
from .sets import Partition


class Quantization:
    def __init__(self, partition: Partition, samples: torch.Tensor):
        self._partition = partition
        self.num_samples = samples.size(0)

        locs_to_samples_distance = torch.cdist(partition.region_locs, samples, p=2)
        mask = locs_to_samples_distance > partition.region_l2_radii.unsqueeze(1)
        in_outer = mask.all(dim=0)
        locs_to_samples_distance[mask] = torch.inf
        labels = torch.argmin(locs_to_samples_distance, dim=0)

        self.cluster_counts = torch.bincount(labels[~in_outer], minlength=len(partition) - 1)
        self.outer_counts = self.num_samples - self.cluster_counts.sum()
        assert self.outer_counts == sum(in_outer) >= 0, "Inconsistent outer counts"

    def __getattr__(self, name):
        if name == '_partition':
            raise AttributeError(name)
        return getattr(self._partition, name)

    def __len__(self) -> int:
        return len(self._partition)

    @property
    def counts(self) -> torch.Tensor:
        return torch.cat((self.cluster_counts, self.outer_counts.view(1)))

    @property
    def probs(self) -> torch.Tensor:
        return self.counts.float() / self.counts.sum()


class FullLearningQuantization(Quantization):
    """Unconditional: all N samples used, complement is the (M+1)-th atom.
    Clopper-Pearson applied jointly over M+1 regions with Bonferroni factor M+1.
    """

    def __init__(
        self,
        partition: Partition,
        samples: torch.Tensor,
        beta: float,
        ConfidenceClass: type = ClopperPearsonConfidence,
    ):
        super().__init__(partition=partition, samples=samples)
        self.confidence = ConfidenceClass(
            beta=beta / len(self._partition),
            n_set=self.counts,
            n=self.num_samples,
        )

    @property
    def lower_probs(self) -> torch.Tensor:
        return self.confidence.lower_proba

    @property
    def upper_probs(self) -> torch.Tensor:
        return self.confidence.upper_proba

    @property
    def lb_complement_prob(self) -> torch.Tensor:
        return self.confidence.lower_proba[-1]

    @property
    def ub_complement_prob(self) -> torch.Tensor:
        return self.confidence.upper_proba[-1]

    @property
    def outer_l2_radius(self) -> torch.Tensor:
        return self._partition.l2_radii[-1]


class ConditionalLearningQuantization(Quantization):

    def __init__(
        self,
        partition: Partition,
        samples: torch.Tensor,
        beta: float,
        ConfidenceClass: type = ClopperPearsonConfidence,
    ):
        super().__init__(partition=partition, samples=samples)
        M_plus_1 = len(self._partition)
        n_cond = int(self.cluster_counts.sum().item())
        if n_cond == 0:
            raise ValueError(
                "No samples fall within any bounded set; ConditionalLearningQuantization is "
                "undefined. Consider increasing M, providing a bounded support, or using "
                "FullLearningQuantization."
            )
        self.confidence = ConfidenceClass(
            beta=beta / M_plus_1,
            n_set=self.cluster_counts,
            n=n_cond,
        )
        self.confidence_complement = ConfidenceClass(
            beta=beta / M_plus_1,
            n_set=self.outer_counts.view(1),
            n=self.num_samples,
        )

    def __len__(self) -> int:
        return self._partition.region_locs.size(0)

    @property
    def counts(self) -> torch.Tensor:
        return self.cluster_counts

    @property
    def probs(self) -> torch.Tensor:
        return self.cluster_counts.float() / self.cluster_counts.sum()

    @property
    def locs(self) -> torch.Tensor:
        return self._partition.region_locs

    @property
    def l2_radii(self) -> torch.Tensor:
        return self._partition.region_l2_radii

    @property
    def l1_radii(self) -> torch.Tensor:
        return self._partition.l1_radii[:-1]

    @property
    def l2_distance_locs_to_locs(self) -> torch.Tensor:
        return self._partition.l2_distance_locs_to_locs[:-1, :-1]

    @property
    def l1_distance_locs_to_locs(self) -> torch.Tensor:
        return self._partition.l1_distance_locs_to_locs[:-1, :-1]

    @property
    def l2_distance_locs_to_region(self) -> torch.Tensor:
        d = self._partition.l2_distance_locs_to_locs[:-1, :-1]
        return d + self._partition.region_l2_radii.unsqueeze(-1)

    @property
    def l1_distance_locs_to_region(self) -> torch.Tensor:
        d = self._partition.l1_distance_locs_to_locs[:-1, :-1]
        return d + self._partition.l1_radii[:-1].unsqueeze(-1)


    @property
    def lower_probs(self) -> torch.Tensor:
        return self.confidence.lower_proba

    @property
    def upper_probs(self) -> torch.Tensor:
        return self.confidence.upper_proba

    @property
    def lb_complement_prob(self) -> torch.Tensor:
        return self.confidence_complement.lower_proba[0]

    @property
    def ub_complement_prob(self) -> torch.Tensor:
        return self.confidence_complement.upper_proba[0]

    @property
    def outer_l2_radius(self) -> torch.Tensor:
        return self._partition.l2_radii[-1]


UncertainQuantization = FullLearningQuantization

