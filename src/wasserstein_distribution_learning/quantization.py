import torch

from .confidence import ClopperPearsonConfidence
from .dataclasses import ProbabilityInterval
from .sets import Partition


class Quantization:
    def __init__(
        self,
        partition: Partition,
        samples: torch.Tensor,
        conditional: bool = False,
    ):
        self._partition = partition
        self.conditional = conditional
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
        if self.conditional:
            return self._partition.region_locs.size(0)
        return len(self._partition)

    @property
    def counts(self) -> torch.Tensor:
        if self.conditional:
            return self.cluster_counts
        return torch.cat((self.cluster_counts, self.outer_counts.view(1)))

    @property
    def probs(self) -> torch.Tensor:
        return self.counts.float() / self.counts.sum()

    @property
    def locs(self) -> torch.Tensor:
        if self.conditional:
            return self._partition.region_locs
        return self._partition.locs

    @property
    def l2_radii(self) -> torch.Tensor:
        if self.conditional:
            return self._partition.region_l2_radii
        return self._partition.l2_radii

    @property
    def l1_radii(self) -> torch.Tensor:
        if self.conditional:
            return self._partition.l1_radii[:-1]
        return self._partition.l1_radii

    @property
    def l2_distance_locs_to_locs(self) -> torch.Tensor:
        if self.conditional:
            return self._partition.l2_distance_locs_to_locs[:-1, :-1]
        return self._partition.l2_distance_locs_to_locs

    @property
    def l1_distance_locs_to_locs(self) -> torch.Tensor:
        if self.conditional:
            return self._partition.l1_distance_locs_to_locs[:-1, :-1]
        return self._partition.l1_distance_locs_to_locs

    @property
    def l2_distance_locs_to_region(self) -> torch.Tensor:
        if self.conditional:
            d = self._partition.l2_distance_locs_to_locs[:-1, :-1]
            return d + self._partition.region_l2_radii.unsqueeze(-1)
        return self._partition.l2_distance_locs_to_region

    @property
    def l1_distance_locs_to_region(self) -> torch.Tensor:
        if self.conditional:
            d = self._partition.l1_distance_locs_to_locs[:-1, :-1]
            return d + self._partition.l1_radii[:-1].unsqueeze(-1)
        return self._partition.l1_distance_locs_to_region

    @property
    def outer_l2_radius(self) -> torch.Tensor:
        return self._partition.l2_radii[-1]


class UncertainQuantization(Quantization):
    def __init__(
        self,
        partition: Partition,
        samples: torch.Tensor,
        beta: float,
        conditional: bool = False,
        ConfidenceClass: type = ClopperPearsonConfidence,
    ):
        super().__init__(partition=partition, samples=samples, conditional=conditional)

        M_plus_1 = len(self._partition)
        if self.conditional:
            n_cond = int(self.cluster_counts.sum().item())
            if n_cond == 0:
                raise ValueError(
                    "No samples fall within any bounded set; conditional quantization is "
                    "undefined. Consider increasing M, providing a bounded support, or using "
                    "full quantization."
                )
            confidence = ConfidenceClass(
                beta=beta / M_plus_1,
                n_set=self.cluster_counts,
                n=n_cond,
            )
            self._interval = ProbabilityInterval(
                lower=confidence.lower_proba,
                upper=confidence.upper_proba,
            )

            confidence_complement = ConfidenceClass(
                beta=beta / M_plus_1,
                n_set=self.outer_counts.view(1),
                n=self.num_samples,
            )
            self._complement_interval = ProbabilityInterval(
                lower=confidence_complement.lower_proba[0],
                upper=confidence_complement.upper_proba[0],
            )
        else:
            confidence = ConfidenceClass(
                beta=beta / M_plus_1,
                n_set=self.counts,
                n=self.num_samples,
            )
            self._interval = ProbabilityInterval(
                lower=confidence.lower_proba,
                upper=confidence.upper_proba,
            )
            self._complement_interval = ProbabilityInterval(
                lower=confidence.lower_proba[-1],
                upper=confidence.upper_proba[-1],
            )

    @property
    def interval(self) -> ProbabilityInterval:
        return self._interval

    @property
    def complement_interval(self) -> ProbabilityInterval:
        return self._complement_interval

