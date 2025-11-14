import torch

from confidence import ClopperPearsonConfidence
from sets import BoundedVoronoiPartition


class Quantization(BoundedVoronoiPartition):
    def __init__(
            self, 
            partition: BoundedVoronoiPartition, 
            samples: torch.Tensor
    ):
        super().__init__(
            support=partition.support,
            region_locs=partition.region_locs,
            region_l2_radii=partition.region_l2_radii
        )
        
        self.num_samples = samples.size(0)

        locs_to_samples_distance = torch.cdist(partition.region_locs, samples, p=2)
        mask = locs_to_samples_distance > partition.region_l2_radii.unsqueeze(1)
        in_outer = mask.all(dim=0)
        locs_to_samples_distance[mask] = torch.inf
        labels = torch.argmin(locs_to_samples_distance, dim=0)

        self.cluster_counts = torch.bincount(labels[~in_outer], minlength=len(partition) - 1)
        self.outer_counts = self.num_samples - self.cluster_counts.sum()
        assert self.outer_counts == sum(in_outer) >= 0, "Inconsistent outer counts"

    @property
    def counts(self):
        return torch.cat((self.cluster_counts, self.outer_counts.view(1)))

    @property
    def probs(self):
        return self.counts.float() / self.counts.sum()


class UncertainQuantization(Quantization):
    def __init__(
            self, 
            partition: BoundedVoronoiPartition, 
            samples: torch.Tensor,
            beta: float, 
            ConfidenceClass: type = ClopperPearsonConfidence
    ):

        super().__init__(partition=partition, samples=samples)

        self.confidence = ConfidenceClass(
            beta=beta / self.__len__(), 
            n_set=self.counts, 
            n=self.num_samples
        )

    @property
    def lower_probs(self):
        return self.confidence.lower_proba
    
    @property
    def upper_probs(self):
        return self.confidence.upper_proba
