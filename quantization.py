import torch

from confidence import ClopperPearsonConfidence
from sets import BoundedVoronoiPartition


class Quantization:
    def __init__(
            self, 
            partition: BoundedVoronoiPartition, 
            samples: torch.Tensor
    ):
        self.partition = partition
        self.samples = samples
        self.nsamples = samples.size(0)

        centers_to_samples_distance = torch.cdist(partition.cluster_centers, samples, p=2)
        mask = centers_to_samples_distance > partition.cluster_radii.unsqueeze(1)
        in_outer = mask.all(dim=0)
        centers_to_samples_distance[mask] = torch.inf
        labels = torch.argmin(centers_to_samples_distance, dim=0)

        self.cluster_counts = torch.bincount(labels[~in_outer], minlength=len(partition) - 1)

        self.outer_counts = self.nsamples - self.cluster_counts.sum()
        assert self.outer_counts == sum(in_outer) >= 0, "Inconsistent outer counts"

    @property
    def counts(self):
        return torch.cat((self.cluster_counts, self.outer_counts.view(1)))

    @property
    def probs(self):
        return self.counts.float() / self.counts.sum()
    
    @property
    def locs(self):
        return self.partition.locs

    @property
    def ndim(self):
        return self.partition.ndim
    
    @property
    def cluster_radii(self):
        return self.partition.cluster_radii
    
    @property
    def distance_locs(self):
        return self.partition.distance_locs

    def __len__(self):
        return len(self.partition)


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
            n=self.nsamples
        )

    @property
    def lower_probs(self):
        return self.confidence.lower_proba
    
    @property
    def upper_probs(self):
        return self.confidence.upper_proba
