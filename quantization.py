import torch

from sets import Partition, VoronoiPartition


class Quantization:
    def __init__(self, partition: Partition, samples: torch.Tensor):
        self.partition = partition
        self.samples = samples
        self.nsamples = samples.size(0)

        if isinstance(partition, VoronoiPartition):
            distances = torch.cdist(partition.cluster_centers, samples, p=2)
            assignment = torch.argmin(distances, dim=0)
            self.cluster_counts = torch.bincount(assignment, minlength=partition.cluster_centers.size(0))
        else:
            raise NotImplementedError
    
        self.outer_counts = self.nsamples - self.cluster_counts.sum()
        assert self.outer_counts >= 0

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

    def __len__(self):
        return len(self.partition)
