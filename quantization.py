import torch

from sets import Partition, BoundedVoronoiPartition


class Quantization:
    def __init__(self, partition: Partition, samples: torch.Tensor):
        self.partition = partition
        self.samples = samples
        self.nsamples = samples.size(0)

        if isinstance(partition, BoundedVoronoiPartition):
            centers_to_samples_distance = torch.cdist(partition.cluster_centers, samples, p=2)
            labels = torch.argmin(centers_to_samples_distance, dim=0)

            sample_to_center_distance = torch.norm(samples - partition.cluster_centers[labels], dim=-1)
            in_outer = sample_to_center_distance > partition.cluster_radii[labels]

            self.cluster_counts = torch.bincount(labels[~in_outer], minlength=len(partition) - 1)

            self.outer_counts = self.nsamples - self.cluster_counts.sum()
            assert self.outer_counts == sum(in_outer) >= 0, "Inconsistent outer counts"
        else:
            raise NotImplementedError
    
        

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
