import torch

from sets import Partition


class Quantization:
    def __init__(self, partition: Partition, samples: torch.Tensor):

        self.partition = partition
        self.samples = samples
        self.nsamples = samples.size(0)

        distances_locs_samples = torch.cdist(partition.locs, samples, p=2)
        assignment = torch.argmin(distances_locs_samples, dim=0)
        self.counts = torch.bincount(assignment, minlength=partition.npartitions)

    @property
    def probs(self):
        return self.counts.float() / self.counts.sum()
    
    @property
    def locs(self):
        self.partition.locs

    @property
    def ndim(self):
        return self.partition.ndim


    def __len__(self):
        return len(self.partition)
