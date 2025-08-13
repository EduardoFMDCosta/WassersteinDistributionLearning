import torch
from torch_kmeans import KMeans


class HyperRectangle:
    def __init__(self, lower, upper):
        assert lower.size() == upper.size(), "Lower and upper bounds must have the same shape"
        self.lower, self.upper = lower, upper
        self.ndim = lower.size(-1)

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    def __len__(self):
        return self.lower.size(0)

    def size(self, dim=None):
        if dim is None:
            return self.lower.size()

        return self.lower.size(dim)
    
    def included(self, point: torch.Tensor):
        """Check if a point is included in the hyperrectangle."""
        return torch.all((point >= self.lower) & (point <= self.upper), dim=-1)

    @staticmethod
    def from_eps(x, eps):
        lower, upper = x - eps, x + eps
        return HyperRectangle(lower, upper)

class Partition:
    def __init__(
            self, 
            support: HyperRectangle,
            locs: torch.Tensor, 
            distance_locs: torch.Tensor, 
            diameters: torch.Tensor
        ):
        assert locs.size(0) == distance_locs.size(0) == distance_locs.size(1) == diameters.size(0), "All tensors must have the same number of elements"

        self.support = support
        self.ndim = support.ndim
        self.locs = locs
        self.distance_locs = distance_locs
        self.diameters = diameters
        self.npartitions = locs.size(0) # TODO remove

    def __len__(self):
        return self.locs.size(0)


class ConvexHullPartition(Partition):
    def __init__(self, support: HyperRectangle, samples: torch.Tensor, k: int):
        assert len(samples.shape) == 2, "Samples must be a 2D tensor (num_samples, num_features)"
        assert support.ndim == samples.shape[-1], "Support dimension must match sample features"

        nsamples = samples.size(0)

        if nsamples > k:
            kmeans_torch = KMeans(n_clusters=k)
            cluster_result = kmeans_torch(samples.unsqueeze(0)) # inputs should be at least of shape (BS, N, D)

            locs = cluster_result.centers.squeeze(0)
            labels = cluster_result.labels.squeeze(0)

            distances = torch.norm(samples - locs[labels], dim=-1)
            diameters = torch.zeros(k)
            for i in range(k): # TODO do this using vmap
                if (labels == i).any():
                    diameters[i] = distances[labels == i].max()
        else:
            locs = samples
            diameters = torch.zeros(nsamples)

        # Append outer region
        locs = torch.cat((locs, support.center.unsqueeze(0)))
        distance_locs =torch.cdist(locs, locs, p=2)
        diameters = torch.cat((diameters, torch.norm(support.width).unsqueeze(0) / 2. ))

        super().__init__(support, locs, distance_locs, diameters)
