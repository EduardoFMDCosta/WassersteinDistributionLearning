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
            cluster_centers: torch.Tensor,
            cluster_diameters: torch.Tensor,
        ):
        assert cluster_centers.size(0) == cluster_diameters.size(0), "All tensors must have the same number of elements"

        self.support = support
        self.ndim = support.ndim
        
        self.cluster_centers = cluster_centers
        self.cluster_diameters = cluster_diameters
        self.outer_loc = support.center.unsqueeze(0)
        
        self.distance_locs = torch.cdist(self.locs, self.locs, p=2)

    def __len__(self):
        return self.locs.size(0)
    
    @property
    def locs(self):
        return torch.cat((self.cluster_centers, self.outer_loc), dim=0)

    @property
    def diameters(self):
        return torch.cat((self.cluster_diameters, torch.norm(self.support.width).unsqueeze(0) / 2. ))



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

        super().__init__(support, locs, diameters)
