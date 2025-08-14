from typing import Tuple
import torch
from torch_kmeans import KMeans

import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist


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
            cluster_radii: torch.Tensor,
        ):
        assert cluster_centers.size(0) == cluster_radii.size(0), "All tensors must have the same number of elements"

        self.support = support
        self.ndim = support.ndim
        
        self.cluster_centers = cluster_centers
        self.cluster_radii = cluster_radii
        self.outer_loc = support.center.unsqueeze(0)
        
        self.distance_locs = torch.cdist(self.locs, self.locs, p=2)

    def __len__(self):
        return self.locs.size(0)
    
    @property
    def locs(self):
        return torch.cat((self.cluster_centers, self.outer_loc), dim=0)

    @property
    def radii(self):
        return torch.cat((self.cluster_radii, torch.norm(self.support.width).unsqueeze(0) / 2. ))



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
            radii = torch.zeros(k)
            for i in range(k): # TODO do this using vmap
                if (labels == i).any():
                    radii[i] = distances[labels == i].max()
        else:
            locs = samples
            radii = torch.zeros(nsamples)

        super().__init__(support, locs, radii)


class BoundedVoronoiPartition(Partition):
    def __init__(self, support: HyperRectangle, samples: torch.Tensor, k: int, radius_scale_factor: float = 1.2):
        assert len(samples.shape) == 2, "Samples must be a 2D tensor (num_samples, num_features)"
        assert support.ndim == samples.shape[-1], "Support dimension must match sample features"

        nsamples = samples.size(0)

        if nsamples > k:
            kmeans_torch = KMeans(n_clusters=k)
            cluster_result = kmeans_torch(samples.unsqueeze(0)) # inputs should be at least of shape (BS, N, D)

            locs = cluster_result.centers.squeeze(0)
            labels = cluster_result.labels.squeeze(0)

            # Set the radii to half the diameter of each Voronoi cell in R^n with respect to the cluster centers.
            # For unbounded cells, the diameter will be infinite.
            diameters, bounded_mask = compute_voronoi_diameter(locs)
            radii = diameters / 2.

            # set the radii to radius_scale_factor * the max distance to any sample in the region
            sample_to_center_distance = radius_scale_factor * torch.norm(samples - locs[labels], dim=-1)
            for i in range(k):
                if (labels == i).any():
                    radii[i].clamp_(min=0, max=radius_scale_factor * sample_to_center_distance[labels == i].max().item())
        else:
            locs = samples
            radii = torch.zeros(nsamples)

        super().__init__(support, locs, radii)


@torch.no_grad
def compute_voronoi_diameter(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a per-site Voronoi diameter-like measure in R^d.

    Parameters
    ----------
    points : (N, d) array-like or torch.Tensor
        Input sites. Must be two-dimensional with N >= 2.

    Returns
    -------
    diameters : (N,) torch.Tensor of float64
        The maximum pairwise Euclidean distance between
        the cell's Voronoi vertices. 
        Unbounded cells return inf.
        Degenerate cases with fewer than two finite vertices return 0.0.
    bounded_mask : (N,) torch.Tensor of bool
        True if the Voronoi region is bounded, False otherwise.

    Notes
    -----
    - A region is unbounded if its vertex index list contains -1.
    - Finite vertices are taken directly from `scipy.spatial.Voronoi.vertices`.
    - Qhull options are controlled via `self.qhull_options` (default "QJ").
    """
    assert points.ndim == 2 and points.size(0) >= 2, "points must be (N, d) with N>=2"
    points_np = points.detach().cpu().numpy()
    
    vor = Voronoi(points_np, qhull_options="QJ")
    diameters = np.full(points_np.shape[0], np.inf)
    bounded_mask = np.zeros(points_np.shape[0], dtype=bool)

    for i, reg_idx in enumerate(vor.point_region):
        region = vor.regions[reg_idx]
        if not region:
            diameters[i] = 0.0
            continue

        finite_idx = [v for v in region if v != -1]
        if len(finite_idx) == 0:
            # Extremely degenerate; no finite vertices found.
            diameters[i] = 0.0
            continue

        verts = vor.vertices[np.array(finite_idx, dtype=int)]

        if -1 in region:
            ## Unbounded: 2x max finite distance from site to any finite vertex.
            # diffs = verts - points_np[i]                    # (m, d)
            # dists = np.sqrt(np.sum(diffs * diffs, axis=1))  # (m,)
            # diameters[i] = 2.0 * (dists.max() if dists.size else 0.0)
            bounded_mask[i] = False
        else:
            # Bounded: true diameter via vertex–vertex pairs.
            diameters[i] = pdist(verts).max() if len(verts) > 1 else 0.0
            bounded_mask[i] = True

    return torch.from_numpy(diameters).to(dtype=points.dtype), torch.from_numpy(bounded_mask)
