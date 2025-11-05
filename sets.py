from typing import Tuple, Optional
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


class BoundedVoronoiPartition:
    def __init__(
        self, 
        support: HyperRectangle, 
        region_locs: torch.Tensor,
        region_l2_radii: torch.Tensor  # TODO provide either l2 or l1 (based on what is used in .from_samples)
    ):
        self.support = support
        self.region_locs = region_locs
        self.region_l2_radii = region_l2_radii
        self.l2_distance_locs_to_locs = torch.cdist(self.locs, self.locs, p=2)
    
    def __len__(self):
        return self.locs.size(0)

    @property
    def ndim(self):
        return self.support.ndim
    
    @property
    def locs(self):
        return torch.cat((self.region_locs, self.support.center.unsqueeze(0)), dim=0)

    @property
    def l2_radii(self):
        return torch.cat((self.region_l2_radii, torch.norm(self.support.width).unsqueeze(0) / 2. ))
    
    @property
    def l2_distance_locs_to_region(self):
        return self.l2_distance_locs_to_locs + self.region_l2_radii.unsqueeze(-1)
    
    @property
    def l1_radii(self):
        return (2**0.5) * self.l2_radii
    
    @property
    def l1_distance_locs_to_region(self):
        return self.l1_distance_locs_to_locs + self.l1_radii.unsqueeze(-1)

    @classmethod
    def from_samples(
            cls,
            support: HyperRectangle, 
            samples: torch.Tensor, 
            M: int, 
            radius_scale_factor: float = 1.5, 
            use_voronoi_radii: bool = False
        ):
        assert len(samples.shape) == 2, "Samples must be a 2D tensor (num_samples, num_features)"
        assert support.ndim == samples.shape[-1], "Support dimension must match sample features"

        nsamples = samples.size(0)

        if nsamples > M:
            kmeans_torch = KMeans(n_clusters=M)
            cluster_result = kmeans_torch(samples.unsqueeze(0)) # inputs should be at least of shape (BS, N, D)

            cluster_locs = cluster_result.centers.squeeze(0)
            labels = cluster_result.labels.squeeze(0)

            max_sample_distances = compute_inner_cluster_max_l2_radii(samples, cluster_locs, labels)    
        else:
            cluster_locs = samples
            max_sample_distances = torch.zeros(M)

        distance_locs = torch.cdist(cluster_locs, cluster_locs, p=2)

        # Set the radii to half the diameter of each Voronoi cell in R^n with respect to the cluster locs.
        # For unbounded cells, the diameter will be infinite.
        if use_voronoi_radii:
            l2_radii = compute_voronoi_radius(cluster_locs)
        else:
            l2_radii = torch.full((M,), torch.inf)

        l2_radii.clamp_(max=radius_scale_factor * max_sample_distances)
        l2_radii.clamp_(max=torch.norm(support.width * 0.5).item())

        if not use_voronoi_radii: # TODO test if robust for small M and num_neigh
            num_neigh = max(int(M*0.05), 5)
            distance_closest_neighbor = torch.topk(distance_locs, num_neigh, dim=1, largest=False).values[:, num_neigh-1]
            l2_radii.clamp_(min=radius_scale_factor * distance_closest_neighbor / 2)
        
        return cls(
            support=support,
            region_locs=cluster_locs,
            region_l2_radii=l2_radii
        )


def compute_inner_cluster_max_l2_radii(samples: torch.Tensor, cluster_locs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum distance from samples to their assigned cluster locs.
    
    Args:
        samples: Sample points (n_samples, n_features)
        region_locs: Cluster center locations (k, n_features)  
        labels: Cluster assignments for each sample (n_samples,)
        
    Returns:
        l2_radii: Maximum distance for each cluster (k,)
    """
    k = cluster_locs.size(0)
    sample_to_center_distance = torch.norm(samples - cluster_locs[labels], dim=-1)
    return torch.zeros(k).scatter_reduce(0, labels, sample_to_center_distance, reduce='amax', include_self=False)


@torch.no_grad
def compute_voronoi_radius(points: torch.Tensor) -> torch.Tensor:
    """
    Compute a per-site Voronoi radius-like measure in R^d.

    Parameters
    ----------
    points : (N, d) array-like or torch.Tensor
        Input sites. Must be two-dimensional with N >= 2.

    Returns
    -------
    l2_radii : (N,) torch.Tensor of float64
        Half the maximum pairwise Euclidean distance between
        the cell's Voronoi vertices (i.e., the radius). 
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
    l2_radii = np.full(points_np.shape[0], np.inf)
    bounded_mask = np.zeros(points_np.shape[0], dtype=bool)

    for i, reg_idx in enumerate(vor.point_region):
        region = vor.regions[reg_idx]
        if not region:
            l2_radii[i] = 0.0
            continue

        finite_idx = [v for v in region if v != -1]
        if len(finite_idx) == 0:
            # Extremely degenerate; no finite vertices found.
            l2_radii[i] = 0.0
            continue

        verts = vor.vertices[np.array(finite_idx, dtype=int)]

        if -1 in region:
            bounded_mask[i] = False
        else:
            # Bounded: radius is half the true diameter via vertex–vertex pairs.
            l2_radii[i] = (pdist(verts).max() / 2.0) if len(verts) > 1 else 0.0
            bounded_mask[i] = True

    return torch.from_numpy(l2_radii).to(dtype=points.dtype)
