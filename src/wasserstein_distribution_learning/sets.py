from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import heapq
import torch
from torch_kmeans import KMeans

import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist

from .utils import _detect_modes


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
        return torch.all((point >= self.lower) & (point <= self.upper), dim=-1)

    @staticmethod
    def from_eps(x, eps):
        lower, upper = x - eps, x + eps
        return HyperRectangle(lower, upper)


class Partition(ABC):

    def __init__(
        self,
        support: Optional['HyperRectangle'],
        region_locs: torch.Tensor,
        region_l2_radii: torch.Tensor,
    ):
        self.support = support
        self.region_locs = region_locs
        self.region_l2_radii = region_l2_radii
        self.l2_distance_locs_to_locs = torch.cdist(self.locs, self.locs, p=2)
        self.l1_distance_locs_to_locs = torch.cdist(self.locs, self.locs, p=1)

    def __len__(self) -> int:
        return self.locs.size(0)

    @property
    def ndim(self) -> int:
        if self.support is not None:
            return self.support.ndim
        return self.region_locs.size(-1)

    @property
    def locs(self) -> torch.Tensor:
        if self.support is not None:
            outer_loc = self.support.center.unsqueeze(0)
        else:
            outer_loc = torch.zeros(1, self.ndim, device=self.region_locs.device, dtype=self.region_locs.dtype)
        return torch.cat((self.region_locs, outer_loc), dim=0)

    @property
    def l2_radii(self) -> torch.Tensor:
        if self.support is not None:
            outer_radius = torch.norm(self.support.width).unsqueeze(0) / 2.
        else:
            outer_radius = torch.tensor([torch.inf], dtype=self.region_l2_radii.dtype)
        return torch.cat((self.region_l2_radii, outer_radius))

    @property
    def l2_distance_locs_to_region(self) -> torch.Tensor:
        return self.l2_distance_locs_to_locs + self.l2_radii.unsqueeze(-1)

    @property
    @abstractmethod
    def l1_radii(self) -> torch.Tensor: ...

    @property
    def l1_distance_locs_to_region(self) -> torch.Tensor:
        return self.l1_distance_locs_to_locs + self.l1_radii.unsqueeze(-1)

    @classmethod
    @abstractmethod
    def from_samples(cls, support, samples: torch.Tensor, M: int, **kwargs): ...


class HyperRectanglePartition(Partition):
    def __init__(
        self,
        support: 'HyperRectangle',
        region_locs: torch.Tensor,
        region_lower: torch.Tensor,
        region_upper: torch.Tensor
    ):
        self.region_lower = region_lower
        self.region_upper = region_upper

        dist_to_lower = torch.abs(region_lower - region_locs)
        dist_to_upper = torch.abs(region_upper - region_locs)
        max_dist_per_dim = torch.max(dist_to_lower, dist_to_upper)
        region_l2_radii = torch.sqrt(torch.sum(max_dist_per_dim ** 2, dim=-1))
        self.region_l1_radii = torch.sum(max_dist_per_dim, dim=-1)

        super().__init__(support=support, region_locs=region_locs, region_l2_radii=region_l2_radii)

    @property
    def l1_radii(self) -> torch.Tensor:
        if self.support is not None:
            outer_l1_radius = torch.sum(self.support.width).unsqueeze(0) / 2.
        else:
            outer_l1_radius = torch.tensor([torch.inf], dtype=self.region_l1_radii.dtype)
        return torch.cat((self.region_l1_radii, outer_l1_radius))

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        return torch.all((points.unsqueeze(1) >= self.region_lower) & (points.unsqueeze(1) <= self.region_upper), dim=-1)

    @classmethod
    def from_samples(
            cls,
            support: Optional['HyperRectangle'],
            samples: torch.Tensor,
            M: int,
            n_modes_max: int = 5,
            **kwargs
        ):
        assert len(samples.shape) == 2
        nsamples, ndim = samples.shape
        M = min(M, nsamples)

        bsp_bound = support if support is not None else HyperRectangle(
            lower=samples.min(dim=0).values,
            upper=samples.max(dim=0).values,
        )
        assert bsp_bound.ndim == ndim

        n_modes = _detect_modes(
            samples.detach().cpu().numpy().astype(np.float64),
            n_max=min(n_modes_max, M),
        )

        if n_modes > 1:
            try:
                from sklearn.mixture import GaussianMixture
                mode_labels = GaussianMixture(n_components=n_modes, random_state=0).fit_predict(samples.detach().cpu().numpy())
                mode_labels = torch.from_numpy(mode_labels).to(samples.device)
            except Exception:
                n_modes = 1
                mode_labels = torch.zeros(nsamples, dtype=torch.long, device=samples.device)
        else:
            mode_labels = torch.zeros(nsamples, dtype=torch.long, device=samples.device)

        heap: list = []
        uid = 0
        all_idx = torch.arange(nsamples, device=samples.device)
        for k in range(n_modes):
            idx = all_idx[mode_labels == k]
            if len(idx) == 0:
                continue
            pts = samples[idx]
            heapq.heappush(heap, (-len(idx), uid, pts.min(0).values, pts.max(0).values, idx))
            uid += 1

        while len(heap) < M:
            neg_n, _, lo, hi, idx = heapq.heappop(heap)
            pts = samples[idx]

            if len(idx) < 2:
                heapq.heappush(heap, (neg_n, uid, lo, hi, idx))
                uid += 1
                break

            d = int(pts.var(dim=0).argmax().item())
            split_val = float(pts[:, d].median().item())
            left_mask  = pts[:, d] <= split_val
            right_mask = ~left_mask

            if not left_mask.any() or not right_mask.any():
                heapq.heappush(heap, (neg_n, uid, lo, hi, idx))
                uid += 1
                break

            for mask in (left_mask, right_mask):
                child_idx = idx[mask]
                child_pts = samples[child_idx]
                heapq.heappush(heap, (-len(child_idx), uid, child_pts.min(0).values, child_pts.max(0).values, child_idx))
                uid += 1

        n_boxes = len(heap)
        cluster_locs = torch.zeros(n_boxes, ndim, device=samples.device, dtype=samples.dtype)
        lower        = torch.zeros(n_boxes, ndim, device=samples.device, dtype=samples.dtype)
        upper        = torch.zeros(n_boxes, ndim, device=samples.device, dtype=samples.dtype)

        for k, (_, _, lo, hi, idx) in enumerate(heap):
            cluster_locs[k] = samples[idx].mean(dim=0)
            lower[k]        = lo
            upper[k]        = hi

        return cls(
            support=support,
            region_locs=cluster_locs,
            region_lower=lower,
            region_upper=upper,
        )


class BoundedVoronoiPartition(Partition):
    def __init__(
        self, 
        support: Optional[HyperRectangle], 
        region_locs: torch.Tensor,
        region_l2_radii: torch.Tensor
    ):
        super().__init__(support=support, region_locs=region_locs, region_l2_radii=region_l2_radii)

    @property
    def l1_radii(self) -> torch.Tensor:
        return (2**0.5) * self.l2_radii

    @classmethod
    def from_samples(
            cls,
            support: Optional[HyperRectangle],
            samples: torch.Tensor, 
            M: int, 
            radius_scale_factor: float = 1.5, 
            use_voronoi_radii: bool = False
        ):
        assert len(samples.shape) == 2
        if support is not None:
            assert support.ndim == samples.shape[-1]

        nsamples = samples.size(0)

        if nsamples > M:
            kmeans_torch = KMeans(n_clusters=M)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                import faiss
                import numpy as np

                X_np = samples.detach().cpu().numpy().astype(np.float32)
                kmeans = faiss.Kmeans(d=X_np.shape[1], k=M, niter=20, verbose=False, gpu=True)
                kmeans.train(X_np)
                cluster_locs = torch.from_numpy(kmeans.centroids)
                labels       = torch.from_numpy(kmeans.index.search(X_np, 1)[1][:, 0])

            else:
                with torch.no_grad():
                    cluster_result = kmeans_torch(samples.unsqueeze(0).float())
                cluster_locs = cluster_result.centers.squeeze(0).float()
                labels = cluster_result.labels.squeeze(0)

            max_sample_distances = compute_inner_cluster_max_l2_radii(samples, cluster_locs, labels)    
        else:
            M = nsamples
            cluster_locs = samples
            max_sample_distances = torch.zeros(M)

        distance_locs = torch.cdist(cluster_locs, cluster_locs, p=2)

        if use_voronoi_radii:
            l2_radii = compute_voronoi_radius(cluster_locs)
        else:
            l2_radii = torch.full((M,), torch.inf)

        l2_radii.clamp_(max=radius_scale_factor * max_sample_distances)
        if support is not None:
            l2_radii.clamp_(max=torch.norm(support.width * 0.5).item())

        if not use_voronoi_radii:
            num_neigh = max(int(M*0.05), min(5, M))
            distance_closest_neighbor = torch.topk(distance_locs, num_neigh, dim=1, largest=False).values[:, num_neigh-1]
            l2_radii.clamp_(min=radius_scale_factor * distance_closest_neighbor / 2)
        
        return cls(
            support=support,
            region_locs=cluster_locs,
            region_l2_radii=l2_radii
        )


def compute_inner_cluster_max_l2_radii(samples: torch.Tensor, cluster_locs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    k = cluster_locs.size(0)
    sample_to_center_distance = torch.norm(samples - cluster_locs[labels], dim=-1)
    return torch.zeros(k).scatter_reduce(0, labels, sample_to_center_distance, reduce='amax', include_self=False)


@torch.no_grad
def compute_voronoi_radius(points: torch.Tensor) -> torch.Tensor:
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
            l2_radii[i] = 0.0
            continue

        verts = vor.vertices[np.array(finite_idx, dtype=int)]

        if -1 in region:
            bounded_mask[i] = False
        else:
            l2_radii[i] = (pdist(verts).max() / 2.0) if len(verts) > 1 else 0.0
            bounded_mask[i] = True

    return torch.from_numpy(l2_radii).to(dtype=points.dtype)
