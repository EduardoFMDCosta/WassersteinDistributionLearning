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
        """Check if a point is included in the hyperrectangle."""
        return torch.all((point >= self.lower) & (point <= self.upper), dim=-1)

    @staticmethod
    def from_eps(x, eps):
        lower, upper = x - eps, x + eps
        return HyperRectangle(lower, upper)


class Partition(ABC):
    """Base class for all partition types used in quantization.

    A partition divides the ambient space into M + 1 regions:
    - M bounded regions, each described by a centroid in ``region_locs`` and a
      bounding radius in ``region_l2_radii``.
    - 1 implicit complement (outer) region: everything not covered by the M
      bounded regions. Its representative location is the support centre (or
      the zero vector when ``support is None``) and its bounding radius is half
      the support diameter (or ``inf`` when ``support is None``).

    The complement set is always the **last** element of ``locs``, ``l2_radii``,
    ``l2_distance_locs_to_region``, and related tensors, and its probability is
    tracked explicitly in :class:`~quantization.UncertainQuantization`.
    """

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

        # Compute per-region radii before calling super (super needs region_l2_radii)
        dist_to_lower = torch.abs(region_lower - region_locs)
        dist_to_upper = torch.abs(region_upper - region_locs)
        max_dist_per_dim = torch.max(dist_to_lower, dist_to_upper)
        region_l2_radii = torch.sqrt(torch.sum(max_dist_per_dim ** 2, dim=-1))
        self.region_l1_radii = torch.sum(max_dist_per_dim, dim=-1)

        super().__init__(support=support, region_locs=region_locs, region_l2_radii=region_l2_radii)

    @property
    def l1_radii(self) -> torch.Tensor:
        return torch.cat((self.region_l1_radii, torch.sum(self.support.width).unsqueeze(0) / 2.))

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """
        Checks which region a set of new points belongs to.
        Returns an (N, M) boolean tensor.
        """
        # points: (N, D), lower: (M, D), upper: (M, D)
        # Reshape for broadcasting -> points: (N, 1, D)
        points_expanded = points.unsqueeze(1)
        
        # Check bounds: (N, M, D)
        in_bounds = (points_expanded >= self.region_lower) & (points_expanded <= self.region_upper)
        
        # Must be in bounds for all dimensions: (N, M)
        return torch.all(in_bounds, dim=-1)

    @classmethod
    def from_samples(
            cls,
            support: Optional['HyperRectangle'],
            samples: torch.Tensor,
            M: int,
            n_modes_max: int = 5,
            **kwargs
        ):
        """Build a disjoint hyper-rectangle partition via GMM-seeded BSP.

        Algorithm
        ---------
        1. Fit a GMM for k = 1 … min(n_modes_max, M) and pick k* by BIC.
        2. Assign every sample to its most-likely component; build one tight
           bounding box per component.  These are guaranteed to be disjoint
           because they start as bounding boxes of *disjoint* sample subsets.
        3. Iteratively split the box that contains the *most* samples (greedy
           max-heap).  Each split:
             - chooses the dimension with the highest sample variance in the box
               (most informative split direction);
             - cuts at the *median* sample value along that dimension
               (guarantees the two children share the samples as evenly as
               possible, avoiding degenerate empty-child situations).
           The two children are axis-aligned halves of the parent, so they are
           disjoint by construction — no intersection checks needed.
        4. Stop when the total number of boxes reaches M (or when no box can be
           split further).
        5. Centroids are set to the mean of samples inside each final box.
        """
        assert len(samples.shape) == 2, "Samples must be a 2D tensor (num_samples, num_features)"
        nsamples, ndim = samples.shape
        M = min(M, nsamples)

        # --- 0. Support ---
        if support is None:
            support = HyperRectangle(
                lower=samples.min(dim=0).values,
                upper=samples.max(dim=0).values,
            )
        assert support.ndim == ndim, "Support dimension must match sample features"

        # --- 1. Mode detection via GMM + BIC ---
        n_modes = _detect_modes(
            samples.detach().cpu().numpy().astype(np.float64),
            n_max=min(n_modes_max, M),
        )

        # --- 2. Initial boxes: one per GMM component ---
        if n_modes > 1:
            try:
                from sklearn.mixture import GaussianMixture
                mode_labels = GaussianMixture(
                    n_components=n_modes, random_state=0
                ).fit_predict(samples.detach().cpu().numpy())
                mode_labels = torch.from_numpy(mode_labels).to(samples.device)
            except Exception:
                n_modes = 1
                mode_labels = torch.zeros(nsamples, dtype=torch.long, device=samples.device)
        else:
            mode_labels = torch.zeros(nsamples, dtype=torch.long, device=samples.device)

        # Build one tight bounding box per non-empty mode.
        # Bounding boxes of disjoint sample subsets are already disjoint.
        # heap entries: (-n_samples_in_box, unique_id, lo, hi, idx_tensor)
        # unique_id breaks ties so heapq never compares tensors.
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

        # --- 3. BSP: split until M boxes ---
        while len(heap) < M:
            neg_n, _, lo, hi, idx = heapq.heappop(heap)
            pts = samples[idx]

            if len(idx) < 2:
                # Box has only one sample; can't split — put back and stop.
                heapq.heappush(heap, (neg_n, uid, lo, hi, idx))
                uid += 1
                break

            # Split dimension: highest variance among samples in this box.
            d = int(pts.var(dim=0).argmax().item())

            # Split value: median → each child gets roughly half the samples.
            split_val = float(pts[:, d].median().item())

            left_mask  = pts[:, d] <= split_val
            right_mask = ~left_mask

            # Guard: if all samples fall on one side (e.g. all identical in d),
            # skip this box — it can't be split meaningfully.
            if not left_mask.any() or not right_mask.any():
                heapq.heappush(heap, (neg_n, uid, lo, hi, idx))
                uid += 1
                break

            for mask in (left_mask, right_mask):
                child_idx = idx[mask]
                child_pts = samples[child_idx]
                heapq.heappush(
                    heap,
                    (-len(child_idx), uid,
                     child_pts.min(0).values,
                     child_pts.max(0).values,
                     child_idx)
                )
                uid += 1

        # --- 4. Extract final boxes ---
        n_boxes = len(heap)
        cluster_locs = torch.zeros(n_boxes, ndim, device=samples.device, dtype=samples.dtype)
        lower        = torch.zeros(n_boxes, ndim, device=samples.device, dtype=samples.dtype)
        upper        = torch.zeros(n_boxes, ndim, device=samples.device, dtype=samples.dtype)

        for k, (_, _, lo, hi, idx) in enumerate(heap):
            cluster_locs[k] = samples[idx].mean(dim=0)  # centroid = mean of box samples
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
        assert len(samples.shape) == 2, "Samples must be a 2D tensor (num_samples, num_features)"
        if support is not None:
            assert support.ndim == samples.shape[-1], "Support dimension must match sample features"

        nsamples = samples.size(0)

        if nsamples > M:
            kmeans_torch = KMeans(n_clusters=M)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                import faiss
                import numpy as np

                # FAISS expects float32 on CPU
                X_np = samples.detach().cpu().numpy().astype(np.float32)   # (N, D)

                # build + run GPU k-means
                kmeans = faiss.Kmeans(
                    d=X_np.shape[1],
                    k=M,
                    niter=20,            # match your torch default if needed
                    verbose=False,
                    gpu=True
                )

                kmeans.train(X_np)

                # outputs (CPU numpy)
                centroids_np = kmeans.centroids                    # (M, D)
                _, labels_np = kmeans.index.search(X_np, 1)        # (N, 1)

                # convert to torch tensors (CPU)
                cluster_locs = torch.from_numpy(centroids_np)      # float32, (M, D)
                labels       = torch.from_numpy(labels_np[:, 0])   # int64, (N,)

            else: # CPU fallback
                with torch.no_grad():
                    cluster_result = kmeans_torch(samples.unsqueeze(0).float())   # (1, N, D)

                cluster_locs = cluster_result.centers.squeeze(0).float()
                labels = cluster_result.labels.squeeze(0)

            max_sample_distances = compute_inner_cluster_max_l2_radii(samples, cluster_locs, labels)    
        else:
            M = nsamples
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
