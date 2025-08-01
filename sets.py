import torch
from utils import in_set, generate_grid_from_locs


class HyperRectangle:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper

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

    @staticmethod
    def from_eps(x, eps):
        lower, upper = x - eps, x + eps
        return HyperRectangle(lower, upper)

class Partition:
    def __init__(self,
                 support: HyperRectangle,
                 locs: torch.Tensor=None,
                 regions: HyperRectangle=None):
        if (locs is None and regions is None) or (locs is not None and regions is not None):
            raise ValueError("Either locs or regions should be provided")

        self.support = support

        if locs is not None:
            if not self._are_locs_in_grid(locs):
                raise ValueError("locs must be in a grid")

            self.locs = locs
            self.regions = self._get_regions()
        else:
            self.regions = regions
            self.locs = self._get_locs()

    @staticmethod
    def _are_locs_in_grid(locs: torch.Tensor):
        unique_elements_per_n = torch.tensor([torch.unique(locs[:, i]).numel() for i in range(locs.size(1))])
        return unique_elements_per_n.prod() == torch.unique(locs, dim=0).size(0)

    @property
    def num_locs(self):
        return self.locs.size(0)

    def _get_upper(self):
        pos_diff = (self.locs.unsqueeze(-3) - self.locs.unsqueeze(-2)).clip(0, torch.inf)
        mask = pos_diff == 0.
        pos_diff[mask] = torch.inf

        upper = self.locs + 0.5 * pos_diff.min(dim=-2).values

        upper = torch.minimum(
            torch.where(torch.isposinf(upper), self.support.upper, upper),
            self.support.upper
        )
        return upper

    def _get_lower(self):
        neg_diff = (self.locs.unsqueeze(-3) - self.locs.unsqueeze(-2)).clip(-torch.inf, 0)
        mask = neg_diff == 0.
        neg_diff[mask] = -torch.inf

        lower = self.locs + 0.5 * neg_diff.max(dim=-2).values

        lower = torch.maximum(
            torch.where(torch.isneginf(lower), self.support.lower, lower),
            self.support.lower
        )
        return lower

    def _get_regions(self):

        upper = self._get_upper()
        lower = self._get_lower()

        return HyperRectangle(lower=lower, upper=upper)

    def _get_locs(self):
        return self.regions.center

    def sup_distance_within_regions(self):
        # Compute distances to lower and upper bounds
        dist_to_lower = torch.abs(self.locs - self.regions.lower)
        dist_to_upper = torch.abs(self.regions.upper - self.locs)

        # Choose further vertice
        furthest_corner = torch.where(dist_to_lower > dist_to_upper, self.regions.lower, self.regions.upper)

        # Compute distance
        squared_diff = (self.locs - furthest_corner) ** 2
        maximum_squared_distance = torch.sum(squared_diff, dim=1)

        return maximum_squared_distance

    def distance_locs(self):

        squared_distance = torch.cdist(self.locs, self.locs, p=2) ** 2
        return squared_distance

    def refine(self,
               samples: torch.Tensor,
               prob_thr: float = 0.01,
               diam_thr: float = 0.1):

        locs = self.locs
        regions = self.regions

        # Compute empirical prob in each region
        num_samples = samples.shape[0]
        n_set = in_set(samples=samples, regions=regions, include_complement=False)
        empirical = n_set / num_samples

        # Compute regions diameter
        diameters = torch.max(torch.abs(regions.upper - regions.lower), dim=1).values

        # Include new points in regions with high prob or high diameter
        mask = empirical.max() > prob_thr or diameters.max() > diam_thr
        to_upper = locs[mask] + (regions.upper[mask] - locs[mask]) / 2
        to_lower = locs[mask] + (regions.lower[mask] - locs[mask]) / 2

        locs = torch.cat([to_lower.squeeze(dim=0), to_upper.squeeze(dim=0)], dim=0)
        grid = generate_grid_from_locs(locs)
        partition = Partition(locs=grid, support=self.support)

        return partition