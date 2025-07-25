import torch

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
                 locs: torch.Tensor,
                 support: HyperRectangle):
        if not self._are_locs_in_grid(locs):
            raise ValueError("locs must be in a grid")

        self.support = support
        self.locs = locs
        self.regions = self._get_regions()

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