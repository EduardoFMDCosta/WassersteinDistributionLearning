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

    def distance_matrix(self, support_assumption, p=2):
        n, d = self.lower.shape

        lower_full_partition = torch.cat([self.lower, support_assumption.lower.unsqueeze(0)], dim=0)
        upper_full_partition = torch.cat([self.upper, support_assumption.upper.unsqueeze(0)], dim=0)

        lower_i = lower_full_partition[:, None, :]
        upper_i = upper_full_partition[:, None, :]
        lower_j = lower_full_partition[None, :, :]
        upper_j = upper_full_partition[None, :, :]

        # Compute both possible max distances
        d1 = torch.norm(lower_i - upper_j, dim=2, p=p) ** p
        d2 = torch.norm(upper_i - lower_j, dim=2, p=p) ** p

        return torch.maximum(d1, d2)

    def distance_centers(self, support_assumption, p=2):

        centers = self.center
        center_complement = support_assumption.center.unsqueeze(dim=0)

        locations = torch.cat([centers, center_complement], dim=0)

        distance_matrix = torch.cdist(locations, locations, p=p) ** p
        return distance_matrix