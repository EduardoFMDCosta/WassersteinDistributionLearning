import torch
from sets import HyperRectangle
import torch.distributions as ds

class Gaussian(ds.MultivariateNormal):
    def __init__(self, mean: torch.Tensor, covariance_matrix: torch.Tensor):
        super().__init__(loc=mean, covariance_matrix=covariance_matrix)

    def __call__(self, num_samples: int):
        return self.sample((num_samples,))

    def compute_probabilities(self, regions: HyperRectangle):

        mean = self.mean
        sigma = torch.sqrt(self.covariance_matrix.diagonal(dim1=-2, dim2=-1))

        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
            sigma = sigma.unsqueeze(0)

        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(0).expand_as(mean)

        lower_norm = (regions.lower.unsqueeze(0) - mean.unsqueeze(1)) / sigma.unsqueeze(1)
        upper_norm = (regions.upper.unsqueeze(0) - mean.unsqueeze(1)) / sigma.unsqueeze(1)

        normal = torch.distributions.Normal(0.0, 1.0)
        lower_cdf = normal.cdf(lower_norm)
        upper_cdf = normal.cdf(upper_norm)

        probs = torch.prod(upper_cdf - lower_cdf, dim=2)

        if probs.shape[0] == 1:
            probs = probs.squeeze(0)  # Squeeze it back to (n,) if means represent a sole Gaussian

        probs.clamp_(min=0.0, max=1.0)  # avoid numerical issues
        return probs

class Uniform:
    def __init__(self, support: HyperRectangle):
        self.support = support
        self.dim = support.lower.shape[-1]

    def __call__(self, num_samples: int):
        u = torch.rand((num_samples, self.dim))
        samples = self.support.lower + u * (self.support.upper - self.support.lower)
        return samples

    def compute_probabilities(self, regions: HyperRectangle):

        lower_overlap = torch.maximum(self.support.lower.unsqueeze(0), regions.lower)
        upper_overlap = torch.minimum(self.support.upper.unsqueeze(0), regions.upper)

        side_lengths = (upper_overlap - lower_overlap).clamp(min=0.0)
        intersection_volume = torch.prod(side_lengths, dim=-1)
        support_volume = torch.prod(self.support.upper - self.support.lower)

        probs = intersection_volume / support_volume
        return probs