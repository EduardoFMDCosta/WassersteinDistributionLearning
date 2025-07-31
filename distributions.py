import torch
from sets import HyperRectangle
import torch.distributions as ds

class Gaussian(ds.MultivariateNormal):
    def __init__(self, mean: torch.Tensor, covariance_matrix: torch.Tensor, **kwargs):
        super().__init__(loc=mean, covariance_matrix=covariance_matrix)

    def __call__(self, num_samples: int, support_assumption: HyperRectangle):
        max_iter = 100
        remaining = num_samples

        # Initial sampling
        samples = self.sample((num_samples,))

        for i in range(max_iter):
            reject = ((samples < support_assumption.lower) | (samples > support_assumption.upper)).any(dim=-1)
            remaining = int(reject.sum().item())
            if remaining == 0:
                break
            else:
                samples[reject, :] = self.sample((remaining,))

        assert remaining == 0, "Maximum rejection iterations reached in sampling."

        return samples

class Uniform:
    def __init__(self, support: HyperRectangle, **kwargs):
        self.support = support
        self.dim = support.lower.shape[-1]

    def __call__(self, num_samples: int, support_assumption: HyperRectangle):
        u = torch.rand((num_samples, self.dim))
        samples = self.support.lower + u * (self.support.upper - self.support.lower)

        valid = (samples >= support_assumption.lower) & (samples <= support_assumption.upper)
        assert valid.all(), "Distribution support must be contained in support assumption."

        return samples

    def compute_probabilities(self, regions: HyperRectangle):

        lower_overlap = torch.maximum(self.support.lower.unsqueeze(0), regions.lower)
        upper_overlap = torch.minimum(self.support.upper.unsqueeze(0), regions.upper)

        side_lengths = (upper_overlap - lower_overlap).clamp(min=0.0)
        intersection_volume = torch.prod(side_lengths, dim=-1)
        support_volume = torch.prod(self.support.upper - self.support.lower)

        probs = intersection_volume / support_volume
        return probs