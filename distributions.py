import torch
from sets import HyperRectangle
import torch.distributions as ds

class Distributions:
    def generate_samples(self, num_samples: int):
        pass

    def __call__(self, num_samples: int, support_assumption: HyperRectangle):
        max_iter = 100
        remaining = num_samples

        # Initial sampling
        samples = self.generate_samples(num_samples)

        for i in range(max_iter):
            reject = ((samples < support_assumption.lower) | (samples > support_assumption.upper)).any(dim=-1)
            remaining = int(reject.sum().item())
            if remaining == 0:
                break
            else:
                samples[reject, :] = self.generate_samples(remaining)

        assert remaining == 0, "Maximum rejection iterations reached in sampling."

        return samples

class Gaussian(ds.MultivariateNormal, Distributions):
    def __init__(self, mean: torch.Tensor, covariance_matrix: torch.Tensor, **kwargs):
        super().__init__(loc=mean, covariance_matrix=covariance_matrix)

    def generate_samples(self, num_samples: int):
        return self.sample((num_samples,))

class GaussianMixture(torch.distributions.MixtureSameFamily, Distributions):
    def __init__(
            self,
            mixture_distribution: torch.distributions.Categorical,
            component_distribution: Gaussian
    ):
        super(GaussianMixture, self).__init__(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
            validate_args=False)

    def generate_samples(self, num_samples: int):
        return self.sample((num_samples,))

class Uniform(Distributions):
    def __init__(self, support: HyperRectangle, **kwargs):
        self.support = support
        self.dim = support.lower.shape[-1]

    def generate_samples(self, num_samples: int):
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