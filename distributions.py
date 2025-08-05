import torch
import torch.distributions as ds
import stable_trunc_gaussian


__all__ = ['MultivariateUniform', 'TruncatedMultivariateNormal', 'MixtureTruncatedMultivariateNormal']


class TruncatedMultivariateNormal(ds.Distribution):
    arg_constraints = {
        "loc": ds.constraints.real,
        "scale": ds.constraints.positive,
        "a": ds.constraints.dependent,
        "b": ds.constraints.dependent,
    }
    has_rsample = True
    support = ds.constraints.independent(ds.constraints.real, 1)

    def __init__(self, loc, scale, a, b, validate_args=False):
        assert loc.shape == scale.shape == a.shape == b.shape, "All shapes must match"
        self.loc = loc
        self.scale = scale
        self.a = a
        self.b = b
        self._base = stable_trunc_gaussian.TruncatedGaussian(loc, scale, a, b)
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        return self._base.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self._base.rsample(sample_shape)

    def log_prob(self, value):
        return self._base.log_prob(value)

    def entropy(self):
        return self._base.entropy()

    def cdf(self, value):
        return self._base.cdf(value)

    @property
    def mean(self):
        return self._base.mean

    @property
    def stddev(self):
        return self._base.stddev


class MixtureTruncatedMultivariateNormal(torch.distributions.MixtureSameFamily):
    def __init__(
            self,
            mixture_distribution: torch.distributions.Categorical,
            component_distribution: TruncatedMultivariateNormal
    ):
        super(MixtureTruncatedMultivariateNormal, self).__init__(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
            validate_args=False)


class MultivariateUniform(ds.Distribution):
    arg_constraints = {'low': ds.constraints.real, 'high': ds.constraints.real}
    support = ds.constraints.independent(ds.constraints.real, 1)
    has_rsample = True

    def __init__(self, low: torch.Tensor, high: torch.Tensor, validate_args=False):
        assert low.shape == high.shape, "low and high must have the same shape"
        self.low = low
        self.high = high
        self._univariate = ds.Uniform(low, high, validate_args=validate_args)
        batch_shape = self._univariate.batch_shape
        event_shape = batch_shape[-1:]
        super().__init__(batch_shape=batch_shape[:-1], event_shape=event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        return self._univariate.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self._univariate.rsample(sample_shape)

    def log_prob(self, value):
        log_probs = self._univariate.log_prob(value)
        return log_probs.sum(-1)

    def entropy(self):
        return self._univariate.entropy().sum(-1)
    

class CategoricalFloat(ds.Distribution):
    arg_constraints = {'probs': ds.constraints.simplex,
                       'locs': ds.constraints.real_vector}

    def __init__(
            self, 
            locs: torch.Tensor, 
            probs: torch.Tensor, 
            validate_args=None
    ):
        """ 
        only accepts single dimensionsl event_shape
        """

        if locs.shape[:-2] != probs.shape[:-1]:
            raise ValueError('locs and probs must have the same batch shape')
        if locs.shape[-2] != probs.shape[-1]:
            raise ValueError('locs and probs must have the same support size')

        self.locs = locs
        self.probs = probs / probs.sum(-1, keepdim=True)

        batch_shape = probs.shape[:-1]
        event_shape = torch.Size((locs.shape[-1],))
        self.num_components = probs.shape[-1]

        super(CategoricalFloat, self).__init__(
            batch_shape=batch_shape, 
            event_shape=event_shape,
            validate_args=validate_args
        )

    @property
    def mean(self):
        return torch.einsum('...i, ...ij->...j', self.probs, self.locs)

    @property
    def covariance_matrix(self):
        centered_locs = self.locs - self.mean.unsqueeze(-2)
        return torch.einsum('...e,...ei,...ej->...ij', self.probs, centered_locs, centered_locs)

    @property
    def variance(self):
        return torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample from the distribution.

        :param sample_shape: Shape of the samples to be drawn.
        :return: Samples drawn from the distribution.
        """
        if isinstance(sample_shape, tuple):
            sample_shape = torch.Size(sample_shape)
        elif isinstance(sample_shape, int):
            sample_shape = torch.Size((sample_shape,))

        flat_probs = self.probs.view(-1, self.num_components)
        flat_locs = self.locs.view(-1, self.num_components, *self.event_shape)

        indices = torch.multinomial(flat_probs, sample_shape.numel(), replacement=True)
        batch_indices = torch.arange(flat_probs.size(0), device=indices.device).repeat_interleave(indices.size(1))
        samples = flat_locs[batch_indices, indices.view(-1)]
        samples = samples.view(*sample_shape, *self.batch_shape, *self.event_shape)
        return samples