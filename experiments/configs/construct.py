from typing import Optional, List, Optional, Union, Callable, Tuple
import math
import torch
import os
from distributions import MultivariateUniform, TruncatedMultivariateNormal, MixtureTruncatedMultivariateNormal, CategoricalFloat
from wasserstein_distribution_learning.sets import HyperRectangle
from ucimlrepo import fetch_ucirepo
import medmnist

import pandas as pd


def get_support_assumption(
    num_dims: int,
    support_linf_radius: Optional[float] = None, 
    support_linf_radius_assumed: Optional[float] = None, 
    **kwargs
):
    if support_linf_radius_assumed is not None and not math.isinf(support_linf_radius_assumed):
        return HyperRectangle.from_eps(x=torch.zeros(num_dims), eps=support_linf_radius_assumed)
    elif support_linf_radius is not None and not math.isinf(support_linf_radius):
        return HyperRectangle.from_eps(x=torch.zeros(num_dims), eps=torch.as_tensor(support_linf_radius))
    else:
        return None
    
def construct_uniform(
    num_dims: int,
    support_linf_radius: float, 
    **kwargs
) -> MultivariateUniform:
    return MultivariateUniform(
        low=torch.ones(num_dims) * -support_linf_radius, 
        high=torch.ones(num_dims) * support_linf_radius
    )

def construct_loc(
    num_dims: int,
    mean: Union[float, List[float]]
) -> torch.Tensor:
    if isinstance(mean, float):
        return torch.ones(num_dims) * mean
    else:
        return torch.as_tensor(mean)

def construct_scale(
    num_dims: int,
    variance: Union[float, List[float]]
) -> torch.Tensor:
    if isinstance(variance, float):
        return torch.ones(num_dims) * (variance ** 0.5)
    else:
        return torch.as_tensor(variance) ** 0.5

def construct_gaussian(
    num_dims: int,
    mean: Union[float, List[float]],
    variance: Union[float, List[float]],
    **kwargs
):
    """Unbounded isotropic Gaussian with diagonal covariance."""
    loc = construct_loc(num_dims=num_dims, mean=mean)
    scale = construct_scale(num_dims=num_dims, variance=variance)
    return torch.distributions.Independent(
        torch.distributions.Normal(loc=loc, scale=scale), 1
    )

def construct_gaussian_mixture(
    num_dims: int,
    weight: List[float],
    mean: Union[List[float], List[List[float]]],
    variance: Union[List[float], List[List[float]]],
    **kwargs
):
    """Unbounded Gaussian mixture with diagonal covariance."""
    assert len(weight) == len(mean) == len(variance), "Inconsistent number of components."
    mixture = torch.distributions.Categorical(probs=torch.as_tensor(weight))
    loc   = torch.stack([construct_loc(num_dims=num_dims, mean=m) for m in mean])
    scale = torch.stack([construct_scale(num_dims=num_dims, variance=v) for v in variance])
    component = torch.distributions.Independent(
        torch.distributions.Normal(loc=loc, scale=scale), 1
    )
    return torch.distributions.MixtureSameFamily(mixture, component)

def construct_trunc_mult_norm(
    num_dims: int,
    mean: Union[float, List[float]], 
    variance: Union[float, List[float]],
    support_linf_radius: float, 
    **kwargs
) -> TruncatedMultivariateNormal:
    return TruncatedMultivariateNormal(
        loc=construct_loc(num_dims=num_dims, mean=mean),
        scale=construct_scale(num_dims=num_dims, variance=variance),
        a=torch.ones(num_dims) * -support_linf_radius,
        b=torch.ones(num_dims) * support_linf_radius
    )

def construct_mixture_trunc_mult_norm(
    num_dims: int,
    weight: List[float], 
    mean: Union[List[float], List[List[float]]], 
    variance: Union[List[float], List[List[float]]],
    support_linf_radius: float, 
    **kwargs
) -> MixtureTruncatedMultivariateNormal:
    assert len(weight) == len(mean) == len(variance), "Inconsistent number of components."

    mixture_distribution = torch.distributions.Categorical(probs=torch.as_tensor(weight))

    loc = torch.stack([construct_loc(num_dims=num_dims, mean=m) for m in mean])
    scale = torch.stack([construct_scale(num_dims=num_dims, variance=v) for v in variance])

    component_distribution = TruncatedMultivariateNormal(
        loc=loc,
        scale=scale,
        a=torch.ones(len(weight), num_dims) * -support_linf_radius,
        b=torch.ones(len(weight), num_dims) * support_linf_radius
    )
    return MixtureTruncatedMultivariateNormal(mixture_distribution=mixture_distribution, component_distribution=component_distribution)

def construct_random_categorical_float(
    support_linf_radius_assumed: float,      # TODO use support_linf_radius here?
    support_size: int, 
    num_dims: int,
    **kwargs
):
    return CategoricalFloat(
        probs=torch.ones(support_size) / support_size, 
        locs=(torch.rand(support_size, num_dims) * 2 - 1) - support_linf_radius_assumed
    )

class MinMaxNormalizer:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.registered = False

    def fit(
        self,
        X: torch.Tensor,
        min: float | torch.Tensor | None = None,
        max: float | torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            Data used only if min or max is None.
        min : float or Tensor of shape (d,), optional
        max : float or Tensor of shape (d,), optional
        """

        if min is None:
            self.min = X.min(dim=0).values
        else:
            self.min = torch.as_tensor(min, device=X.device, dtype=X.dtype)
            if self.min.ndim == 0:
                self.min = self.min.expand(X.shape[1])

        if max is None:
            self.max = X.max(dim=0).values
        else:
            self.max = torch.as_tensor(max, device=X.device, dtype=X.dtype)
            if self.max.ndim == 0:
                self.max = self.max.expand(X.shape[1])

        if self.min.shape != self.max.shape:
            raise ValueError("min and max must have the same shape")

        self.registered = True
        return self

    def __call__(self, X):
        assert self.registered
        scale = (self.max - self.min).clamp_min(self.eps)
        return (X - self.min) / scale - 0.5

class EmpiricalDistribution:
    """
    Stateful sampling without replacement across calls until reset().

    - sample(n) consumes indices from an internal shuffled buffer.
    - once exhausted, further sampling raises unless you reset().
    - keeps track of the indices sampled since last reset.
    """
    def __init__(
        self,
        X: torch.Tensor,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        *,
        generator: Optional[torch.Generator] = None,
        device_for_perm: Optional[torch.device] = None,
    ):
        self.X = X
        self.transform = transform
        self.generator = generator
        self.device_for_perm = device_for_perm  # where to create randperm (often CPU)

        self._perm: Optional[torch.Tensor] = None
        self._cursor: int = 0
        self._sampled_indices: list[torch.Tensor] = []

        self.reset()

    def reset(self, *, reshuffle: bool = True) -> None:
        """Reset the internal state. By default reshuffles indices."""
        N = len(self)
        dev = self.device_for_perm if self.device_for_perm is not None else torch.device("cpu")

        if reshuffle:
            self._perm = torch.randperm(N, generator=self.generator, device=dev)
        else:
            self._perm = torch.arange(N, device=dev)

        self._cursor = 0
        self._sampled_indices.clear()

    def remaining(self) -> int:
        """How many unused points are left before you must reset."""
        return len(self) - self._cursor

    @property
    def sampled_indices(self) -> torch.Tensor:
        """All indices sampled since last reset, concatenated."""
        if len(self._sampled_indices) == 0:
            # return an empty long tensor on CPU by default
            return torch.empty(0, dtype=torch.long)
        return torch.cat(self._sampled_indices, dim=0)

    @staticmethod
    def _parse_n(n: Union[int, Tuple[int], torch.Size]) -> int:
        # Accept int
        if isinstance(n, int):
            n_int = n
        # Accept tuple of length 1: (n,)
        elif isinstance(n, tuple):
            if len(n) != 1:
                raise TypeError(f"Expected a tuple of length 1 for n, got length {len(n)}: {n}")
            if not isinstance(n[0], int):
                raise TypeError(f"Expected n[0] to be int, got {type(n[0])}: {n}")
            n_int = n[0]
        # Accept torch.Size of length 1: torch.Size([n])
        elif isinstance(n, torch.Size):
            if len(n) != 1:
                raise TypeError(f"Expected torch.Size of length 1 for n, got {n}")
            n_int = int(n[0])
        else:
            raise TypeError(f"n must be int, (int,), or torch.Size([int]). Got {type(n)}: {n}")

        if n_int < 0:
            raise ValueError(f"n must be non-negative, got {n_int}")
        return n_int

    def sample(self, n: Union[int, Tuple[int], torch.Size], *, return_indices: bool = False)  -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        n = self._parse_n(n)
        
        if self._perm is None:
            raise RuntimeError("Internal permutation buffer is not initialized. Call reset().")

        if n > self.remaining():
            raise ValueError(
                f"Cannot sample {n} points without replacement: only {self.remaining()} remaining. "
                f"Call reset() to start a new pass."
            )

        idx = self._perm[self._cursor : self._cursor + n]
        self._cursor += n
        self._sampled_indices.append(idx.detach().clone())

        # idx might live on CPU; advanced indexing works fine either way.
        X = self.X[idx.to(self.X.device)]

        if self.transform is not None:
            X = self.transform(X)

        return (X, idx) if return_indices else X

    def __len__(self) -> int:
        return self.X.shape[0]

def construct_uci_turbine(**kwargs) -> EmpiricalDistribution:
    df = fetch_ucirepo(id=551).data.features
    df.pop('year')

    features = torch.from_numpy(df.to_numpy()).float()
    transform = MinMaxNormalizer().fit(features)
    return EmpiricalDistribution(features, transform=transform)

def construct_uci_miniboone(base_dir: str, **kwargs) -> EmpiricalDistribution:
    df = pd.read_csv(os.path.join(base_dir, "data", "MiniBooNE.csv"), header=None, delimiter=';')

    features = torch.from_numpy(df.to_numpy()).float()
    transform = MinMaxNormalizer().fit(features)
    return EmpiricalDistribution(features, transform=transform)

def construct_octmnist(**kwargs) -> EmpiricalDistribution:
    ds = medmnist.OCTMNIST(split='train', download=True)
    features = torch.from_numpy(ds.imgs).float().flatten(start_dim=1)
    transform = MinMaxNormalizer().fit(features, min=0., max=255.)
    return EmpiricalDistribution(features, transform=transform)

def get_distribution(distribution, **kwargs):
    if distribution == 'Uniform':
        return construct_uniform(**kwargs)
    elif distribution == 'Gaussian':
        return construct_gaussian(**kwargs)
    elif distribution == 'GaussianMixture':
        return construct_gaussian_mixture(**kwargs)
    elif distribution == 'TruncatedGaussian':
        return construct_trunc_mult_norm(**kwargs)
    elif distribution in ('TruncatedGaussianMixture', 'GaussianMixture'):
        return construct_mixture_trunc_mult_norm(**kwargs)
    elif distribution == 'Discrete':
        return construct_random_categorical_float(**kwargs)
    elif distribution == "UCI-Turbine":
        return construct_uci_turbine(**kwargs)
    elif distribution == "UCI-MiniBooNE":
        return construct_uci_miniboone(**kwargs)
    elif distribution == "OCTMNIST":
        return construct_octmnist(**kwargs)
    else:
        raise ValueError('Unknown distribution.')