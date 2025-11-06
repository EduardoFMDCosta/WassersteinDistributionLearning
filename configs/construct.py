from typing import Optional, List, Optional, Union
import torch
from distributions import MultivariateUniform, TruncatedMultivariateNormal, MixtureTruncatedMultivariateNormal, CategoricalFloat
from sets import HyperRectangle


def get_support_assumption(
    num_dims: int,
    support_linf_radius: Optional[float] = None, 
    support_linf_radius_assumed: Optional[float] = None, 
    **kwargs
):
    if support_linf_radius_assumed is not None:
        return HyperRectangle.from_eps(x=torch.zeros(num_dims), eps=support_linf_radius_assumed)
    elif support_linf_radius is not None:
        return HyperRectangle.from_eps(x=torch.zeros(num_dims), eps=torch.as_tensor(support_linf_radius))
    else:
        raise ValueError("Either 'support' or 'support_linf_radius_assumed' must be provided.")
    
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
        return torch.ones(num_dims) * (variance ** 0.5) * (1 / num_dims**0.5)
    else:
        return torch.as_tensor(variance) ** 0.5

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


def get_distribution(distribution, **kwargs):
    if distribution == 'Uniform':
        return construct_uniform(**kwargs)
    elif distribution == 'Gaussian':
        return construct_trunc_mult_norm(**kwargs)
    elif distribution == 'GaussianMixture':
        return construct_mixture_trunc_mult_norm(**kwargs)
    elif distribution == 'Discrete':
        return construct_random_categorical_float(**kwargs)
    else:
        raise ValueError('Unknown distribution.')