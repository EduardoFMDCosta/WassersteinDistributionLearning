from typing import Optional, List, Optional
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

def construct_trunc_mult_norm(
    num_dims: int,
    mean, 
    variance, 
    support_linf_radius: float, 
    **kwargs
) -> TruncatedMultivariateNormal:
    return TruncatedMultivariateNormal(
        loc=torch.as_tensor(mean), 
        scale=torch.as_tensor(variance) ** 0.5,
        a=torch.ones(num_dims) * -support_linf_radius,
        b=torch.ones(num_dims) * support_linf_radius
    )

def construct_mixture_trunc_mult_norm(
    num_dims: int,
    weight, 
    mean, 
    variance, 
    support_linf_radius: float, 
    **kwargs
) -> MixtureTruncatedMultivariateNormal:
    mixture_distribution = torch.distributions.Categorical(probs=torch.as_tensor(weight))
    component_distribution = TruncatedMultivariateNormal(
        loc=torch.as_tensor(mean), 
        scale=torch.tensor(variance) ** 0.5,
        a=torch.ones(num_dims) * -support_linf_radius,
        b=torch.ones(num_dims) * support_linf_radius
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