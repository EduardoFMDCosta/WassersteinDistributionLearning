from typing import Optional, List, Optional
import torch
from distributions import MultivariateUniform, TruncatedMultivariateNormal, MixtureTruncatedMultivariateNormal, CategoricalFloat
from sets import HyperRectangle


def get_support_assumption(
    num_dims: int,
    support: Optional[List[List[float]]] = None, 
    support_linf_radius_assumed: Optional[float] = None, 
    **kwargs
):
    if support_linf_radius_assumed is not None:
        return HyperRectangle.from_eps(x=torch.zeros(num_dims), eps=support_linf_radius_assumed)
    elif support is not None:
        return HyperRectangle(lower=torch.as_tensor(support[0]), upper=torch.as_tensor(support[1]))
    else:
        raise ValueError("Either 'support' or 'support_linf_radius_assumed' must be provided.")
    
def construct_uniform(support, **kwargs) -> MultivariateUniform:
    return MultivariateUniform(low=torch.as_tensor(support[0]), high=torch.as_tensor(support[1]))

def construct_trunc_mult_norm(mean, variance, support, **kwargs) -> TruncatedMultivariateNormal:
    return TruncatedMultivariateNormal(
        loc=torch.as_tensor(mean), 
        scale=torch.as_tensor(variance) ** 0.5,
        a=torch.as_tensor(support[0]),
        b=torch.as_tensor(support[1])
    )

def construct_mixture_trunc_mult_norm(weight, mean, variance, support, **kwargs) -> MixtureTruncatedMultivariateNormal:
    mixture_distribution = torch.distributions.Categorical(probs=torch.as_tensor(weight))

    component_distribution = TruncatedMultivariateNormal(
        loc=torch.as_tensor(mean), 
        scale=torch.tensor(variance) ** 0.5,
        a=torch.as_tensor(support[0]).expand(len(mean), -1),
        b=torch.as_tensor(support[1]).expand(len(mean), -1)
    )

    return MixtureTruncatedMultivariateNormal(mixture_distribution=mixture_distribution, component_distribution=component_distribution)

def construct_random_categorical_float(
    support_linf_radius_assumed: float, 
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