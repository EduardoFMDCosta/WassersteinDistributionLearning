import torch
from distributions import MultivariateUniform, TruncatedMultivariateNormal, MixtureTruncatedMultivariateNormal
from sets import HyperRectangle


def get_support_assumption(support_assumption, **kwargs):
    lower = torch.as_tensor(support_assumption[0])
    upper = torch.as_tensor(support_assumption[1])
    return HyperRectangle(lower, upper)

def get_support(distribution, support, **kwargs):
    if distribution == 'Uniform':
        lower = torch.as_tensor(support[0])
        upper = torch.as_tensor(support[1])
        return HyperRectangle(lower, upper)
    elif distribution == 'Gaussian':
        return get_support_assumption(**kwargs)
    elif distribution == 'GaussianMixture':
        return get_support_assumption(**kwargs)
    else:
        raise ValueError('Unknown distribution, thus cannot define support.')


def construct_uniform(support, support_assumption, **kwargs) -> MultivariateUniform:
    return MultivariateUniform(low=torch.as_tensor(support[0]), high=torch.as_tensor(support[1]))

def construct_trunc_mult_norm(mean, covariance_matrix, support_assumption, **kwargs) -> TruncatedMultivariateNormal:
    return TruncatedMultivariateNormal(
        loc=torch.as_tensor(mean), 
        scale=torch.as_tensor(covariance_matrix) ** 0.5,
        a=torch.as_tensor(support_assumption[0]),
        b=torch.as_tensor(support_assumption[1])
    )

def construct_mixture_trunc_mult_norm(weight, mean, covariance_matrix, support_assumption, **kwargs) -> MixtureTruncatedMultivariateNormal:
    mixture_distribution = torch.distributions.Categorical(probs=torch.as_tensor(weight))

    component_distribution = TruncatedMultivariateNormal(
        loc=torch.as_tensor(mean), 
        scale=torch.tensor(covariance_matrix) ** 0.5,
        a=torch.as_tensor(support_assumption[0]).expand(len(mean), -1),
        b=torch.as_tensor(support_assumption[1]).expand(len(mean), -1)
    )

    return MixtureTruncatedMultivariateNormal(mixture_distribution=mixture_distribution, component_distribution=component_distribution)


def get_distribution(distribution, **kwargs):
    if distribution == 'Uniform':
        return construct_uniform(**kwargs)
    elif distribution == 'Gaussian':
        return construct_trunc_mult_norm(**kwargs)
    elif distribution == 'GaussianMixture':
        return construct_mixture_trunc_mult_norm(**kwargs)
    else:
        raise ValueError('Unknown distribution.')