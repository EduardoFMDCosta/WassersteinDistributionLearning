import torch
from distributions import Uniform, Gaussian, GaussianMixture
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

def construct_diag_gaussian(mean, covariance_matrix, **kwargs):
    loc_dist = torch.as_tensor(mean)
    covariance_dist = torch.diag(torch.as_tensor(covariance_matrix))
    return Gaussian(mean=loc_dist, covariance_matrix=covariance_dist)

def construct_gaussian_mixture(weight, mean, covariance_matrix, **kwargs):
    weight = torch.as_tensor(weight)
    mixture_distribution = torch.distributions.Categorical(probs=weight)

    loc_dist = torch.as_tensor(mean)
    covariance_dist = torch.diag_embed(torch.tensor(covariance_matrix))
    component_distribution = Gaussian(mean=loc_dist, covariance_matrix=covariance_dist)

    return GaussianMixture(mixture_distribution=mixture_distribution, component_distribution=component_distribution)


def get_distribution(distribution, **kwargs):
    if distribution == 'Uniform':
        support = get_support(distribution, **kwargs)
        return Uniform(support=support)
    elif distribution == 'Gaussian':
        return construct_diag_gaussian(**kwargs)
    elif distribution == 'GaussianMixture':
        return construct_gaussian_mixture(**kwargs)
    else:
        raise ValueError('Unknown distribution.')