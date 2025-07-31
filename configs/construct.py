import torch
from distributions import Uniform, Gaussian
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
    else:
        return ValueError('Unknown distribution, thus cannot define support.')

def construct_diag_gaussian_dist(mean, covariance_matrix, **kwargs):
    loc_dist = torch.as_tensor(mean)
    covariance_dist = torch.diag(torch.as_tensor(covariance_matrix))
    return Gaussian(mean=loc_dist, covariance_matrix=covariance_dist)

def get_distribution(distribution, **kwargs):
    if distribution == 'Uniform':
        support = get_support(distribution, **kwargs)
        return Uniform(support=support)
    elif distribution == 'Gaussian':
        return construct_diag_gaussian_dist(**kwargs)
    else:
        return ValueError('Unknown distribution.')