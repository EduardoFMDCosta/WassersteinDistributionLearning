import torch
from configs.handlers import load_json
from distributions import Uniform, TruncatedGaussian
from sets import HyperRectangle


def get_support_assumption(support_assumption, **kwargs):
    lower = torch.as_tensor(support_assumption[0])
    upper = torch.as_tensor(support_assumption[1])
    return HyperRectangle(lower, upper)

def get_support(support, **kwargs):
    lower = torch.as_tensor(support[0])
    upper = torch.as_tensor(support[1])
    return HyperRectangle(lower, upper)

def get_distribution(distribution, **kwargs):
    if distribution == 'Uniform':
        support = get_support(**kwargs)
        return Uniform(support=support)
    else:
        ValueError('Unknown distribution.')