from typing import Union
import torch
import math

from distributions import TruncatedMultivariateNormal

SQRT_2 = math.sqrt(2)


def cdf(x: Union[torch.Tensor, float], mu: Union[torch.Tensor, float] = 0., scale: Union[torch.Tensor, float] = 1.):
    """
    cdf normal distribution
    :param x: input point
    :param mu: mean
    :param scale: standard deviation
    :return:
    """
    return 0.5 * (1 + torch.erf((torch.as_tensor(x) - mu) / (SQRT_2 * scale)))

def inv_cdf(p: Union[torch.Tensor, float], mu: Union[torch.Tensor, float] = 0., scale: Union[torch.Tensor, float] = 1.):
    """
    Inverse CDF (Quantile function) for the normal distribution
    :param p: probability
    :param mu: mean
    :param scale: standard deviation
    :return: corresponding value of the normal distribution
    """
    return mu + scale * torch.erfinv(2 *  torch.as_tensor(p) - 1) * SQRT_2


if __name__ == "__main__":

    # ## --  Gaussian ------------------------------------------------------------------------------------------------- ##
    # support_radius = 0.5

    # ref_2d_std = 0.03 ** 0.5
    # ref_2d_norm = TruncatedMultivariateNormal(
    #     loc = torch.zeros(2),
    #     scale = torch.ones(2) * ref_2d_std,
    #     a = torch.ones(2) * - support_radius,
    #     b = torch.ones(2) * support_radius
    # )

    # print(f"2d target probability mass on support: {ref_2d_norm._base._Z.prod():.4f}")

    # n = 100

    # nd_std = 1 / (2 * inv_cdf(0.5 * (1 + (2 * cdf(0.5 / ref_2d_std) - 1) ** (2 / n))))

    # # nd_norm = TruncatedMultivariateNormal(
    # #     loc = torch.zeros(n),
    # #     scale = torch.ones(n) * nd_std,
    # #     a = torch.ones(n) * - support_radius,
    # #     b = torch.ones(n) * support_radius
    # # )

    # # print(f"nd target probability mass on support: {nd_norm._base._Z.prod():.4f}")
    # print(f"nd variance: {nd_std**2}")

    ## -- Uniform --------------------------------------------------------------------------------------------------- ##
    radius_2d = 0.3
    ndim = 75

    radius_nd = (radius_2d*2)**(2 / ndim) /2

    volume_2d = (2 * radius_2d) ** 2
    volume_nd = (2 * radius_nd) ** ndim

    print(f"nd radius: {radius_nd}, ratio: {volume_nd / volume_2d:.4f}")