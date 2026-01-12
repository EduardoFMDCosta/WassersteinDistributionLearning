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



import math
import torch

SQRT_2 = math.sqrt(2.0)

def nd_std_match_linf_mass(
    n: torch.Tensor | int,
    support_radius: float,
    ref_std_2d: float,
    *,
    dtype=torch.float64,
    device=None,
    eps: float = 1e-15,
) -> torch.Tensor:
    """
    Compute σ_n such that P(|X_i|<=r for all i=1..n) matches the 2D reference mass
    of a N(0, σ_2d^2 I) over [-r,r]^2, assuming independent coords.

    Vectorized over n.
    """
    n = torch.as_tensor(n, dtype=dtype, device=device)

    r = torch.as_tensor(support_radius, dtype=dtype, device=device)
    sigma2 = torch.as_tensor(ref_std_2d, dtype=dtype, device=device)

    # z = 2*cdf(r/sigma2) - 1 = erf(r/(sqrt(2)*sigma2))
    z = torch.erf(r / (SQRT_2 * sigma2))  # in (0,1)

    # z_pow = z^(2/n) computed stably near 1
    logz = torch.log1p(z - 1.0)           # stable for z≈1
    z_pow = torch.exp((2.0 / n) * logz)

    # Clamp away from exactly 1 to avoid infs
    z_pow = z_pow.clamp(min=0.0 + eps, max=1.0 - eps)

    # Need q = sqrt(2) * erfinv(z_pow)  (since inv_cdf(p) with p=0.5*(1+z_pow))
    # For z_pow close to 1, use erfcinv(1 - z_pow): erfinv(1 - t) = erfcinv(t)
    t = (1.0 - z_pow).clamp(min=eps)  # tiny

    if hasattr(torch.special, "erfcinv"):
        erfinv_z = torch.special.erfcinv(t)
    else:
        # Fallback: less stable in the extreme tail, but float64 + clamp helps a lot.
        erfinv_z = torch.erfinv(z_pow)

    q = SQRT_2 * erfinv_z  # q > 0

    # For N(0, σ_n^2), we have z_pow = erf(r/(sqrt(2)*σ_n))  =>  r/(sqrt(2)*σ_n) = erfinv(z_pow)
    # Hence σ_n = r / (sqrt(2) * erfinv(z_pow)) = r / q
    sigma_n = r / q
    return sigma_n


if __name__ == "__main__":
    # ## --  Gaussian ------------------------------------------------------------------------------------------------- ##
    # support_radius = 0.5

    # ref_2d_std = 0.01 ** 0.5
    # ref_2d_norm = TruncatedMultivariateNormal(
    #     loc = torch.zeros(2),
    #     scale = torch.ones(2) * ref_2d_std,
    #     a = torch.ones(2) * - support_radius,
    #     b = torch.ones(2) * support_radius
    # )

    # print(f"2d target probability mass on support: {ref_2d_norm._base._Z.prod():.8f}")

    # n = 25

    # nd_std = 1 / (2 * inv_cdf(0.5 * (1 + (2 * cdf(0.5 / ref_2d_std) - 1) ** (2 / n))))

    # # nd_norm = TruncatedMultivariateNormal(
    # #     loc = torch.zeros(n),
    # #     scale = torch.ones(n) * nd_std,
    # #     a = torch.ones(n) * - support_radius,
    # #     b = torch.ones(n) * support_radius
    # # )

    # # print(f"nd target probability mass on support: {nd_norm._base._Z.prod():.4f}")
    # print(f"nd variance: {nd_std**2}")

    ## --  Gaussian (numerical stable) ------------------------------------------------------------------------------ ##
    support_radius = 0.5
    ref_2d_std = (0.001) ** 0.5 

    n = 10
    nd_std = nd_std_match_linf_mass(n, support_radius, ref_2d_std, eps=1e-8)
    print(nd_std.item()**2 )


    # ## -- Uniform --------------------------------------------------------------------------------------------------- ##
    # radius_2d = 0.3
    # ndim = 75

    # radius_nd = (radius_2d*2)**(2 / ndim) /2

    # volume_2d = (2 * radius_2d) ** 2
    # volume_nd = (2 * radius_nd) ** ndim

    # print(f"nd radius: {radius_nd}, ratio: {volume_nd / volume_2d:.4f}")