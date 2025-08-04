import torch
from utils import in_set
from sets import HyperRectangle
from distributions import MultivariateUniform
from plotting.plot import plot_confidence
from confidence import DuchiConfidence, HoeffdingConfidence, ClopperPearsonConfidence

if __name__ == '__main__':
    torch.manual_seed(0)

    # Store structures
    hoeff, duchi, pearson = [], [], []
    empirical = []

    # Parameters
    beta = 1e-4
    nums_samples = [10, 50, 100, 500, 1000, 5000, 10000]

    # Replicate experiment from Figure 5 in Badings et al., 2025 (https://dl.acm.org/doi/pdf/10.1613/jair.1.14253)
    support = HyperRectangle(lower=torch.tensor([-4.0]), upper=torch.tensor([4.0]))
    distribution = MultivariateUniform(low=support.lower, high=support.upper)

    region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))

    actual_region_prob = region.width / support.width

    for num_samples in nums_samples:
        samples = distribution.sample((num_samples,))
        n_set = in_set(samples=samples, regions=region)

        empirical.append(n_set / num_samples)

        hoeffding_confidence = HoeffdingConfidence(beta=beta, n_set=n_set, n=num_samples)
        duchi_confidence = DuchiConfidence(beta=beta, n_set=n_set, n=num_samples)
        pearson_confidence = ClopperPearsonConfidence(beta=beta, n_set=n_set, n=num_samples)

        # Store
        hoeff.append(hoeffding_confidence)
        duchi.append(duchi_confidence)
        pearson.append(pearson_confidence)

    plot_confidence(nums_samples, empirical, hoeff, duchi, pearson, actual_region_prob)