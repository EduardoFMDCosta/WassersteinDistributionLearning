import torch
from confidence import DuchiConfidence, HoeffdingConfidence, ClopperPearsonConfidence
from distributions import Uniform, Gaussian
from plotting import plot_confidence
from sets import HyperRectangle
from utils import in_set

if __name__ == '__main__':
    torch.manual_seed(0)

    # Store structures
    hoeff_lower, hoeff_upper = [], []
    duchi_lower, duchi_upper = [], []
    pearson_lower, pearson_upper = [], []
    empirical = []

    # Parameters
    beta = 1e-5
    nums_samples = [10, 50, 100, 500, 1000, 5000]

    # Experiment 1: Figure 5 in Badings et al., 2025 (https://dl.acm.org/doi/pdf/10.1613/jair.1.14253)
    region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))
    support = HyperRectangle(lower=torch.tensor([-4.0]), upper=torch.tensor([4.0]))
    distribution = Uniform(support=support)

    for num_samples in nums_samples:
        samples = distribution(num_samples=num_samples)
        n_set = in_set(samples=samples, regions=region)

        empirical.append(n_set / num_samples)

        hoeffding_confidence = HoeffdingConfidence(beta=beta, n_set=n_set, n=num_samples)
        duchi_confidence = DuchiConfidence(beta=beta, n_set=n_set, n=num_samples)
        pearson_confidence = ClopperPearsonConfidence(beta=beta, n_set=n_set, n=num_samples)

        # Store
        hoeff_lower.append(hoeffding_confidence.lower_proba)
        hoeff_upper.append(hoeffding_confidence.upper_proba)

        duchi_lower.append(duchi_confidence.lower_proba)
        duchi_upper.append(duchi_confidence.upper_proba)

        pearson_lower.append(pearson_confidence.lower_proba)
        pearson_upper.append(pearson_confidence.upper_proba)

    plot_confidence(nums_samples, empirical, hoeff_lower, hoeff_upper, duchi_lower, duchi_upper, pearson_lower, pearson_upper)