import torch
from confidence import DuchiConfidence, HoeffdingConfidence
from distributions import Uniform, Gaussian
from plotting import plot_confidence
from sets import HyperRectangle
from utils import empirical_probability

if __name__ == '__main__':
    torch.manual_seed(0)

    # Experiment 1: Figure 5 in Badings et al., 2025 (https://dl.acm.org/doi/pdf/10.1613/jair.1.14253)
    region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))
    support = HyperRectangle(lower=torch.tensor([-4.0]), upper=torch.tensor([4.0]))
    distribution = Uniform(support=support)

    delta = torch.tensor(1e-9)

    badings = []
    nums_samples = torch.logspace(start=torch.log10(torch.tensor(25.0)), end=torch.log10(torch.tensor(2500.0)), steps=50).int().unique()
    for num_samples in nums_samples:
        samples = distribution(num_samples=num_samples)

        alpha = empirical_probability(samples, region)
        confidence = HoeffdingConfidence(delta=delta, alpha=alpha, num_samples=num_samples)

        badings.append(confidence.gamma.item() + alpha.item())

    plot_confidence(nums_samples, badings)

    # Ours
    delta = torch.tensor(1e-9)
    alpha = torch.tensor(0.95)
    n = 1000
    confidence = DuchiConfidence(delta=delta, alpha=alpha, num_samples=n)
    print(confidence.gamma)

    confidence = HoeffdingConfidence(delta=delta, alpha=alpha, num_samples=n)
    print(confidence.gamma)