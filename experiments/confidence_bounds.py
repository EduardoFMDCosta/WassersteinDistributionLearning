import torch

from sets import HyperRectangle
from distributions import MultivariateUniform
from confidence import DuchiConfidence, HoeffdingConfidence, ClopperPearsonConfidence

from plotting.plot import plot_confidence, plot_confidence_delta


if __name__ == '__main__':
    torch.manual_seed(0)

    ### Experiment 1: Compare Hoeffding, Duchi and Clopper-Pearson probability bounds
    # Store structures
    hoeff, duchi, pearson = [], [], []
    empirical = []

    # Parameters
    beta = 1e-4
    nums_samples = [10, 50, 100, 1000, 5000, 10000]

    # Replicate experiment from Figure 5 in Badings et al., 2025 (https://dl.acm.org/doi/pdf/10.1613/jair.1.14253)
    support = HyperRectangle(lower=torch.tensor([-4.0]), upper=torch.tensor([4.0]))
    distribution = MultivariateUniform(low=support.lower, high=support.upper)

    region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))

    actual_region_prob = region.width / support.width

    for num_samples in nums_samples:
        samples = distribution.sample((num_samples,))
        n_set = region.included(samples).sum()

        empirical.append(n_set / num_samples)

        hoeffding_confidence = HoeffdingConfidence(beta=beta, n_set=n_set, n=num_samples)
        duchi_confidence = DuchiConfidence(beta=beta, n_set=n_set, n=num_samples)
        pearson_confidence = ClopperPearsonConfidence(beta=beta, n_set=n_set, n=num_samples)

        # Store
        hoeff.append(hoeffding_confidence)
        duchi.append(duchi_confidence)
        pearson.append(pearson_confidence)

    plot_confidence(nums_samples, empirical, hoeff, duchi, pearson, actual_region_prob, save=True)


    ### Experiment 2: Compare Clopper-Pearson's upperbound for diminishing confidence
    # Store structures
    empiricals, adjusted_betas, upper_deltas, upper_probs = [], [], [], []

    # Parameters
    num_samples = 10000
    beta = 1e-8
    nums_clusters = list(range(3, num_samples + 1))

    for num_clusters in nums_clusters:
        adjusted_beta = beta / num_clusters

        n_set = torch.tensor(num_samples / num_clusters)
        empirical = n_set / num_samples
        empiricals.append(empirical)

        pearson_confidence = ClopperPearsonConfidence(beta=adjusted_beta, n_set=n_set, n=num_samples)

        # Store
        adjusted_betas.append(adjusted_beta)
        upper_deltas.append(pearson_confidence.upper_proba - empirical)
        upper_probs.append(pearson_confidence.upper_proba.item())

    plot_confidence_delta(adjusted_betas, empiricals, upper_deltas, save=True)