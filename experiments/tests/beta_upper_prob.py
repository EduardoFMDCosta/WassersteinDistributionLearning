import torch
from plotting.plot import plot_confidence_delta
from confidence import ClopperPearsonConfidence

if __name__ == '__main__':
    torch.manual_seed(0)

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

    plot_confidence_delta(adjusted_betas, empiricals, upper_deltas)
    print(f"Upper probs: {upper_probs}")