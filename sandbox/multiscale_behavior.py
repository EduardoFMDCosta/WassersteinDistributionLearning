import os
import torch
from torch_kmeans import KMeans
import ot

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def compute_w2(dist, n_samples=10000):
    emp_dist = dist.sample((n_samples,))

    w2_opt, w2_emp = dict(), dict()
    for support_size in [20, 40, 60, 80, 100, 200, 500, 1000, 2000]:
        # Construct optimal (via k-means) and empirical quantization
        kmeans = KMeans(n_clusters=support_size)
        kmeans_result = kmeans(emp_dist.unsqueeze(0))  # inputs should be at least of shape (BS, N, D)

        opt_locs = kmeans_result.centers.squeeze(0)
        opt_probs = torch.bincount(kmeans_result.labels.squeeze(0), minlength=support_size) / emp_dist.size(0)

        emp_locs = dist.sample((support_size,))

        # Compute Wasserstein errors (alternatively use ot.sinkhorn2 to speed up)
        w2_opt[support_size] = ot.solve_sample(X_a=emp_dist, X_b=opt_locs, b=opt_probs).value.sqrt().item()
        w2_emp[support_size] = ot.solve_sample(X_a=emp_dist, X_b=emp_locs).value.sqrt().item()

    return w2_opt, w2_emp

if __name__ == '__main__':
    torch.manual_seed(0)

    save = True

    # Define distributions
    dist1 = torch.distributions.MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2))
    dist2 = torch.distributions.MultivariateNormal(loc=torch.zeros(2), covariance_matrix=0.2 * torch.eye(2))

    # Compute Wasserstein distances
    w2_opt_1, w2_emp_1 = compute_w2(dist1)
    w2_opt_2, w2_emp_2 = compute_w2(dist2)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].plot(list(w2_opt_1.keys()), list(w2_opt_1.values()), label='Our proposed quantization')
    axes[0].plot(list(w2_emp_1.keys()), list(w2_emp_1.values()), label='Empirical distribution')
    axes[0].set_xlabel('Support size')
    axes[0].set_ylabel(r'$2$-Wasserstein distance')
    axes[0].legend()

    axes[1].plot(list(w2_opt_2.keys()), list(w2_opt_2.values()), label='Our proposed quantization')
    axes[1].plot(list(w2_emp_2.keys()), list(w2_emp_2.values()), label='Empirical distribution')
    axes[1].set_xlabel('Support size')
    axes[1].legend()

    plt.tight_layout()
    if save:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'multiscale')
        plt.savefig(os.path.join(results_dir, f"multiscale_behavior.pdf"), format='pdf')

    plt.show()

        





