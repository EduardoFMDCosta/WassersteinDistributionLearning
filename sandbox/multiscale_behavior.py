import torch
from torch_kmeans import KMeans
import ot

import matplotlib.pyplot as plt


if __name__ == '__main__':
    torch.manual_seed(0)

    dist = torch.distributions.MultivariateNormal(loc=torch.zeros(2), covariance_matrix=torch.eye(2))
    emp_dist = dist.sample((1000,))
    
    w2_opt, w2_emp = dict(), dict()
    for support_size in range(10, 100, 10):
        # Construct optimal (via k-means) and empirical quantization
        kmeans = KMeans(n_clusters=support_size)
        kmeans_result = kmeans(emp_dist.unsqueeze(0)) # inputs should be at least of shape (BS, N, D)

        opt_locs = kmeans_result.centers.squeeze(0)
        opt_probs = torch.bincount(kmeans_result.labels.squeeze(0), minlength=support_size) / emp_dist.size(0)

        emp_locs = dist.sample((support_size,))

        # Compute Wasserstein errors (alternatively use ot.sinkhorn2 to speed up)
        w2_opt[support_size] = ot.solve_sample(X_a=emp_dist, X_b=opt_locs, b=opt_probs).value.sqrt().item()
        w2_emp[support_size] = ot.solve_sample(X_a=emp_dist, X_b=emp_locs).value.sqrt().item()


    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(list(w2_opt.keys()), list(w2_opt.values()), label='Optimal quantization')
    ax.plot(list(w2_emp.keys()), list(w2_emp.values()), label='Empirical quantization')
    ax.set_xlabel('Support size')
    ax.set_ylabel('Wasserstein distance')
    ax.legend()
    plt.show()

        





