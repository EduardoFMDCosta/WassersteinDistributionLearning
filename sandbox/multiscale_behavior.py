import os
import torch
from torch_kmeans import KMeans
import ot

import matplotlib.pyplot as plt
from configs.handlers import ensure_dir

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def compute_w1(dist, n_samples=5000):
    reference_dist = dist.sample((n_samples,))
    reference_probs = torch.ones(n_samples) / n_samples

    w1_opt, w1_emp = dict(), dict()
    for support_size in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150, 200, 300, 500, 1000]:

        # Construct optimal (via k-means) quantization
        kmeans = KMeans(n_clusters=support_size)
        kmeans_result = kmeans(reference_dist.unsqueeze(0))  # inputs should be at least of shape (BS, N, D)
        opt_locs = kmeans_result.centers.squeeze(0)
        opt_probs = torch.bincount(kmeans_result.labels.squeeze(0), minlength=support_size) / reference_dist.size(0)

        # Compute W1 distance for quantization
        M_opt = ot.dist(reference_dist, opt_locs, metric='euclidean')
        w1_opt[support_size] = ot.emd2(reference_probs, opt_probs, M_opt)

        # Compute avg W1 distance for empirical distributions
        w1_emp_list = []
        for _ in range(1000):
            # Empirical distribution with N samples
            emp_locs = dist.sample((support_size,))
            emp_probs = torch.ones(support_size) / support_size

            M_emp = ot.dist(reference_dist, emp_locs, metric='euclidean')
            w1_emp_list.append(ot.emd2(reference_probs, emp_probs, M_emp))
        w1_emp[support_size] = sum(w1_emp_list) / len(w1_emp_list)

    return w1_opt, w1_emp

if __name__ == '__main__':
    torch.manual_seed(0)

    # Parameters
    dimension = 4
    save = True

    # Parameters of GMMs
    mix_weights = torch.tensor([0.5, 0.5])

    locs = torch.stack([
        torch.zeros(dimension),
        torch.ones(dimension)
    ])

    covs = torch.stack([
        0.5 * torch.eye(dimension),
        0.5 * torch.eye(dimension)
    ])

    # Construct distributions
    dist1 = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(mix_weights),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=locs,
            covariance_matrix=covs
        )
    )

    dist2 = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(mix_weights),
        component_distribution=torch.distributions.MultivariateNormal(
            loc=locs,
            covariance_matrix=0.05 * covs
        )
    )

    # Compute Wasserstein distances
    w1_opt_1, avg_w1_emp_1 = compute_w1(dist1)
    w1_opt_2, avg_w1_emp_2 = compute_w1(dist2)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left subplot ---
    Ns1 = list(w1_opt_1.keys())
    vals_opt_1 = list(w1_opt_1.values())
    vals_emp_1 = list(avg_w1_emp_1.values())

    # Plot main curves and capture colors
    line_opt_1, = axes[0].plot(Ns1, vals_opt_1, label='Our proposed quantization')
    line_emp_1, = axes[0].plot(Ns1, vals_emp_1, label='Empirical distribution')

    # Add N^{-1/dimension} rate lines (dashed)
    Ns_tensor = torch.tensor(Ns1, dtype=torch.float32)
    rate_opt_1 = vals_opt_1[0] * (Ns_tensor / Ns_tensor[0]).pow(-1 / dimension)
    rate_emp_1 = vals_emp_1[0] * (Ns_tensor / Ns_tensor[0]).pow(-1 / dimension)

    axes[0].plot(Ns1, rate_opt_1.tolist(), '--', color=line_opt_1.get_color())
    axes[0].plot(Ns1, rate_emp_1.tolist(), '--', color=line_emp_1.get_color())

    axes[0].set_xlabel('Support size')
    axes[0].set_ylabel(r'$1$-Wasserstein distance')

    rate_handle, = axes[0].plot([], [], '--', color='gray', label=rf'$N^{{-1/{dimension}}}$ rate')
    axes[0].legend(handles=[line_opt_1, line_emp_1, rate_handle])

    # --- Right subplot ---
    Ns2 = list(w1_opt_2.keys())
    vals_opt_2 = list(w1_opt_2.values())
    vals_emp_2 = list(avg_w1_emp_2.values())

    line_opt_2, = axes[1].plot(Ns2, vals_opt_2, label='Our proposed quantization')
    line_emp_2, = axes[1].plot(Ns2, vals_emp_2, label='Empirical distribution')

    # Add N^{-1/dimension} rate lines (dashed)
    Ns_tensor = torch.tensor(Ns2, dtype=torch.float32)
    rate_opt_2 = vals_opt_2[0] * (Ns_tensor / Ns_tensor[0]).pow(-1 / dimension)
    rate_emp_2 = vals_emp_2[0] * (Ns_tensor / Ns_tensor[0]).pow(-1 / dimension)

    axes[1].plot(Ns2, rate_opt_2.tolist(), '--', color=line_opt_2.get_color())
    axes[1].plot(Ns2, rate_emp_2.tolist(), '--', color=line_emp_2.get_color())

    axes[1].set_xlabel('Support size')

    rate_handle, = axes[1].plot([], [], '--', color='gray', label=rf'$N^{{-1/{dimension}}}$ rate')
    axes[1].legend(handles=[line_opt_2, line_emp_2, rate_handle])

    plt.tight_layout()
    if save:
        figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'multiscale')
        ensure_dir(figures_dir)
        plt.savefig(os.path.join(figures_dir, f"multiscale_behavior.pdf"), format='pdf')

    plt.show()

        





