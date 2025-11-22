import torch
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import data_driven_radii_for_combinations, fournier_radii_for_combinations

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 200,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

def sci_label(N):
    s = f"{N:.0e}"
    base, exp = s.split("e")
    exp = int(exp)
    return rf"${base} \times 10^{{{exp}}}$"

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=10,
        setting=0,
        num_samples_training=1000,
        beta=1e-6,
        method='triangle_inequality_vertex',
        plot=True,
        save=False,
        compute_moment_bound=True,
        compute_discrete_bound=True,
    )

    N_options = [1000, 5000, 10000, 100000, 1000000]
    M_options = [5, 10, 20, 30, 40, 50, 100]

    combinations = [(N, M) for N in N_options for M in M_options]

    (quantizations, data_driven_radii), _ = data_driven_radii_for_combinations(args, combinations=combinations,
                                                                               generate_partition_if_missing=True)
    fournier_radii = fournier_radii_for_combinations(args, combinations)

    # Plot inflection curves
    fig, ax = plt.subplots(figsize=(6, 4))

    for N in N_options:
        y_vals = []
        for M in M_options:
            key = (args.num_samples_training, N, M)
            radius = data_driven_radii.data[key].radius
            fournier_radius = fournier_radii.data[key]
            y_vals.append(radius / fournier_radius)

        ax.plot(M_options, y_vals, marker="o", label=sci_label(N))

    ax.set_xlabel(r"$M$")
    ax.set_ylabel("Our bound / Fournier")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=r"$N$", loc="best")

    plt.tight_layout()
    plt.show()