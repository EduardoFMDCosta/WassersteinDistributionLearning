"""Visualise the two partition types (Voronoi and HyperRectangle) side by side
for a 2-D distribution.

Run from the project root:
    py experiments/partition.py
"""

import torch
import matplotlib.pyplot as plt

from wasserstein_distribution_learning import BoundedVoronoiPartition, HyperRectanglePartition

from plotting.plot import plot_partition
from configs.handlers import parse_arguments
from configs.construct import get_support_assumption, get_distribution

if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution='GaussianMixture',
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_samples_training=5000,
        num_clusters=20,
        plot=False,
        save=False,
    )

    assert args.num_dims == 2, "partition.py only supports 2-D distributions."

    support = get_support_assumption(**vars(args))
    distribution = get_distribution(**vars(args))
    samples = distribution.sample((args.num_samples_training,))

    voronoi_partition = BoundedVoronoiPartition.from_samples(
        support=support,
        samples=samples,
        M=args.num_clusters,
    )
    hyperrect_partition = HyperRectanglePartition.from_samples(
        support=support,
        samples=samples,
        M=args.num_clusters,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax_v = plot_partition(
        partition=voronoi_partition,
        ax=axes[0],
        title=f'Voronoi  (M={args.num_clusters})',
    )
    ax_v.scatter(*samples.t(), s=2, alpha=0.25, color='deepskyblue', zorder=0)

    ax_h = plot_partition(
        partition=hyperrect_partition,
        ax=axes[1],
        title=f'HyperRectangle  (M={args.num_clusters})',
    )
    ax_h.scatter(*samples.t(), s=2, alpha=0.25, color='deepskyblue', zorder=0)

    # Share axis limits across both subplots so the comparison is fair
    all_axes = [ax_v, ax_h]
    xlim = (min(ax.get_xlim()[0] for ax in all_axes),
            max(ax.get_xlim()[1] for ax in all_axes))
    ylim = (min(ax.get_ylim()[0] for ax in all_axes),
            max(ax.get_ylim()[1] for ax in all_axes))
    for ax in all_axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()
