import os
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from configs.handlers import parse_arguments, load_json, process_args
from experiments.utils import quantizations_for_combinations, load_quantization_samples
from experiments.partitions import get_dict_of_partitions
import plotting.plot as plot
from plotting.utils_plot import set_style

set_style()


def plot_prob(
    ax: plt.Axes,
    locs: torch.Tensor,
    probs: torch.Tensor,
    s_min: float = 0.01,
    s_max: float = 0.05,   
    color: str = "red",   
    offset: float = 0.0,
):
    p0, p1 = probs.min(), probs.max()
    sizes = s_min + (probs - p0) / (p1 - p0 + 1e-12) * (s_max - s_min)

    locs[:, 0] += offset

    ax.quiver(
        locs[:, 0], locs[:, 1], torch.zeros_like(probs), sizes,
        angles='xy', scale_units='xy', scale=1,
        width=0.01, headwidth=2, headlength=2, headaxislength=2, 
        color=color)
    
    # for i in range(locs.size(0)):
    #     arrow = FancyArrowPatch(
    #     locs[i] + torch.tensor([offset, 0.0]), locs[i] + torch.tensor([offset, sizes[i]]),
    #     arrowstyle='->',
    #     linewidth=2,
    #     color=color,
    #     mutation_scale=15
    #     )

    #     ax.add_patch(arrow)

    # ax.scatter(*locs.t(), s=sizes, color=color)
    return ax
    
            

def save(ax, tag: str, save: bool):
    ax.set_aspect('equal', adjustable='box')
    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), f"quantization_process_step{tag}.pdf"))
        plt.close('all')
    else:
        plt.show()

if __name__ == '__main__':
    args = parse_arguments( # Only parse arguments once, updated afterwards
        random_seed=0,
        distribution='Uniform',
        num_dims=2,
        setting=1,
        num_samples=10_000,
        num_clusters=30,
        save=True,
    )

    quantization = quantizations_for_combinations(
        args, 
        combinations=[(args.num_samples_training, args.num_samples, args.num_clusters)], 
        generate_partition_if_missing=False
    ).at((args.num_samples_training, args.num_samples, args.num_clusters))

    samples_partition = get_dict_of_partitions(
        args=args,
        combinations=[(args.num_samples_training, args.num_clusters)],
        return_all_available_combinations=True,
        generate_partition_if_missing=False
    ).samples

    if samples_partition is None:
        raise ValueError("Samples for partition could not be loaded.")
    else:
        samples_partition = samples_partition[:args.num_samples_training]

    samples_quantization = load_quantization_samples(args, args.num_samples, generate_samples_if_missing=False)[:args.num_samples]
    

    # fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(5 * 5, 6 * 1))

    fig0, ax0 = plt.subplots(figsize=(5, 5))
    ax0.scatter(*samples_partition.t(), s=0.05, alpha=0.8, color="mediumseagreen")
    ax0 = plot.plot_support(quantization.support, ax=ax0)
    save(ax0, "0", args.save)    

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.scatter(*samples_partition.t(), s=0.05, alpha=0.8, color="mediumseagreen")
    ax1.scatter(*quantization.locs.t(), s=5, color="black", label=r"$\{c_i\}_{i=1}^M$")
    ax1 = plot.plot_support(quantization.support, ax=ax1)
    save(ax1, "1", args.save)    

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.scatter(*samples_partition.t(), s=0.05, alpha=0.8, color="mediumseagreen")
    ax2 = plot.plot_partition(partition=quantization, ax=ax2)
    save(ax2, "2", args.save)

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3 = plot.plot_quantization(ax=ax3, quantization=quantization, samples=samples_quantization)
    save(ax3, "3", args.save)

    fig4, ax4 = plt.subplots(figsize=(5, 5))
    ax4 = plot.plot_support(quantization.support, ax=ax4)
    for i in range(args.num_clusters):
        circle = Circle(
            quantization.locs[i], 
            radius=quantization.region_l2_radii[i].item(), 
            edgecolor='black', 
            facecolor='none', 
            linestyle='-', 
            alpha=0.5
        )
        ax4.add_patch(circle)

    ax4.scatter(*quantization.locs.t(), s=5, color="black")
    ax4 = plot_prob(ax=ax4, locs=quantization.locs, probs=quantization.lower_probs, color="blue", offset=-0.01)
    ax4 = plot_prob(ax=ax4, locs=quantization.locs, probs=quantization.upper_probs, color="red", offset=0.01)

    ax4.set_ylim(*ax3.get_ylim())
    ax4.set_xlim(*ax3.get_xlim())
    save(ax4, "4", args.save)