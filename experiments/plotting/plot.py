import os
from typing import Optional
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from wasserstein_distribution_learning.quantization import Quantization
from wasserstein_distribution_learning.sets import BoundedVoronoiPartition, HyperRectanglePartition, HyperRectangle
from wasserstein_distribution_learning.confidence import Confidence
import plotting.utils_plot as utils_plot  # noqa: E402 (experiments/ on sys.path)

colors = [
        "lightcoral",
        "olive",
        "mediumseagreen",
        "deepskyblue",
        "orchid"
    ]

plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})





@torch.no_grad()
def plot_confidence(
    nums_samples:list,
    empirical: list,
    hoeff_list: list[Confidence],
    duchi_list: list[Confidence],
    pearson_list: list[Confidence],
    actual_prob: Optional[torch.Tensor] = None,
    save=False
):

    sns.set_style("darkgrid")

    # Get lists
    hoeff_lower, hoeff_upper = utils_plot.get_bounds_from_confidence_list(hoeff_list)
    duchi_lower, duchi_upper = utils_plot.get_bounds_from_confidence_list(duchi_list)
    pearson_lower, pearson_upper = utils_plot.get_bounds_from_confidence_list(pearson_list)

    # Plot
    plt.fill_between(nums_samples, hoeff_lower, hoeff_upper, color="deepskyblue", label = r'Hoeffding', alpha=0.2)
    plt.fill_between(nums_samples, duchi_lower, duchi_upper, color="lightcoral", label=r'Duchi', alpha=0.2)
    plt.fill_between(nums_samples, pearson_lower, pearson_upper, color="olive", label=r'Clopper-Pearson', alpha=0.2)

    plt.plot(nums_samples, empirical, label=r'Empirical probability', linestyle='-', marker='o', color='mediumseagreen')

    if actual_prob is not None:
        plt.axhline(y=actual_prob.item(), color='black', linestyle='--', linewidth=1)

    plt.xlabel("Number of samples")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)
    plt.grid(True)

    #plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    plt.xscale('log')
    plt.xticks(nums_samples, labels=[str(val) for val in nums_samples])
    plt.xlim(nums_samples[0], nums_samples[-1])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin

    if save:
        figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'confidence_bounds')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f"confidence_bounds_comparison.pdf"), format='pdf')

    plt.show()

@torch.no_grad()
def plot_confidence_delta(beta: list, empirical_prob: list, upper_prob: list, save=False):
    sns.set_style("whitegrid")

    fig, ax = plt.subplots()
    sc = ax.scatter(empirical_prob, upper_prob, c=beta, cmap='viridis', s=15)

    # Add colorbar to show beta values
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label(r'$\beta$')

    ax.set_xlabel(r'$\hat\mathbb{P}(R_i)$')
    ax.set_ylabel(r'$p_u(R_\ell) - \hat\mathbb{P}(R_i)$')

    # Format axes to use scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    plt.tight_layout()
    if save:
        figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'confidence_bounds')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f"confidence_bounds_sublinearity.pdf"), format='pdf')

    plt.show()


def plot_support(
    support: HyperRectangle,
    ax: Optional[plt.Axes] = None
):
    if not support.lower.size(-1) == 2:
        raise ValueError("Can only plot 2D quantizations.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    lower = support.lower
    upper = support.upper
    width = upper[0] - lower[0]
    height = upper[1] - lower[1]
    rect = Rectangle(lower, width, height, linewidth=0.5, edgecolor="black", facecolor='none')
    ax.add_patch(rect)
    return ax
    

def plot_partition(
    partition: BoundedVoronoiPartition, 
    ax: Optional[plt.Axes] = None,
    title: str = '',
    plot_locs: bool = True,
):
    if not partition.ndim == 2:
        raise ValueError("Can only plot 2D quantizations.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    ax = plot_support(partition.support, ax=ax)

    # Plot Voronoi cells
    if len(partition) <= 110 and isinstance(partition, BoundedVoronoiPartition):
        ax = utils_plot.plot_clipped_voronoi_2d(
            centers=partition.region_locs,
            max_diameters=partition.region_l2_radii * 2,
            ax=ax,
            face_alpha=0.15
        )

    # Plot HyperRectangle cells
    if isinstance(partition, HyperRectanglePartition):
        ax = utils_plot.plot_hyperrectangle_partition_2d(
            region_lower=partition.region_lower,
            region_upper=partition.region_upper,
            ax=ax,
            face_alpha=0.3,
        )

    if plot_locs:
        ax.scatter(*partition.locs.t(), s=5, color="black", label=r"$\{c_i\}_{i=1}^M$")

    # ax.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    return ax


def plot_quantization(
    quantization: BoundedVoronoiPartition, 
    samples: Optional[torch.Tensor] = None,
    ax: Optional[plt.Axes] = None,
    title: str = ''
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    if samples is not None:
        ax.scatter(*samples.t(), s=0.05, alpha=0.25, color="deepskyblue")
    ax.scatter([], [], s=5, color="deepskyblue", label=r'$\mathcal{D}_N$') # For legend
    ax = plot_partition(partition=quantization, ax=ax, title=title)
    # ax.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    return ax


@torch.no_grad()
def colored_scatter(x, y, c, title, s: int = 200, file_name: Optional[str] = None):
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(x, y, s=s, c=c, cmap='coolwarm', alpha=1.0)
    plt.colorbar(scatter)

    plt.xscale('log')  # Log scale for x-axis
    plt.yscale('log')  # Log scale for y-axis

    plt.xlabel('Number of samples (N)')
    plt.ylabel('Number of clusters (M)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

@torch.no_grad()
def plot_optimization_curves(values, best_values, grad_norms, lr_sizes):

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(values, label=r"$f(\alpha^{(t)})$")
    axs[0].set_ylabel("Value")
    axs[0].legend()
    axs[0].set_ylim(-0.5, 0.0)

    axs[1].plot(best_values, label=r"$f(\alpha^{(best)})$", color="green")
    axs[1].set_ylabel("Best Value")
    axs[1].legend()
    axs[1].set_ylim(-0.5, 0.0)

    axs[2].plot(grad_norms, label=r"$‖\nabla_{\alpha} f(\alpha)‖$", color="red")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Gradient Norm")
    axs[2].legend()
    axs[2].set_ylim(0.0, 0.5)

    axs[3].plot(lr_sizes, label=r"$\eta^{(t)}$", color="blue")
    axs[3].set_xlabel("Iteration")
    axs[3].set_ylabel("Learning rate")
    axs[3].legend()
    axs[3].set_ylim(0.0, 0.5)

    plt.tight_layout()
    plt.show()
