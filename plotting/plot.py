from typing import Optional
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from quantization import Quantization
from sets import BoundedVoronoiPartition
from confidence import Confidence
import plotting.utils_plot as utils_plot
from experiments.utils import DataDrivenRadii, Quantizations

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
def plot_w2_slice(
    ax, 
    data_driven_radii: DataDrivenRadii, 
    N: Optional[int] = None, 
    M: Optional[int] = None
):
    if (N is not None and M is not None) or (N is None and M is None):
        raise ValueError("Only N or M should be specified.")

    data_sliced = data_driven_radii._slice(N=N, M=M)
    idx = 1 if N is not None else 0
    options = [key[idx] for key in data_driven_radii.keys()]

    ax.plot(options, data_sliced.radius, label='w2', marker='o')
    ax.plot(options, data_sliced.moment_bound, label='e1', linestyle='--')
    ax.plot(options, data_sliced.discrete_bound, label='e2', linestyle=':')
    ax.plot(options, data_sliced.lower_bound, label='lower_bound', linestyle='--')
    # ax.set_xscale('log')
    ax.set_xlim(min(options), max(options))
    ax.legend(loc='best')
    return ax

@torch.no_grad()
def plot_quantization_slice(
    ax, 
    quantizations: Quantizations, 
    stat: str,
    N: Optional[int] = None, 
    M: Optional[int] = None
):
    if (N is not None and M is not None) or (N is None and M is None):
        raise ValueError("Only N or M should be specified.")
    
    data_sliced = quantizations._slice(N=N, M=M)
    idx = 1 if N is not None else 0
    options = [key[idx] for key in quantizations.keys()]

    if stat == 'probs':
        ax.plot(options, data_sliced.mean_range_probs, label='avg probs range', color='black')
        ax.fill_between(
            options,
            data_sliced.mean_range_probs - data_sliced.std_range_probs,
            data_sliced.mean_range_probs + data_sliced.std_range_probs,
            color='black',
            alpha=0.2,
            label='std dev'
        )
    elif stat == 'radii':
        ax.plot(options, data_sliced.mean_cluster_radii, label='avg cluster radii', color='black')
        ax.fill_between(
            options,
            data_sliced.mean_cluster_radii - data_sliced.std_cluster_radii,
            data_sliced.mean_cluster_radii + data_sliced.std_cluster_radii,
            color='black',
            alpha=0.2,
            label='std dev'
        )
    elif stat == 'counts':    
        ax.plot(options, data_sliced.outer_counts, label='outer counts', marker='o', color='red')
        ax.plot(options, data_sliced.mean_cluster_counts, label='avg cluster counts', marker='o', color='blue')
        ax.fill_between(
            options,
            data_sliced.mean_cluster_counts - data_sliced.std_cluster_counts,
            data_sliced.mean_cluster_counts + data_sliced.std_cluster_counts,
            color='blue',
            alpha=0.2,
            label='std dev'
        )
    elif stat == 'locs':
        ax.plot(options, data_sliced.mean_distances_locs, label='avg distances', color='black')
        ax.fill_between(
            options,
            data_sliced.mean_distances_locs - data_sliced.std_distances_locs,
            data_sliced.mean_distances_locs + data_sliced.std_distances_locs,
            color='black',
            alpha=0.2,
            label='std dev'
        )
    else:
        raise ValueError(f"Stat {stat} not recognized. Choose from 'probs', 'radii', 'counts', 'locs'.")
    
    # ax.set_xscale('log')
    ax.set_xlim(min(options), max(options))
    ax.legend(loc='best')
    return ax


@torch.no_grad()
def plot_confidence(nums_samples:list,
                    empirical: list,
                    hoeff_list: list[Confidence],
                    duchi_list: list[Confidence],
                    pearson_list: list[Confidence],
                    actual_prob: Optional[torch.Tensor] = None):

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
    plt.show()

@torch.no_grad()
def plot_confidence_delta(beta: list, empirical_prob: list, upper_prob: list):
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
    plt.show()

def plot_quantization(
    quantization: Quantization, 
    ax: Optional[plt.Axes] = None,
    title: str = ''
):
    if not quantization.ndim == 2:
        raise ValueError("Can only plot 2D quantizations.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*quantization.samples.t(), s=0.05, alpha=1.0, color="deepskyblue", label="Data")

    # Plot shell
    lower = quantization.partition.support.lower
    upper = quantization.partition.support.upper
    width = upper[0] - lower[0]
    height = upper[1] - lower[1]
    rect = Rectangle(lower, width, height, linewidth=0.5, edgecolor="black", facecolor='none')
    ax.add_patch(rect)

    # Plot Voronoi cells
    if len(quantization) <= 110 and isinstance(quantization.partition, BoundedVoronoiPartition):
        ax = utils_plot.plot_clipped_voronoi_2d(
            centers=quantization.partition.cluster_centers,
            max_diameters=quantization.partition.cluster_radii * 2,
            ax=ax
        )

    # Plot locs
    ax.scatter(*quantization.locs.t(), s=5, color="red", label="Cluster Centers")

    ax.legend()
    ax.axis('equal')
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