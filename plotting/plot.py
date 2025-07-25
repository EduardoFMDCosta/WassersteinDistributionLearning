import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sets import HyperRectangle
from confidence import Confidence
from plotting.utils_plot import get_bounds_from_confidence_list

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
def plot_confidence(nums_samples:list,
                    empirical: list,
                    hoeff_list: list[Confidence],
                    duchi_list: list[Confidence],
                    pearson_list: list[Confidence],
                    actual_prob: torch.Tensor = None):

    sns.set_style("darkgrid")

    # Get lists
    hoeff_lower, hoeff_upper = get_bounds_from_confidence_list(hoeff_list)
    duchi_lower, duchi_upper = get_bounds_from_confidence_list(duchi_list)
    pearson_lower, pearson_upper = get_bounds_from_confidence_list(pearson_list)

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
def plot_samples(samples: torch.Tensor,
                 regions: HyperRectangle,
                 support_assumption: HyperRectangle):

    if samples.shape[-1] == 2:

        # Plot samples
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0], samples[:, 1], s=0.01, alpha=0.5, color="deepskyblue")

        # Plot shell
        lower = support_assumption.lower.numpy()
        upper = support_assumption.upper.numpy()
        width = upper[0] - lower[0]
        height = upper[1] - lower[1]
        rect = Rectangle(lower, width, height, linewidth=0.5, edgecolor="black", facecolor='none')
        plt.gca().add_patch(rect)

        # Plot partition
        n = regions.lower.shape[0]
        lower = regions.lower.numpy()
        upper = regions.upper.numpy()
        for i in range(n):
            width = upper[i][0] - lower[i][0]
            height = upper[i][1] - lower[i][1]
            rect = Rectangle(lower[i], width, height, linewidth=0.5, edgecolor='black', facecolor='none')
            plt.gca().add_patch(rect)

        plt.axis('equal')
        plt.show()