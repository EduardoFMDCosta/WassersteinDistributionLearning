import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

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
                             hoeff_lower: list,
                             hoeff_upper: list,
                             duchi_lower: list,
                             duchi_upper: list,
                             pearson_lower: list,
                             pearson_upper: list):

    sns.set_style("darkgrid")

    plt.fill_between(nums_samples, hoeff_lower, hoeff_upper, color="deepskyblue", label = r'Hoeffding', alpha=0.2)

    plt.fill_between(nums_samples, duchi_lower, duchi_upper, color="lightcoral", label=r'Duchi', alpha=0.2)

    plt.fill_between(nums_samples, pearson_lower, pearson_upper, color="olive", label=r'Clopper-Pearson', alpha=0.2)

    plt.plot(nums_samples, empirical, label=r'Empirical probability', linestyle='-', marker='o', color='mediumseagreen')

    plt.xlabel("Number of samples")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)
    plt.grid(True)

    #plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin
    plt.show()