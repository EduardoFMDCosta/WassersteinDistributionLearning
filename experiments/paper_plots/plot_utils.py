import matplotlib.pyplot as plt

def set_style():
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

def save_fig(fig, filename, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")

def convert_to_sci_notation(N):
    s = f"{N:.0e}"
    base, exp = s.split("e")
    exp = int(exp)
    return fr"{base} \times 10^{{{exp}}}"

