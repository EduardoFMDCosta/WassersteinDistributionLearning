import os
import matplotlib.pyplot as plt

from configs.handlers import parse_arguments
from experiments.utils import quantizations_for_combinations, load_samples
import plotting.plot as plot
from plotting.utils_plot import set_style

set_style()

if __name__ == '__main__':
    args = parse_arguments( # Only parse arguments once, updated afterwards
        random_seed=0,
        distribution='TruncatedGaussianMixture',
        num_dims=2,
        setting=0,
        num_samples=10_000,
        num_clusters=75,
        save=True,
    )

    quantization = quantizations_for_combinations(
        args, 
        combinations=[(args.num_samples_training, args.num_samples, args.num_clusters)], 
        generate_partition_if_missing=False
    ).at((args.num_samples_training, args.num_samples, args.num_clusters))

    
    fig, ax = plt.subplots(figsize=(5., 5,))

    samples = load_samples(args, args.num_samples, generate_samples_if_missing=False)

    ax = plot.plot_quantization(ax=ax, quantization=quantization, samples=samples)

    if args.save:
        fig.tight_layout()
        tag = f"{args.distribution.lower()}_dims={args.num_dims}_setting={args.setting}_N={args.num_samples}_M={args.num_clusters}"
        plt.savefig(os.path.join(args.figures_dir, f"quantization_{tag}.pdf"))
        plt.close('all')
    else:
        plt.show()