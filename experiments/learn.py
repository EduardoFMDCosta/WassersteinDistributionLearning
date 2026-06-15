import torch
import matplotlib.pyplot as plt

from wasserstein_distribution_learning import WassersteinDistributionLearning

from plotting.plot import plot_quantization
from configs.handlers import parse_arguments
from configs.construct import get_support_assumption, get_distribution

if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="TruncatedGaussian",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_samples_training=1000,
        num_clusters=10,
        wasserstein_order=2,
        beta=1e-6,
        method='triangle_inequality_vertex',
        save=False,
        plot=True,
    )

    support = get_support_assumption(**vars(args))
    distribution = get_distribution(**vars(args))

    pretraining_samples = distribution.sample((args.num_samples_training,))
    samples = distribution.sample((args.num_samples,))

    # Convert HyperRectangle → (2, d) tensor expected by the API
    support_tensor = (
        torch.stack([support.lower, support.upper]) if support is not None else None
    )

    wdl = WassersteinDistributionLearning(
        pretraining_samples=pretraining_samples,
        samples=samples,
        beta=args.beta,
        support=support_tensor,
        learning_type=args.learning_type,
        partition_type=args.partition_type,
        num_clusters=args.num_clusters,
        method=args.method,
        wasserstein_order=args.wasserstein_order,
    )

    if args.plot:
        plot_quantization(
            quantization=wdl.ambiguity_set.center,
            samples=samples,
            title=f"M={args.num_clusters}, N={args.num_samples}",
        )
        plt.show()

    print(f"Number of clusters (M) / num_samples (N): {args.num_clusters} / {args.num_samples}")
    print(f"\t Fournier: {wdl.fournier_radius:.4f}")
    print(f"\t Ours (radius): {wdl.ambiguity_set.radius:.4f}")
    if wdl.complement_interval is not None:
        print(f"\t Complement interval: [{wdl.complement_interval.lower:.4f}, {wdl.complement_interval.upper:.4f}]")
    print("Process finished.")

