import torch
import matplotlib.pyplot as plt

from wasserstein_distribution_learning import WassersteinDistributionLearning

from configs.handlers import parse_arguments
from configs.construct import get_support_assumption, get_distribution

if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_samples_training=1000,
        num_clusters=10,
        wasserstein_order=2,
        beta=1e-6,
        method='stochastic_vertice_ascent',
        partition_type='hyperrectangle',
        learning_type='conditional_learning',
        save=False,
        plot=False,
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
        wasserstein_order=args.wasserstein_order,
        pretraining_samples=pretraining_samples,
        samples=samples,
        beta=args.beta,
        support=support_tensor,
        num_clusters=args.num_clusters,
        learning_type=args.learning_type,
        partition_type=args.partition_type,
        method=args.method,
    )


    print(f"Number of clusters (M) / num_samples (N): {args.num_clusters} / {args.num_samples}")
    print(f"\t Fournier: {wdl.fournier_radius:.4f}")
    print(f"\t Ours (radius): {wdl.ambiguity_set.radius:.4f}")
    if wdl.complement_interval is not None:
        print(f"\t Complement interval: [{wdl.complement_interval.lower:.4f}, {wdl.complement_interval.upper:.4f}]")
    print("Process finished.")

