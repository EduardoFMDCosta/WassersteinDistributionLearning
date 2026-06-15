import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from wasserstein_distribution_learning import EmpiricalPartition, AmbiguitySetLearner

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

    partition = EmpiricalPartition(
        pretraining_samples=pretraining_samples,
        num_clusters=args.num_clusters,
        support=support,
        partition_type=args.partition_type,
    )

    learner = AmbiguitySetLearner(
        partition=partition,
        samples=samples,
        beta=args.beta,
        learning_type=args.learning_type,
        method=args.method,
        wasserstein_order=args.wasserstein_order,
    )

    print(f"Number of clusters (M) / num_samples (N): {args.num_clusters} / {args.num_samples}")
    print(f"\t Fournier: {learner.fournier_radius:.4f}")
    print(f"\t Ours (radius): {learner.ambiguity_set.radius:.4f}")
    if learner.complement_interval is not None:
        print(f"\t Complement interval: [{learner.complement_interval.lower:.4f}, {learner.complement_interval.upper:.4f}]")
    print("Process finished.")

