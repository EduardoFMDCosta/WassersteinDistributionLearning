import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt

from wasserstein_distribution_learning import EmpiricalPartition, AmbiguitySetLearner
from wasserstein_distribution_learning.sets import HyperRectanglePartition
from plotting.utils_plot import plot_hyperrectangle_partition_2d
from configs.handlers import parse_arguments


class DegenerateGaussian2D:
    def __init__(self, mean_x0: float, std_x0: float, constant_x1: float):
        self.mean_x0 = mean_x0
        self.std_x0 = std_x0
        self.constant_x1 = constant_x1

    def sample(self, sample_shape=torch.Size()):
        x0 = torch.randn(sample_shape) * self.std_x0 + self.mean_x0
        x1 = torch.full(sample_shape, self.constant_x1, dtype=x0.dtype, device=x0.device)
        return torch.stack((x0, x1), dim=-1)


def run_case(
    case_name: str,
    pretraining_samples: torch.Tensor,
    samples: torch.Tensor,
    num_clusters: int,
    beta: float,
    wasserstein_order: int,
    method: str,
    conditional: bool,
):
    partition = EmpiricalPartition(
        pretraining_samples=pretraining_samples,
        num_clusters=num_clusters,
        support=None,
        partition_type="hyperrectangle",
    )

    learner = AmbiguitySetLearner(
        partition=partition,
        samples=samples,
        beta=beta,
        conditional=conditional,
        method=method,
        wasserstein_order=wasserstein_order,
    )

    part = partition.partition
    if not isinstance(part, HyperRectanglePartition):
        raise TypeError("Expected HyperRectanglePartition.")

    widths = part.region_upper - part.region_lower
    width_stats = {
        "mean": widths.mean(dim=0),
        "median": widths.median(dim=0).values,
        "min": widths.min(dim=0).values,
        "max": widths.max(dim=0).values,
    }

    print(f"\n=== {case_name} ===")
    print(f"M={num_clusters}, N={samples.shape[0]}, N_train={pretraining_samples.shape[0]}")
    print(f"Fournier radius: {float(learner.fournier_radius):.6f}")
    print(f"Learned radius:  {float(learner.ambiguity_set.radius):.6f}")
    if learner.complement_interval is not None:
        lo = float(learner.complement_interval.lower)
        up = float(learner.complement_interval.upper)
        print(f"Complement interval: [{lo:.6f}, {up:.6f}]")

    for dim in range(widths.shape[1]):
        print(
            f"width dim {dim}: "
            f"mean={float(width_stats['mean'][dim]):.6f}, "
            f"median={float(width_stats['median'][dim]):.6f}, "
            f"min={float(width_stats['min'][dim]):.6f}, "
            f"max={float(width_stats['max'][dim]):.6f}"
        )

    return {
        "name": case_name,
        "samples": samples,
        "partition": part,
        "learner": learner,
        "widths": widths,
    }


def plot_cases(result_full: dict, result_manifold: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, result in zip(axes, (result_full, result_manifold)):
        samples = result["samples"]
        partition = result["partition"]
        radius = float(result["learner"].ambiguity_set.radius)

        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.22, color="tab:blue", label="samples")
        plot_hyperrectangle_partition_2d(
            region_lower=partition.region_lower,
            region_upper=partition.region_upper,
            ax=ax,
            face_alpha=0.3,
            edge_width=0.8,
        )

        if "Manifold" in result["name"]:
            y0 = float(samples[0, 1])
            ax.axhline(y=y0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_title(f"{result['name']}\nlearned radius={radius:.4f}")
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.grid(alpha=0.25)

    fig.suptitle("Hyperrectangle partition: full 2D Gaussian vs manifold-supported Gaussian")
    fig.tight_layout()

    plt.show()


def build_distributions(mean: torch.Tensor, variance: torch.Tensor, manifold_value: float = 0.0):
    full = torch.distributions.Independent(
        torch.distributions.Normal(loc=mean, scale=variance.sqrt()), 1
    )

    manifold = DegenerateGaussian2D(
        mean_x0=float(mean[0]),
        std_x0=float(variance[0].sqrt()),
        constant_x1=float(manifold_value),
    )
    return full, manifold


def main() -> None:
    args = parse_arguments(
        random_seed=0,
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        num_samples=1000,
        num_samples_training=1000,
        num_clusters=12,
        wasserstein_order=2,
        beta=1e-6,
        method="stochastic_vertice_ascent",
        partition_type="hyperrectangle",
        conditional=True,
        save=False,
        plot=False,
    )

    mean = torch.as_tensor(args.mean).expand(args.num_dims).float()
    variance = torch.as_tensor(args.variance).expand(args.num_dims).float()
    full_dist, manifold_dist = build_distributions(mean=mean, variance=variance, manifold_value=0.0)

    pretrain_full = full_dist.sample((args.num_samples_training,))
    samples_full = full_dist.sample((args.num_samples,))
    pretrain_manifold = manifold_dist.sample((args.num_samples_training,))
    samples_manifold = manifold_dist.sample((args.num_samples,))

    result_full = run_case(
        case_name="Full 2D Gaussian",
        pretraining_samples=pretrain_full,
        samples=samples_full,
        num_clusters=args.num_clusters,
        beta=args.beta,
        wasserstein_order=args.wasserstein_order,
        method=args.method,
        conditional=args.conditional,
    )

    result_manifold = run_case(
        case_name="Manifold Gaussian (x1 constant)",
        pretraining_samples=pretrain_manifold,
        samples=samples_manifold,
        num_clusters=args.num_clusters,
        beta=args.beta,
        wasserstein_order=args.wasserstein_order,
        method=args.method,
        conditional=args.conditional,
    )

    learned_full = float(result_full["learner"].ambiguity_set.radius)
    learned_manifold = float(result_manifold["learner"].ambiguity_set.radius)
    ratio = learned_manifold / learned_full if learned_full > 0 else float("nan")
    print("\n=== Radius comparison ===")
    print(f"manifold/full ratio: {ratio:.6f}")
    print("Fournier radius is expected to be infinite for unbounded support.")

    width_dim1_full = result_full["widths"][:, 1]
    width_dim1_manifold = result_manifold["widths"][:, 1]
    print(
        "second-coordinate widths: "
        f"full median={float(width_dim1_full.median()):.6f}, "
        f"manifold median={float(width_dim1_manifold.median()):.6f}"
    )

    plot_cases(result_full, result_manifold)


if __name__ == "__main__":
    main()
