import torch
from configs.construct import get_support_assumption, get_distribution
from configs.handlers import parse_arguments
from optimization import inner_lp_maximization
from plotting.plot import plot_quantization
from quantization import UncertainQuantization
from sets import BoundedVoronoiPartition


def is_convex_inequality(func, n_samples, tol=1e-8):

    is_convex = True

    for _ in range(n_samples):
        x = torch.randn(cost.shape[0])
        y = torch.randn(cost.shape[0])
        lam = torch.rand(1)
        z = lam * x + (1 - lam) * y

        lhs = func(z)

        if lhs.item() < 0.0:
            rhs = lam * func(x) + (1 - lam) * func(y)

            if lhs - rhs > tol:
                print("Function is NOT convex!")
                print(f"x = {x}")
                print(f"y = {y}")
                print(f"lambda = {lam.item():.4f}")
                print(f"f(lambda*x + (1-lambda)*y) = {lhs.item():.6f}")
                print(f"lambda*f(x) + (1-lambda)*f(y) = {rhs.item():.6f}")
                is_convex = False
            else:
                print("Test passed for this example.")
    return is_convex

if __name__ == '__main__':
    torch.manual_seed(10)

    args = parse_arguments(
        distribution="Gaussian",
        dimension=2,
        setting=0,
        num_samples_training=1000,
        num_samples=1000,
        num_clusters=20,
        beta=1e-4,
        plot=False
    )

    # Set parameters
    N_training = args.num_samples_training
    M = args.num_clusters
    N = args.num_samples
    beta = args.beta
    method = 'max_oracle_gradient_descent'
    support_assumption = get_support_assumption(**vars(args))

    # (Unknown) Generating probability
    distribution = get_distribution(**vars(args))

    # Generate Partitions
    samples_partition = distribution.sample((N_training,))
    partition = BoundedVoronoiPartition(
        support=support_assumption,
        samples=samples_partition,
        M=M,
        use_voronoi_radii=False # set to false to speed up
    )

    # Generate Quantization
    samples_quantization = distribution.sample((N,))
    quantization = UncertainQuantization(partition=partition, samples=samples_quantization, beta=beta)

    # Plot samples and clusterized distribution
    if args.plot:
        plot_quantization(quantization=quantization, title=f"M={M}, N={N}")

    # Get variables
    cost = quantization.partition.distance_locs ** 2
    lower = quantization.lower_probs
    upper = quantization.upper_probs
    empirical_marginal = quantization.probs

    def c_transform(alpha):
        return (cost - alpha.unsqueeze(1)).min(dim=0).values

    def f(alpha):
        beta = c_transform(alpha)
        y, dual_vector = inner_lp_maximization(alpha.clone().detach(), lower, upper)
        lam, mu, nu = y
        value = -lam - (mu * upper).sum() + (nu * lower).sum() - (beta * empirical_marginal).sum()

        return value


    print(is_convex_inequality(f, n_samples=10000, tol=1e-6))