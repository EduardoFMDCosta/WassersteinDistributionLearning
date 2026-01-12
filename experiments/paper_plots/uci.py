import torch
from ucimlrepo import fetch_ucirepo

from quantization import UncertainQuantization
from bound import DataDrivenRadius, fournier_radius
from solvers import get_solver
from sets import BoundedVoronoiPartition

from configs.construct import get_support_assumption
from configs.handlers import parse_arguments


class MinMaxNormalizer:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.registered = False

    def fit(self, X):
        self.min = X.min(dim=0).values
        self.max = X.max(dim=0).values
        self.registered = True
        return self

    def __call__(self, X):
        assert self.registered
        scale = (self.max - self.min).clamp_min(self.eps)
        return (X - self.min) / scale - 0.5

class EmpiricalDistribution:
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform

    def sample(self, n): # Sampling without replacement
        idx = torch.randperm(len(self.X))[:n]
        X = self.X[idx]
        return self.transform(X) if self.transform else X
    
    def __len__(self):
        return len(self.X)

def load_uci_data(args) -> EmpiricalDistribution:
    if args.distribution == "UCI-Turbine":
        df = fetch_ucirepo(id=551).data.features
        df.pop('year')

        features = torch.from_numpy(df.to_numpy()).float()

        transform = MinMaxNormalizer().fit(features)
        dist = EmpiricalDistribution(features, transform=transform)
    elif args.distribution == "UCI-MiniBooNE":
        # data = fetch_ucirepo(id=199)
        raise NotImplementedError
    else:   
        raise ValueError(f"Unknown UCI dataset: {args.distribution}")
    
    return dist

def main(args):
    support_assumption = get_support_assumption(**vars(args))

    dist = load_uci_data(args)
    if len(dist) < args.num_samples + args.num_samples_training:
        raise ValueError(f"Not enough samples in the dataset: {len(dist)} < {args.num_samples + args.num_samples_training}")
    
    samples = dist.sample(args.num_samples + args.num_samples_training)
    partition_samples = samples[:args.num_samples_training]
    quantization_samples = samples[args.num_samples_training:]

    partition = BoundedVoronoiPartition.from_samples(support=support_assumption, samples=partition_samples, M=args.num_clusters)
    
    quantization = UncertainQuantization(partition=partition, samples=quantization_samples, beta=args.beta)

    # Compute bounds
    fournier_bound = fournier_radius(
        support=partition.support, 
        nsamples=args.num_samples + args.num_samples_training,
        wasserstein_order=args.wasserstein_order,
        beta=args.beta
    )
    
    data_driven_output = DataDrivenRadius(
        quantization=quantization, 
        solver= get_solver(method=args.method),
        wasserstein_order=args.wasserstein_order,
    )

    print(f"Number of clusters (M) / num_samples (N): {args.num_clusters} / {args.num_samples} \n"
          f"\t Fournier: {fournier_bound:.4f} \n"
          f"\t Ours : {data_driven_output.radius:.4f} \n"
        )


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="UCI-Turbine",
        num_dims=11,
        setting=0,
        num_samples=1000,
        num_samples_training=1000,
        num_clusters=20,
        wasserstein_order=2,
        beta=1e-6,
        method='triangle_inequality_vertex',
        save=False,
    )
 

    main(args)