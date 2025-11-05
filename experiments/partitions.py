import os
import torch
import matplotlib.pyplot as plt

from sets import BoundedVoronoiPartition

from plotting.plot import plot_partition
from configs.handlers import parse_arguments, pickle_load, pickle_dump
from configs.construct import get_support_assumption, get_distribution

from utils import _GridDict


# TODO support GPU

class PartitionBoundedVoronoiPartitionDict(_GridDict[BoundedVoronoiPartition]): # key = (N, M)
    pass

if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=2,
        setting=0,
        save=True,
        plot=False, 
    )

    support_assumption = get_support_assumption(**vars(args))
    distribution = get_distribution(**vars(args))

    num_samples_options = [500, 1000, 5000, 10000]
    num_clusters_options = [5, 20, 30, 40, 50, 75, 100]

    assert torch.as_tensor(num_samples_options).diff().min() > 0, "num_samples_options should be strictly increasing."

    path = os.path.join(args.partitions_dir, f"N={num_samples_options}_M={num_clusters_options}.pickle")

    if os.path.exists(path):
        data = pickle_load(path)
        partitions, samples = data['partitions'], data['samples']
        print(f"Loaded partitions from {path}")
    else:
        partitions = PartitionBoundedVoronoiPartitionDict()

        samples = distribution.sample((num_samples_options[0],))
        for i, N in enumerate(num_samples_options):
            # Guarantee that samples are nested
            if i > 0:
                samples = torch.cat([
                    samples, 
                    distribution.sample((N - num_samples_options[i-1],))
                ], dim=0)

            
            for M in num_clusters_options:
                partitions.append((N, M), BoundedVoronoiPartition(
                    support=support_assumption,
                    samples=samples,
                    M=M,
                ))
                    
        if args.save:
            pickle_dump(dict(partitions=partitions, samples=samples), path)
        

    if args.plot:
        for (N, M) in partitions.keys():
            ax = plot_partition(partition=partitions.at((N, M)), title=f"M={M}, N={N}")
            ax.scatter(*samples.t(), s=0.05, alpha=1.0, color="deepskyblue", label="Data")
            ax.legend()
            plt.show()


