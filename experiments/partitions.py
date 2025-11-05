import os
import sys
from typing import Optional, List, Tuple
import torch
import matplotlib.pyplot as plt

from sets import BoundedVoronoiPartition

from plotting.plot import plot_partition
from configs.handlers import parse_arguments, pickle_load, pickle_dump
from configs.construct import get_support_assumption, get_distribution

from experiments.datastructures import _GridDict


class BoundedVoronoiPartitionDict(_GridDict[BoundedVoronoiPartition]): # key = (N, M)
    _samples = None

    def attach_samples(self, samples):
        self._samples = samples
    
    @property
    def samples(self):
        return self._samples

setattr(sys.modules.get('__main__'), 'BoundedVoronoiPartitionDict', BoundedVoronoiPartitionDict)

def get_partition(
    args, 
    num_samples: int, 
    num_clusters: int, 
) -> BoundedVoronoiPartition:
    partitions = get_dict_of_partitions(
        args=args,
        combinations=[(num_samples, num_clusters)]
    )
    return partitions.at((num_samples, num_clusters))


def get_dict_of_partitions(
    args,
    combinations: Optional[List[Tuple[int, int]]] = None,
    num_samples_options: Optional[List[int]] = None, 
    num_clusters_options: Optional[List[int]] = None, 
    return_all_available_combinations: bool = False,
) -> BoundedVoronoiPartitionDict:

    if os.path.exists(args.partitions_file):
        stored_partitions = pickle_load(args.partitions_file)
    else:
        stored_partitions = BoundedVoronoiPartitionDict()

    requested_partitions = BoundedVoronoiPartitionDict()

    if combinations is not None:
        pass
    elif num_samples_options is not None and num_clusters_options is not None:
        combinations = [(N, M) for N in num_samples_options for M in num_clusters_options]
    else:
        raise ValueError("Either combinations or both num_samples_options and num_clusters_options must be provided.")

    # Collect all stored combinations
    missing_combinations = []
    for (N, M) in combinations:
        if (N, M) in stored_partitions.keys():
            requested_partitions.append((N, M), stored_partitions.at((N, M)))
        else:
            missing_combinations.append((N, M))
    
    # Generate missing combinations
    if len(missing_combinations) > 0:
        missing_partitions = generate_partitions(missing_combinations, args, samples=stored_partitions.samples)
        requested_partitions.attach_samples(missing_partitions.samples)

        for (N, M) in missing_combinations:
            requested_partitions.append((N, M), missing_partitions.at((N, M)))

    # Add all other available combinations if requested
    if return_all_available_combinations:
        for (N, M) in stored_partitions.keys():
            if (N, M) not in combinations:
                requested_partitions.append((N, M), stored_partitions.at((N, M)))

    return requested_partitions


def generate_partitions(
    combinations: List[Tuple[int, int]],
    args,
    samples: Optional[torch.Tensor] = None
):
    support_assumption = get_support_assumption(**vars(args))
    distribution = get_distribution(**vars(args))

    max_num_samples = max([N for N, M in combinations])

    if samples is None:
        samples = distribution.sample((max_num_samples,))
    elif samples.size(0) < max_num_samples:
        samples = torch.cat((samples, distribution.sample((max_num_samples - samples.size(0),), )), dim=0)

    partitions = BoundedVoronoiPartitionDict()
    for (N, M) in combinations:
        print(f"Generating partition for N={N}, M={M}")
        assert N <= samples.size(0), "Not enough samples provided to generate partition."
        partitions.append((N, M), BoundedVoronoiPartition.from_samples(
            support=support_assumption,
            samples=samples[:N],
            M=M,
        ))

    partitions.attach_samples(samples)

    return partitions


if __name__ == '__main__':
    torch.manual_seed(0)

    args = parse_arguments(
        distribution="Gaussian",
        num_dims=3,
        setting=0,
        save=True,
        plot=False, 
    )    

    num_samples_options = [1000, 2500] # [1000, 2500, 5000, 7500, 10000]  # [500, 1000, 5000, 10000]
    num_clusters_options = [5, 20] # [5, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]

    partitions = get_dict_of_partitions(
        args=args,
        num_samples_options=num_samples_options,
        num_clusters_options=num_clusters_options,
        return_all_available_combinations=True
    )

    if args.save:
        pickle_dump(partitions, args.partitions_file)
        
    if args.plot:
        for (N, M) in partitions.keys():
            ax = plot_partition(partition=partitions.at((N, M)), title=f"M={M}, N={N}")
            if partitions.samples is not None:
                ax.scatter(*partitions.samples[:N].t(), s=0.05, alpha=1.0, color="deepskyblue", label="Data")
            ax.legend()
            plt.show()


