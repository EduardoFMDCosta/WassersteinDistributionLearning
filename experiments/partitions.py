import os
import sys
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Generic, TypeVar, Union
import torch
import matplotlib.pyplot as plt

from sets import BoundedVoronoiPartition

from plotting.plot import plot_partition
from configs.handlers import parse_arguments, pickle_dump, pickle_load
from configs.construct import get_support_assumption, get_distribution


S = TypeVar("S", bound="BoundedVoronoiPartitionDict")

@dataclass
class BoundedVoronoiPartitionDict: # key = (N_train, M)
    _samples: Optional[torch.Tensor] = None
    data: Dict[Tuple[int, int], BoundedVoronoiPartition] = field(default_factory=dict)
    
    def append(self, key: Tuple[int, int], rec: BoundedVoronoiPartition) -> None:
        self.data[key] = rec

    def at(self, key: Tuple[int, int]) -> BoundedVoronoiPartition:
        return self.data[key]
    
    def keys(self, N_train: Optional[int] = None, M: Optional[int]= None) -> List[Tuple[int, int]]:
        return [key for key in self.data.keys() if (N_train is None or key[0] == N_train) and (M is None or key[1] == M)]

    def attach_samples(self, samples):
        self._samples = samples

    @property
    def samples(self):
        return self._samples
    
setattr(sys.modules.get('__main__'), 'BoundedVoronoiPartitionDict', BoundedVoronoiPartitionDict)
    
@dataclass
class TimeLoggerPartition: # key = (N_train, M)
    data: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)
    
    def append(self, key: Tuple[int, int], rec: torch.Tensor) -> None:
        self.data[key] = rec

    def at(self, key: Tuple[int, int]) -> torch.Tensor:
        return self.data[key]
    
    def keys(self, N_train: Optional[int] = None, M: Optional[int]= None) -> List[Tuple[int, int]]:
        return [key for key in self.data.keys() if (N_train is None or key[0] == N_train) and (M is None or key[1] == M)]

setattr(sys.modules.get('__main__'), 'TimeLoggerPartition', TimeLoggerPartition)

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

def get_time_logger_partition(
    args
):
    if os.path.exists(args.partitions_timing_file):
        return pickle_load(args.partitions_timing_file)
    else:
        return TimeLoggerPartition()


def get_dict_of_partitions(
    args,
    combinations: Optional[List[Tuple[int, int]]] = None,
    num_samples_options: Optional[List[int]] = None, 
    num_clusters_options: Optional[List[int]] = None, 
    return_all_available_combinations: bool = False,
    generate_partition_if_missing: bool = True,
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
    if generate_partition_if_missing and len(missing_combinations) > 0:
        missing_partitions = generate_partitions(missing_combinations, args, samples=stored_partitions.samples)
        requested_partitions.attach_samples(missing_partitions.samples)

        for (N, M) in missing_combinations:
            requested_partitions.append((N, M), missing_partitions.at((N, M)))
    else:
        requested_partitions.attach_samples(stored_partitions.samples)
        print(f"Warning: {len(missing_combinations)} requested partitions are missing and will not be generated.")

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
    time_logger = get_time_logger_partition(args)

    max_num_samples = max([N for N, M in combinations])

    if samples is None:
        samples = distribution.sample((max_num_samples,))
    elif samples.size(0) < max_num_samples:
        samples = torch.cat((samples, distribution.sample((max_num_samples - samples.size(0),), )), dim=0)

    partitions = BoundedVoronoiPartitionDict()
    for (N, M) in combinations:
        print(f"Generating partition for N={N}, M={M}")
        start = time.time()
        assert N <= samples.size(0), "Not enough samples provided to generate partition."
        partitions.append((N, M), BoundedVoronoiPartition.from_samples(
            support=support_assumption,
            samples=samples[:N],
            M=M,
        ))
        time_logger.append((N, M), torch.tensor(time.time() - start))

    partitions.attach_samples(samples)

    if args.save:
        pickle_dump(time_logger, args.partitions_timing_file)

    return partitions


if __name__ == '__main__':
    args = parse_arguments(
        random_seed=0,
        distribution="Gaussian",
        num_dims=10,
        setting=1,
        save=True,
        plot=False, 
    )    

    num_samples_options = [5000] # , 7500, 10000, 25000
    num_clusters_options = [5, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]


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


