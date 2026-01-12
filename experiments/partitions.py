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
from configs.handlers import parse_arguments, pickle_dump, pickle_load, process_args
from configs.construct import get_support_assumption, get_distribution


S = TypeVar("S", bound="BoundedVoronoiPartitionDict")


def load_samples(args, N: int, generate_samples_if_missing: bool = True, to_construct_quantization: bool = True) -> torch.Tensor:
    file_name = args.quantization_samples_file if to_construct_quantization else args.partition_samples_file

    if os.path.exists(file_name):
        stored_samples = pickle_load(file_name)
    elif generate_samples_if_missing:
        stored_samples = torch.empty((0, args.num_dims))
    else:
        raise ValueError(f"Samples file not found at {file_name}.")

    if stored_samples.shape[0] < N and not generate_samples_if_missing:
        raise ValueError(f"Not enough samples stored ({stored_samples.shape[0]}) to load {N} samples.")
    elif stored_samples.shape[0] < N:
        print(f"Generating additional samples {N - stored_samples.shape[0]} to reach {N} samples.")
        distribution = get_distribution(**vars(args))
        samples = torch.cat((stored_samples, distribution.sample((N - stored_samples.shape[0],))))
        pickle_dump(samples, file_name)
    else:
        samples = stored_samples[:N]
    return samples


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
    
    def _slice(self, N_train: Optional[Union[int, List[int]]] = None, M: Optional[Union[int, List[int]]] = None) -> "TimeLoggerPartition":
        N_train = [N_train] if not isinstance(N_train, list) else N_train
        M = [M] if not isinstance(M, list) else M
    
        new_data = {
            key: self.data[key]
            for i in N_train
            for j in M
            for key in self.keys(N_train=i, M=j)
        }
        return self.__class__(new_data)

    def time_at(self, key: Tuple[int, int]) -> torch.Tensor:
        return self.data[key]

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

def load_time_logger_partition(
    args
) -> TimeLoggerPartition:
    if os.path.exists(args.partitions_timing_file):
        return pickle_load(args.partitions_timing_file)
    else:
        return TimeLoggerPartition()

@dataclass
class ListOfTimeLoggerPartition:
    data: List[TimeLoggerPartition] = field(default_factory=list)
    
    def append(self, rec: TimeLoggerPartition) -> None:
        self.data.append(rec)
    
    def keys(self, N_train: Optional[int] = None, M: Optional[int]= None) -> List[Tuple[int, int]]:
        sets = [set(elem.keys(N_train=N_train, M=M)) for elem in self.data]
        return list(set.intersection(*sets))

    def _slice(self: "ListOfTimeLoggerPartition", N_train: Optional[Union[int, List[int]]] = None, M: Optional[Union[int, List[int]]] = None) -> "ListOfTimeLoggerPartition":
        new_data = [elem._slice(N_train=N_train, M=M) for elem in self.data]
        return self.__class__(new_data)

    @property
    def time_stack(self) -> torch.Tensor:
        return torch.stack([
            torch.stack([elem.time_at(key) for elem in self.data])
            for key in self.keys()
        ])
    
    @property
    def mean_time(self) -> torch.Tensor:
        return torch.tensor([
            torch.stack([elem.time_at(key) for elem in self.data]).mean().item() 
            for key in self.keys()
        ])
    
    @property
    def std_time(self) -> torch.Tensor:
        return torch.tensor([
            torch.stack([elem.time_at(key) for elem in self.data]).std().item() 
            for key in self.keys()
        ])

def load_list_of_time_logger_partition(
    args, 
    random_seed_options,
) -> ListOfTimeLoggerPartition:
    original_random_seed = args.random_seed
    data = ListOfTimeLoggerPartition()
    for seed in random_seed_options:
        args.random_seed = seed
        args = process_args(args)

        data.append(load_time_logger_partition(args))

    args.random_seed = original_random_seed
    return data

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

    # Load samples 
    if 'UCI-' in args.distribution:
        samples = load_samples(args, N=max([N for N, M in combinations]), generate_samples_if_missing=False, to_construct_quantization=False)
    else:
        samples = stored_partitions.samples
    
    # Generate missing combinations
    if generate_partition_if_missing and len(missing_combinations) > 0:
        missing_partitions = generate_partitions(missing_combinations, args, samples=samples)
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
    time_logger = load_time_logger_partition(args)

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

    num_samples_options = [1000, 5000]
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


