import os
import torch
import time
from argparse import Namespace
from typing import List, Dict, Tuple, Optional, Any, Generic, TypeVar, Union, Type

from quantization import UncertainQuantization
from bound import DataDrivenRadius, fournier_radius as compute_fournier_radius, EmpiricalRadius
from solvers import get_solver
from sets import BoundedVoronoiPartition

from configs.handlers import pickle_load, pickle_dump, process_args
from configs.construct import get_support_assumption, get_distribution
from experiments.partitions import get_dict_of_partitions
from experiments.datastructures import ListOfTimeLogger, TimeLogger, DataDrivenRadii, FournierRadii, EmpiricalRadii, Quantizations, _GridDict, ListOfDataDrivenRadii


def quantizations_for_combinations(
    args: Namespace, 
    combinations: List[Tuple[int, int]],  # List of (N, M) pairs
    generate_partition_if_missing: bool = False,
) -> Quantizations:
    partitions = get_dict_of_partitions(
        args, 
        num_samples_options=[args.num_samples_training], 
        num_clusters_options=[M for N, M in combinations], 
        generate_partition_if_missing=generate_partition_if_missing
    )

    quantizations = Quantizations()
    N_train = args.num_samples_training
    for (N, M) in combinations:
        if (N_train, M) not in partitions.keys():
            print(f"Skipping N_train={N_train}, M={M} as partition is not available.")
            continue
        else:
            partition = partitions.at((N_train, M))

            quantizations.append((N_train, N, M), load_quantization(args, partition=partition, N=N))

    return quantizations

def data_driven_radii_for_combinations(
    args: Namespace, 
    combinations: List[Tuple[int, int]],  # List of (N, M) pairs
    time_limit: Optional[float] = None,
    generate_partition_if_missing: bool = False,
    return_all_available_combinations: bool = False,
    generate_data_driven_radii_if_not_stored: bool = True,
) -> Tuple[DataDrivenRadii, TimeLogger]:
    solver = get_solver(method=args.method)

    partitions = get_dict_of_partitions(
        args, 
        num_samples_options=[args.num_samples_training], 
        num_clusters_options=[M for N, M in combinations], 
        generate_partition_if_missing=generate_partition_if_missing
    )

    stored_data_driven_radii = load_data(args.data_driven_radii_file, DataDrivenRadii)
    time_logger = load_data(args.data_driven_radii_file.replace(".pickle", "_timing.pickle"), TimeLogger)

    data_driven_radii = DataDrivenRadii()

    N_train = args.num_samples_training
    for (N, M) in combinations:
        if (N_train, N, M) in stored_data_driven_radii.keys():
            print(f"Data-driven radius for N_train={N_train}, N={N}, M={M} in stored data. Skipping computation.")
            data_driven_radii.append((N_train, N, M), stored_data_driven_radii.at((N_train, N, M)))
        elif generate_data_driven_radii_if_not_stored:
            if (N_train, M) not in partitions.keys():
                print(f"Skipping N_train={N_train}, M={M} as partition is not available.")
                continue
            else:
                partition = partitions.at((N_train, M))

            print(f"Processing N_train={N_train}, N={N}, M={M}")

            quantization = load_quantization(args, partition=partition, N=N)

            try:
                start = time.time()
                data_driven_radii.append((N_train, N, M), DataDrivenRadius(
                    quantization=quantization,
                    solver=solver, 
                    wasserstein_order=args.wasserstein_order,
                    compute_discrete_bound=args.compute_discrete_bound,
                    compute_moment_bound=args.compute_moment_bound,
                    time_limit=time_limit
                ))
                time_logger.append((N_train, N, M), torch.as_tensor(time.time() - start))
            except Exception as e:
                print(f"Unexpected error for N_train={N_train}, N={N}, M={M}: {e}. Skipping this configuration.")
        else:
            pass

    if return_all_available_combinations:
        for (N_train, N, M) in stored_data_driven_radii.keys():
            if (N_train, N, M) not in data_driven_radii.keys():
                data_driven_radii.append((N_train, N, M), stored_data_driven_radii.at((N_train, N, M)))

    return data_driven_radii, time_logger


def fournier_radii_for_combinations(
    args: Namespace, 
    combinations: List[Tuple[int, int]],  # List of (N, M) pairs
) -> FournierRadii:
    support = get_support_assumption(**vars(args))

    fournier_radii = load_data(args.fournier_radii_file, FournierRadii)

    N_train = args.num_samples_training
    for N in list(set([N for N, M in combinations])):
        fournier_radius = compute_fournier_radius(
            support=support, 
            nsamples=N + N_train, 
            wasserstein_order=args.wasserstein_order,
            beta=args.beta,
        )
        for M in [M for N_, M in combinations if N_ == N]:
            fournier_radii.append((N_train, N, M), fournier_radius)

    return fournier_radii


def load_data(
    file: str, 
    Class: Type[_GridDict], 
    num_samples_training_options: Optional[List[int]] = None,
    num_samples_options: Optional[List[int]] = None, 
    num_clusters_options: Optional[List[int]] = None,
    skip_missing_combinations: bool = False,
):
    if os.path.exists(file):
        stored_data = pickle_load(file)
    else:
        print(f"File not found at {file}.")
        stored_data = Class()

    if num_samples_options is None and num_clusters_options is None and num_samples_training_options is None:
        data = stored_data
    elif num_samples_options is not None and num_clusters_options is not None and num_samples_training_options is not None:
        combinations = [(N_train, N, M) for N in num_samples_options for M in num_clusters_options for N_train in num_samples_training_options]
        data = Class()
        for N_train, N, M in combinations:
            if (N_train, N, M) in stored_data.keys():
                data.append((N_train, N, M), stored_data.at((N_train, N, M)))
            elif skip_missing_combinations:
                print(f"Skipping missing combination N_train={N_train}, N={N}, M={M}.")
            else:
                raise KeyError(f"{type(data)} for N_train={N_train}, N={N}, M={M} not found in stored partitions.")
    else:
        raise ValueError("Either both num_samples_options and num_clusters_options must be provided, or neither.")
    
    return data


def load_quantization_samples(args, N: int, generate_samples_if_missing: bool = True) -> torch.Tensor:
    if os.path.exists(args.quantization_samples_file):
        stored_samples = pickle_load(args.quantization_samples_file)
    elif generate_samples_if_missing:
        # print(f"generating samples to be stored at {args.quantization_samples_file}.")
        stored_samples = torch.empty((0, args.num_dims))
    else:
        raise ValueError(f"Quantization samples file not found at {args.quantization_samples_file}.")

    if stored_samples.shape[0] < N and not generate_samples_if_missing:
        raise ValueError(f"Not enough samples stored ({stored_samples.shape[0]}) to load {N} samples.")
    elif stored_samples.shape[0] < N:
        print(f"Generating additional samples {N - stored_samples.shape[0]} to reach {N} samples.")
        distribution = get_distribution(**vars(args))
        samples = torch.cat((stored_samples, distribution.sample((N - stored_samples.shape[0],))))
        pickle_dump(samples, args.quantization_samples_file)
    else:
        samples = stored_samples[:N]
    return samples


def load_quantization(args, partition: BoundedVoronoiPartition, N: int) -> UncertainQuantization:
    return UncertainQuantization(
        partition=partition, 
        samples=load_quantization_samples(args, N=N),
        beta=args.beta
    )

def load_list_of_data_driven_radii(
    args, 
    combinations, 
    random_seed_options
) -> ListOfDataDrivenRadii:
    original_random_seed = args.random_seed
    data = ListOfDataDrivenRadii()
    for seed in random_seed_options:
        args.random_seed = seed
        args = process_args(args)
        data.append(data_driven_radii_for_combinations(
            args, 
            combinations=combinations, 
            generate_partition_if_missing=False, 
            return_all_available_combinations=False,
            generate_data_driven_radii_if_not_stored=False
        )[0])
    args.random_seed = original_random_seed
    return data

def load_list_of_time_loggers(
    args, 
    random_seed_options,
) -> ListOfTimeLogger:
    original_random_seed = args.random_seed
    data = ListOfTimeLogger()
    for seed in random_seed_options:
        args.random_seed = seed
        args = process_args(args)

        data.append(load_data(args.data_driven_radii_timing_file, TimeLogger))

    args.random_seed = original_random_seed
    return data