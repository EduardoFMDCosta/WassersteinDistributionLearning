import os
import torch
import time
from argparse import Namespace
from typing import List, Dict, Tuple, Optional, Any, Generic, TypeVar, Union, Type

from quantization import UncertainQuantization
from bound import DataDrivenRadius, fournier_radius as compute_fournier_radius, EmpiricalRadius
from solvers import get_solver

from configs.handlers import pickle_load
from configs.construct import get_support_assumption, get_distribution
from experiments.partitions import get_dict_of_partitions
from experiments.datastructures import TimeLogger, DataDrivenRadii, FournierRadii, EmpiricalRadii, Quantizations, _GridDict


def data_driven_radii_for_combinations(
    args: Namespace, 
    combinations: List[Tuple[int, int]],  # List of (N, M) pairs
    time_limit: Optional[float] = None,
    generate_partition_if_missing: bool = False,
) -> Tuple[Tuple[Quantizations, DataDrivenRadii], Tuple[TimeLogger, TimeLogger]]:
    solver = get_solver(method=args.method)

    distribution = get_distribution(**vars(args))

    partitions = get_dict_of_partitions(
        args, 
        num_samples_options=[args.num_samples_training], 
        num_clusters_options=[M for N, M in combinations], 
        generate_partition_if_missing=generate_partition_if_missing
    )

    N_max = max([N for N, M in combinations])
    samples_quantization = distribution.sample((N_max,))

    quantizations = Quantizations()
    data_driven_radii = load_data(args.data_driven_radii_file, DataDrivenRadii)

    quantization_times, radius_computation_times = TimeLogger(), TimeLogger()

    N_train = args.num_samples_training
    for (N, M) in combinations:
        if (N, M) in data_driven_radii.keys():
            print(f"Data-driven radius for M={M}, N={N} in stored data. Skipping computation.")
            continue
    
        if (N_train, M) not in partitions.keys():
            print(f"Skipping M={M}, N={N_train} as partition is not available.")
            continue

        print(f"Processing M={M}, N={N}")
        start = time.time()
        partition = partitions.at((N_train, M))
        quantizations.append((N_train, N, M),  UncertainQuantization(
            partition=partition, 
            samples=samples_quantization[:N], 
            beta=args.beta
        ))
        quantization_times.append((N_train, N, M), torch.as_tensor(time.time() - start))

        try:
            start = time.time()
            data_driven_radii.append((N_train, N, M), DataDrivenRadius(
                quantization=quantizations.at((N_train, N, M)),
                solver=solver, 
                wasserstein_order=args.wasserstein_order,
                compute_discrete_bound=args.compute_discrete_bound,
                compute_moment_bound=args.compute_moment_bound,
                time_limit=time_limit
            ))
            radius_computation_times.append((N_train, N, M), torch.as_tensor(time.time() - start))
        except Exception as e:
            print(f"Unexpected error for M={M}, N={N}: {e}. Skipping this configuration.")
            continue

    return (quantizations, data_driven_radii), (quantization_times, radius_computation_times)


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
