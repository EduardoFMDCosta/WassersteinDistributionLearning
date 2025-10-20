from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Generic, TypeVar, Union

import ot
import torch
import time
import os
import csv
from argparse import Namespace

from quantization import UncertainQuantization
from sets import BoundedVoronoiPartition
from bound import DataDrivenRadius, fournier_radius as compute_fournier_radius
from solvers import get_solver

from configs.construct import get_support_assumption, get_distribution
import torch.distributions as ds


## -- Data Structure ------------------------------------------------------------------------------------------------ ##

T = TypeVar("T")

S = TypeVar("S", bound="_GridDict")

@dataclass
class _GridDict(Generic[T]): # key = (N, M)
    data: Dict[Tuple[int, int], T] = field(default_factory=dict)

    def append(self, key: Tuple[int, int], rec: T) -> None:
        self.data[key] = rec

    def at(self, key: Tuple[int, int]) -> T:
        return self.data[key]
    
    def keys(self, N: Optional[int] = None, M: Optional[int]= None) -> List[Tuple[int, int]]:
        return [key for key in self.data.keys() if (N is None or key[0] == N) and (M is None or key[1] == M)]

    def _stack(self, attribute: str, N: Optional[int] = None, M: Optional[int]= None) -> torch.Tensor:
        return torch.tensor([getattr(self.data[key], attribute) for key in self.keys(N=N, M=M)])

    def _slice(self: S, N: Optional[int] = None, M: Optional[int] = None) -> S:
        new_data = {key: self.data[key] for key in self.keys(N=N, M=M)}
        return self.__class__(new_data)


class TimeLogger(_GridDict[torch.Tensor]):
    @property
    def time(self):
        return self._stack('data')


class DataDrivenRadii(_GridDict[DataDrivenRadius]): 
    @property
    def moment_bound(self):
        return self._stack('moment_bound')

    @property
    def discrete_bound(self):
        return self._stack('discrete_bound')

    @property
    def lower_bound(self):
        return self._stack('lower_bound')

    @property
    def radius(self):
        return self._stack('radius')


class FournierRadii(_GridDict[Union[torch.Tensor, float]]):
    @property
    def radius(self):
        return self._stack('data')


class Quantizations(_GridDict[UncertainQuantization]):
    def _mean_stack(self, attribute: str, N: Optional[int] = None, M: Optional[int]= None) -> torch.Tensor:
        return torch.tensor([getattr(self.data[key], attribute).float().mean() for key in self.keys(N=N, M=M)])

    def _std_stack(self, attribute: str, N: Optional[int] = None, M: Optional[int]= None) -> torch.Tensor:
        return torch.tensor([getattr(self.data[key], attribute).float().std() for key in self.keys(N=N, M=M)])
    
    def _min_stack(self, attribute: str, N: Optional[int] = None, M: Optional[int]= None) -> torch.Tensor:
        return torch.tensor([getattr(self.data[key], attribute).float().min() for key in self.keys(N=N, M=M)])
    
    def _max_stack(self, attribute: str, N: Optional[int] = None, M: Optional[int]= None) -> torch.Tensor:
        return torch.tensor([getattr(self.data[key], attribute).float().max() for key in self.keys(N=N, M=M)])

    @property
    def outer_counts(self):
        return self._stack('outer_counts')
    
    @property
    def mean_cluster_counts(self):
        return self._mean_stack('cluster_counts')

    @property
    def std_cluster_counts(self):
        return self._std_stack('cluster_counts')
    
    @property
    def mean_lower_probs(self):
        return self._mean_stack('lower_probs')

    @property
    def mean_upper_probs(self):
        return self._mean_stack('upper_probs')

    @property
    def mean_probs(self):
        return self._mean_stack('probs')

    @property
    def mean_range_probs(self):
        return torch.tensor([(self.data[key].upper_probs - self.data[key].lower_probs).mean() for key in self.keys()])

    @property
    def std_range_probs(self):
        return torch.tensor([(self.data[key].upper_probs - self.data[key].lower_probs).std() for key in self.keys()])

    @property
    def mean_cluster_radii(self):
        return self._mean_stack('cluster_radii')

    @property
    def std_cluster_radii(self):
        return self._std_stack('cluster_radii')

    @property
    def min_cluster_radii(self):
        return self._min_stack('cluster_radii')

    @property
    def max_cluster_radii(self):
        return self._max_stack('cluster_radii')

    @property
    def mean_distances_locs(self):
        return self._mean_stack('distance_locs')

    @property
    def std_distances_locs(self):
        return self._std_stack('distance_locs')


class EmpiricalRadius:
    def __init__(
            self,
            quantization: UncertainQuantization,
            fournier_samples: torch.Tensor,
            distribution: ds.Distribution,
    ):
        self._N = fournier_samples.shape[0]
        self._M = quantization.locs.shape[0]
        self._distribution_representation = distribution.sample((10 * self._N,))

        self._empirical_wasserstein = ot.solve_sample(X_a=self._distribution_representation, X_b=fournier_samples).value.sqrt().item()
        self._quantization_wasserstein = ot.solve_sample(X_a=self._distribution_representation, X_b=quantization.locs, b=quantization.probs).value.sqrt().item()

    @property
    def empirical_wasserstein(self) -> float:
        return self._empirical_wasserstein

    @property
    def quantization_wasserstein(self) -> float:
        return self._quantization_wasserstein

class EmpiricalRadii(_GridDict[EmpiricalRadius]):
    @property
    def empirical_wasserstein(self):
        return self._stack('empirical_wasserstein')

    @property
    def quantization_wasserstein(self):
        return self._stack('quantization_wasserstein')

## -- Experiment Function -------------------------------------------------------------------------------------------- ##
def estimate_memory_usage(N, M, num_dims=2):
    """Estimate memory usage for given N, M parameters"""
    # Samples tensor: N x num_dims
    samples_memory = N * num_dims * 8 / (1024 * 1024)  # 8 bytes for float64
    
    # Distance matrix for K-means: N x M (potentially)
    distance_matrix_memory = N * M * 8 / (1024 * 1024)
    
    # Cluster assignments: N elements
    assignments_memory = N * 8 / (1024 * 1024)
    
    total_estimated_mb = samples_memory + distance_matrix_memory + assignments_memory
    return total_estimated_mb

def estimate_kmeans_complexity(N, k, num_dims=2, max_iterations=100):
    """Estimate computational complexity of K-means"""
    # K-means complexity is roughly O(n * k * d * iterations)
    # where n=samples, k=clusters, d=num_dimss, iterations=convergence steps
    operations = N * k * num_dims * max_iterations
    return operations

def assess_kmeans_feasibility(N, k, num_dims=2, max_memory_mb=2000, max_operations=1e9):
    """Pre-assess if K-means is likely to be feasible"""
    reasons = []
    
    # Memory check
    memory_mb = estimate_memory_usage(N, k, num_dims)
    if memory_mb > max_memory_mb:
        reasons.append(f"Memory: {memory_mb:.1f}MB > {max_memory_mb}MB limit")
    
    # Computational complexity check
    operations = estimate_kmeans_complexity(N, k, num_dims)
    if operations > max_operations:
        reasons.append(f"Complexity: {operations:.1e} > {max_operations:.1e} operations")

    
    estimated_time_seconds = operations / 1e8  # Rough estimate: 100M operations per second

    is_feasible = len(reasons) == 0
    return is_feasible, reasons, {
        'memory_mb': memory_mb,
        'operations': operations,
        'estimated_time_s': estimated_time_seconds
    }

def check_memory_feasibility(N, M, max_memory_mb=8000):  # 8GB default limit
    """Check if the operation is likely to fit in memory"""
    estimated_memory = estimate_memory_usage(N, M)
    if estimated_memory > max_memory_mb:
        print(f"Estimated memory usage: {estimated_memory:.1f} MB (exceeds limit of {max_memory_mb} MB)")
        return False
    print(f"Estimated memory usage: {estimated_memory:.1f} MB")
    return True


def run_combinations(
    args: Namespace, 
    M_options: List[int], 
    N_options: List[int], 
    max_memory_mb: int = 2000, 
    print_timings: bool = False,
    compute_empirical_radii: bool = False,
):
    solver = get_solver(method=args.method)

    distribution = get_distribution(**vars(args))
    support_assumption = get_support_assumption(**vars(args))
    
    quantization_times, radius_computation_times, computation_times = TimeLogger(), TimeLogger(), TimeLogger()
    quantizations, data_driven_radii, fournier_radii, empirical_radii = Quantizations(), DataDrivenRadii(), FournierRadii(), EmpiricalRadii()
    for N in N_options:
        samples_partition = distribution.sample((N,))
        samples_quantization = distribution.sample((N,))

        for M in M_options:
            is_feasible, reasons, stats = assess_kmeans_feasibility(
                N=N, k=M, num_dims=2, 
                max_memory_mb=max_memory_mb, 
                max_operations=1e9
            )
            if not is_feasible:
                print(f"Skipping M={M}, N={N}. Reasons: {'; '.join(reasons)}")
                print(f"  Stats: {stats['memory_mb']:.1f}MB, {stats['operations']:.1e} ops, ~{stats['estimated_time_s']:.1f}s")
                continue

            print(f"Proceeding with M={M}, N={N} (estimated: {stats['memory_mb']:.1f}MB, ~{stats['estimated_time_s']:.1f}s)")
            
            try:
                start = time.time()
                partition = BoundedVoronoiPartition(
                    support=support_assumption, 
                    samples=samples_partition, 
                    M=M
                )
                quantizations.append((N, M),  UncertainQuantization(
                    partition=partition, 
                    samples=samples_quantization, 
                    beta=args.beta
                ))
                quantization_times.append((N, M), torch.as_tensor(time.time() - start))
                if print_timings:
                    print(f"K-means completed in {quantization_times.at((N, M)):.2f} seconds")

                start = time.time()
                data_driven_radii.append((N, M), DataDrivenRadius(
                    quantization=quantizations.at((N, M)),
                    solver=solver, 
                    compute_moment_bound=args.compute_moment_bound, 
                    compute_discrete_bound=args.compute_discrete_bound
                ))
                radius_computation_times.append((N, M), torch.as_tensor(time.time() - start))
                if print_timings:
                    print(f"Data-driven bounding completed in {radius_computation_times.at((N, M)):.2f} seconds")

                fournier_radii.append((N, M), compute_fournier_radius(support=partition.support, nsamples=2*N, beta=args.beta))

                computation_times.append((N, M), quantization_times.at((N, M)) + radius_computation_times.at((N, M)))

            except Exception as e:
                print(f"Unexpected error for M={M}, N={N}: {e}. Skipping this configuration.")
                continue

            if compute_empirical_radii:
                empirical_radii.append((N, M), EmpiricalRadius(
                    quantization=quantizations.at((N, M)),
                    fournier_samples=samples_partition,
                    distribution=distribution,
                ))

    return (quantizations, data_driven_radii, fournier_radii, empirical_radii), (quantization_times, radius_computation_times, computation_times)

def generate_table(data_driven_radii: _GridDict,
                   fournier_radii: _GridDict,
                   empirical_radii: _GridDict,
                   args):

    data_data = data_driven_radii.data
    fournier_data = fournier_radii.data
    empirical_data = empirical_radii.data

    # Prepare CSV file name
    results_dir = os.path.join(args.results_dir, args.distribution.lower())
    csv_path = os.path.join(results_dir, f"ndims={args.num_dims}_set={args.setting}_radii.csv")

    # Prepare rows
    rows = []
    for (N, M), bounds in data_data.items():
        # Extract bounds
        moment_bound = bounds.moment_bound.item()
        discrete_bound = bounds.discrete_bound.item()
        total_bound = moment_bound + discrete_bound
        fournier_value = fournier_data.get((N, M), float("nan"))
        empirical_wasserstein = empirical_data.get((N, M)).empirical_wasserstein
        quantization_wasserstein = empirical_data.get((N, M)).quantization_wasserstein

        rows.append({
            "Distribution": args.distribution,
            "Dimension": args.num_dims,
            "Support radius": args.support_linf_radius,
            "N": N,
            "M": M,
            "Moment bound": f"{moment_bound:.2f}",
            "Discrete bound": f"{discrete_bound:.2f}",
            "Ours": f"{total_bound:.2f}",
            "Fournier": f"{fournier_value:.2f}",
            "Empirical Wasserstein": f"{empirical_wasserstein:.2f}",
            "Quantization Wasserstein": f"{quantization_wasserstein:.2f}",
        })

    # Write to CSV
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Distribution",
                "Dimension",
                "Support radius",
                "N",
                "M",
                "Moment bound",
                "Discrete bound",
                "Ours",
                "Fournier",
                "Empirical Wasserstein",
                "Quantization Wasserstein",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)