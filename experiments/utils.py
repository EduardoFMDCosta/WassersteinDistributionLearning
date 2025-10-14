from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Generic, TypeVar, Union
import torch

from bound import DataDrivenRadius
from quantization import UncertainQuantization


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