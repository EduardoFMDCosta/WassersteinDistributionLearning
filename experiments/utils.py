from typing import List
import torch

from bound import DataDrivenRadius
from quantization import UncertainQuantization


class RadiiStatistics:
    def __init__(self, data_driven_radii: List[DataDrivenRadius]):
        self.data_driven_radii = data_driven_radii

    @property
    def epsilon1(self): 
        return torch.tensor([elem.moment_bound for elem in self.data_driven_radii])
    
    @property
    def epsilon2(self):
        return torch.tensor([elem.discrete_bound for elem in self.data_driven_radii])

    @property
    def radius(self):
        return torch.tensor([elem.radius for elem in self.data_driven_radii])
    
class UncertainQuantizationStatistics:
    def __init__(self, quantizations: List[UncertainQuantization]):
        self.quantizations = quantizations

    @property
    def lower_probs_avg(self):
        return torch.tensor([elem.lower_probs.mean() for elem in self.quantizations])

    @property
    def upper_probs_avg(self):
        return torch.tensor([elem.upper_probs.mean() for elem in self.quantizations])
    
    @property
    def probs_avg(self):
        return torch.tensor([elem.probs.mean() for elem in self.quantizations])
    
    @property
    def cluster_radius_avg(self):
        return torch.tensor([elem.partition.cluster_radii.mean() for elem in self.quantizations])
    
    @property
    def cluster_radius_min(self):
        return torch.tensor([elem.partition.cluster_radii.min() for elem in self.quantizations])

    @property
    def cluster_radius_max(self):
        return torch.tensor([elem.partition.cluster_radii.max() for elem in self.quantizations])

    @property
    def distances_locs(self):
        return None # TODO self.distance_locs = torch.cdist(self.locs, self.locs, p=2)