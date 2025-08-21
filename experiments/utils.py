from typing import List
import torch

from bound import DataDrivenRadius
from quantization import UncertainQuantization


class RadiiStatistics:
    def __init__(self, data_driven_radii: List[DataDrivenRadius]):
        self.data_driven_radii = data_driven_radii

    @property
    def moment_bound(self): 
        return torch.tensor([elem.moment_bound for elem in self.data_driven_radii])

    @property
    def discrete_bound(self):
        return torch.tensor([elem.discrete_bound for elem in self.data_driven_radii])
    
    @property
    def lower_bound(self):
        return torch.tensor([elem.lower_bound for elem in self.data_driven_radii])

    @property
    def radius(self):
        return torch.tensor([elem.radius for elem in self.data_driven_radii])
    
class UncertainQuantizationStatistics:
    def __init__(self, quantizations: List[UncertainQuantization]):
        self.quantizations = quantizations

    @property
    def outer_counts(self):
        return torch.tensor([elem.outer_counts for elem in self.quantizations])
    
    @property
    def cluster_counts_avg(self):
        return torch.tensor([elem.cluster_counts.float().mean() for elem in self.quantizations])

    @property
    def cluster_counts_std(self):
        return torch.tensor([elem.cluster_counts.float().std() for elem in self.quantizations])

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
    def range_probs_avg(self):
        return torch.tensor([(elem.upper_probs - elem.lower_probs).mean() for elem in self.quantizations])
    
    @property
    def range_probs_std(self):
        return torch.tensor([(elem.upper_probs - elem.lower_probs).std() for elem in self.quantizations])

    @property
    def cluster_radius_avg(self):
        return torch.tensor([elem.partition.cluster_radii.mean() for elem in self.quantizations])
    
    @property
    def cluster_radius_std(self):
        return torch.tensor([elem.partition.cluster_radii.std() for elem in self.quantizations])
    
    @property
    def cluster_radius_min(self):
        return torch.tensor([elem.partition.cluster_radii.min() for elem in self.quantizations])

    @property
    def cluster_radius_max(self):
        return torch.tensor([elem.partition.cluster_radii.max() for elem in self.quantizations])

    @property
    def distances_locs_avg(self):
        return torch.tensor([elem.partition.distance_locs[:-1,:-1].mean() for elem in self.quantizations])
    
    @property
    def distances_locs_std(self):
        return torch.tensor([elem.partition.distance_locs[:-1,:-1].std() for elem in self.quantizations])