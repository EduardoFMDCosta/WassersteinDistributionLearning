import torch
from sets import HyperRectangle

def in_set(samples: torch.Tensor,
                          regions: HyperRectangle):

    samples_expanded = samples.unsqueeze(0)
    lower_expanded = regions.lower.unsqueeze(1)
    upper_expanded = regions.upper.unsqueeze(1)

    # Check inclusion
    inside_lower = samples_expanded >= lower_expanded
    inside_upper = samples_expanded <= upper_expanded
    inside_all = inside_lower & inside_upper

    inside = inside_all.all(dim=-1).sum(dim=1).float()
    return int(inside.item())