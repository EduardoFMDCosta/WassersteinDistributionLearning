import torch
from sets import HyperRectangle
import itertools

def in_set(samples: torch.Tensor,
           regions: HyperRectangle,
           include_complement: bool=False):

    samples_expanded = samples.unsqueeze(0)
    lower_expanded = regions.lower.unsqueeze(1)
    upper_expanded = regions.upper.unsqueeze(1)

    # Check inclusion
    inside_lower = samples_expanded >= lower_expanded
    inside_upper = samples_expanded <= upper_expanded
    inside_all = inside_lower & inside_upper

    inside = inside_all.all(dim=-1).sum(dim=1).float()

    if include_complement:
        # Count samples in complement (N - sum(inside for all sets))
        complement = samples.shape[0] - inside.sum()
        inside = torch.cat([inside, complement.unsqueeze(0)])

    return inside

def subdivide_hyperrectangle(region: HyperRectangle, n: int) -> HyperRectangle:

    lower = region.lower.squeeze(0)  # (d,)
    upper = region.upper.squeeze(0)  # (d,)
    d = lower.shape[0]

    # Compute split points (n+1 edges) per dimension
    edges = [torch.linspace(lower[i], upper[i], n + 1) for i in range(d)]

    # Compute all index combinations (n^d)
    idx_combinations = list(itertools.product(range(n), repeat=d))

    # Initialize lower and upper bounds for each subregion
    lowers = []
    uppers = []
    for idx in idx_combinations:
        sub_lower = torch.tensor([edges[dim][i] for dim, i in enumerate(idx)])
        sub_upper = torch.tensor([edges[dim][i + 1] for dim, i in enumerate(idx)])
        lowers.append(sub_lower)
        uppers.append(sub_upper)

    lowers = torch.stack(lowers, dim=0)  # shape (n^d, d)
    uppers = torch.stack(uppers, dim=0)  # shape (n^d, d)

    return HyperRectangle(lowers, uppers)