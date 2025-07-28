import torch

def in_set(samples,
           regions,
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

def generate_grid_from_samples(samples, num_points_per_dim: int):
    d = samples.shape[1]
    mins, _ = samples.min(dim=0)
    maxs, _ = samples.max(dim=0)

    # Create linspace for each dimension
    grids_1d = [torch.linspace(mins[i], maxs[i], num_points_per_dim) for i in range(d)]

    # Create meshgrid
    mesh = torch.meshgrid(*grids_1d, indexing='ij')  # use 'ij' for matrix-style indexing

    # Flatten the meshgrid to (M, d)
    grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)  # shape (M, d)
    return grid

def generate_grid_from_locs(locs: torch.Tensor):
    n, d = locs.shape
    unique_vals = [torch.unique(locs[:, i]) for i in range(d)]
    mesh = torch.meshgrid(*unique_vals, indexing="ij")
    grid = torch.stack([m.flatten() for m in mesh], dim=-1)

    return grid