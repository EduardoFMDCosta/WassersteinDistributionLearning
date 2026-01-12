import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from scipy.spatial import Voronoi

from configs.handlers import parse_arguments
from experiments.utils import quantizations_for_combinations

from plotting.utils_plot import set_style

set_style()


@dataclass(frozen=True)
class VoronoiMaxRadius:
    """
    Compute r_i = max_{x in R_i} ||x - c_i||_2 for Voronoi regions R_i induced by sites c_i.

    Notes:
    - If the Voronoi cell is unbounded, r_i = +inf.
    - If bounded, r_i is the max distance from c_i to the Voronoi vertices of its region.
    """
    qhull_options: Optional[str] = "Qbb Qc Qx"  # reasonable defaults for many N-d cases

    def __call__(self, C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C: (n, d) tensor of sites (float32/float64), on CPU or GPU.

        Returns:
            radii: (n,) tensor on the same device/dtype as C, with +inf for unbounded cells.
        """
        if C.ndim != 2:
            raise ValueError(f"C must have shape (n, d); got {tuple(C.shape)}")
        n, d = C.shape
        if n <= d:
            # Qhull typically needs at least d+1 points for a full-dimensional diagram.
            return torch.full((n,), float("inf"), device=C.device, dtype=C.dtype)

        # SciPy works on CPU numpy
        C_np = C.detach().to("cpu").double().numpy()

        vor = Voronoi(C_np, qhull_options=self.qhull_options)

        radii_np = np.full((n,), np.inf, dtype=np.float64)

        for i in range(n):
            region_id = vor.point_region[i]
            region = vor.regions[region_id]

            # Empty or contains -1 => unbounded region
            if (not region) or (-1 in region):
                radii_np[i] = np.inf
                continue

            verts = vor.vertices[np.asarray(region, dtype=np.int64)]  # (m_i, d)
            diff = verts - C_np[i]                                    # (m_i, d)
            radii_np[i] = np.linalg.norm(diff, axis=1).max()

        return torch.as_tensor(radii_np, device=C.device, dtype=C.dtype)


            
if __name__ == '__main__':    
    args = parse_arguments( # Only parse arguments once, updated afterwards
        random_seed=0,
        distribution='GaussianMixture',
        num_dims=2,
        setting=0,
        num_samples=10_000,
        num_clusters=10,
        save=True,
    )

    M_options = [5, 20, 30, 40, 50, 75, 100]

    combinations = [(args.num_samples_training, args.num_samples, M) for M in M_options]

    quantizations = quantizations_for_combinations(
        args, 
        combinations=combinations, 
        generate_partition_if_missing=False
    )

    tag = f"seed={args.random_seed}"

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    M_options_plot = [key[2] for key in quantizations.keys()]
    
    solver = VoronoiMaxRadius()
    true_l2_radii_mean, true_l2_radii_std = list(), list()
    for M in M_options_plot:
        quantization = quantizations.at((args.num_samples_training, args.num_samples, M))
        r = solver(quantization.locs)
        true_l2_radii = torch.min(r, quantization.l2_radii)
        true_l2_radii_mean.append(true_l2_radii.mean().item())
        true_l2_radii_std.append(true_l2_radii.std().item())

    true_l2_radii_mean, true_l2_radii_std = torch.as_tensor(true_l2_radii_mean), torch.as_tensor(true_l2_radii_std)

    ax.plot(M_options_plot, true_l2_radii_mean, color="blue", marker='o')
    ax.fill_between(
        M_options_plot,
        true_l2_radii_mean - true_l2_radii_std,
        true_l2_radii_mean + true_l2_radii_std,
        color="blue",
        alpha=0.2,
    )

    ax.plot(M_options_plot, quantizations.mean_region_l2_radii, color="black", marker='o')
    ax.fill_between(
        M_options_plot,
        quantizations.mean_region_l2_radii - quantizations.std_region_l2_radii,
        quantizations.mean_region_l2_radii + quantizations.std_region_l2_radii,
        color="black",
        alpha=0.2,
    )

    ax.legend(handles=[
        Patch(facecolor="blue", edgecolor="blue",
            label=r"$\max_{x\in R_i}\|x - c_i\|$"),
        Patch(facecolor="black", edgecolor="black",
            label=r"$r_i$")
    ])
    ax.set_xlabel(r"Support size $M$")

    if args.save:
        fig.savefig(os.path.join(args.figures_dir, f"radii_heuristic_{tag}.pdf"))
        plt.close('all')
    else:
        plt.show()
