import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection

from confidence import Confidence

def get_bounds_from_confidence_list(confidence_list: list[Confidence]):

    lower = [conf.lower_proba.item() for conf in confidence_list]
    upper = [conf.upper_proba.item() for conf in confidence_list]

    return lower, upper


def _voronoi_halfspaces(P: torch.Tensor, i: int) -> np.ndarray:
    """
    Build halfspaces for cell i: ||x - p_i||^2 <= ||x - p_j||^2, j != i
    in A x + b <= 0 form for HalfspaceIntersection.
    """
    pi = P[i]                                 # (2,)
    Pi2 = (pi * pi).sum()                     # scalar
    pj = P[torch.arange(P.shape[0]) != i]     # (N-1, 2)
    a  = pj - pi                              # (N-1, 2)
    b  = 0.5 * (Pi2 - (pj * pj).sum(dim=1))   # (N-1,)
    hs = torch.cat([a, b[:, None]], dim=1)    # (N-1, 3)
    return hs.detach().cpu().numpy()

def _disk_halfspaces(center: torch.Tensor, R: float, m: int) -> np.ndarray:
    """
    Approximate the disk {x: ||x-center|| <= R} with m supporting directions.
    Each constraint: n·x + ( -n·center - R ) <= 0.
    """
    theta = torch.arange(m, dtype=center.dtype, device=center.device) * (2.0 * torch.pi / m)
    ns = torch.stack([theta.cos(), theta.sin()], dim=1)  # (m,2)
    b  = -(ns @ center) - R                              # (m,)
    hs = torch.cat([ns, b[:, None]], dim=1)              # (m,3)
    return hs.detach().cpu().numpy()

def _order_polygon(pts_np: np.ndarray) -> np.ndarray:
    c = pts_np.mean(axis=0)
    ang = np.arctan2(pts_np[:,1] - c[1], pts_np[:,0] - c[0])
    return pts_np[np.argsort(ang)]

def plot_clipped_voronoi_2d(
    centers: torch.Tensor,
    max_diameters,                 # float or torch.Tensor of shape (N,)
    m_directions: int = 64,
    ax=None,
    point_size: float = 18.0,
    face_alpha: float = 0.25,
    edge_width: float = 1.0,
):
    """
    Plot a 2D Voronoi partition clipped to per-cell max diameters.

    Parameters
    ----------
    centers : torch.Tensor, shape (N, 2)
        Sites in R^2 (torch only).
    max_diameters : float or torch.Tensor (N,)
        Max cell diameter(s). Each cell i is intersected with a disk of radius D_i/2.
        If a float is given, it is broadcast to all cells.
    m_directions : int
        Number of directions to approximate the clipping disk (>=16 recommended).
    ax : matplotlib.axes.Axes or None
        Target axes. If None, a new figure is created.
    point_size : float
        Marker size for centers.
    face_alpha : float
        Polygon face alpha.
    edge_width : float
        Width of polygon edges.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if not (isinstance(centers, torch.Tensor) and centers.ndim == 2 and centers.shape[1] == 2):
        raise ValueError("centers must be a torch.Tensor of shape (N, 2).")

    N = centers.shape[0]
    if isinstance(max_diameters, (int, float)):
        diams = torch.full((N,), float(max_diameters), dtype=centers.dtype, device=centers.device)
    elif isinstance(max_diameters, torch.Tensor):
        if max_diameters.ndim == 0:
            diams = max_diameters.expand(N)
        else:
            assert max_diameters.shape == (N,), "max_diameters must be scalar or shape (N,)."
            diams = max_diameters
    else:
        raise ValueError("max_diameters must be a float or torch.Tensor.")

    radii = 0.5 * diams
    if (radii <= 0).any():
        raise ValueError("All max diameters must be positive.")

    # Make axes
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_ax = True

    polys = []
    for i in range(N):
        R = float(radii[i].item())
        hs_voro = _voronoi_halfspaces(centers, i)
        hs_disk = _disk_halfspaces(centers[i], R, m=m_directions)
        hs = np.vstack([hs_voro, hs_disk])

        # Slight nudge keeps interior point strictly feasible for all halfspaces
        interior = (centers[i] + 1e-9).detach().cpu().numpy()
        hsi = HalfspaceIntersection(hs, interior_point=interior)
        verts = _order_polygon(hsi.intersections.astype(np.float64))
        polys.append(verts)

        ax.fill(verts[:,0], verts[:,1], alpha=face_alpha)
        ax.plot(verts[:,0], verts[:,1], linewidth=edge_width, color="k")

    # C = centers.detach().cpu().numpy()
    # ax.scatter(C[:,0], C[:,1], s=point_size, c="k", zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Voronoi partition clipped to max diameters (D_i)")
    if created_ax:
        plt.tight_layout()
    return ax

def set_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.dpi": 200,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    })

def convert_to_sci_notation(N):
    s = f"{N:.0e}"
    base, exp = s.split("e")
    exp = int(exp)
    return fr"{base} \times 10^{{{exp}}}"
