# MERci/plots/spot_localization_plots.py
"""Plots for ``analysis.spot_localization``."""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.spot_localization import compute_display_limits


def plot_max_projections(
    volume: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    *,
    title: str = "",
    lo_pct: float = 50.0,
    hi_pct: float = 95.0,
    print_stats: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot XY, ZX, and ZY max-intensity projections of a 3-D volume.

    Contrast is set per panel using ``compute_display_limits``: vmin and vmax
    are the *lo_pct* / *hi_pct* percentiles of each projection independently,
    so each view is optimally stretched regardless of projection statistics.
    White = high intensity (fluorescence convention).

    Parameters
    ----------
    volume      : (n_z, n_y, n_x) array
    voxel_size_um : (vx, vy, vz) µm per pixel/plane — used for axis labels
                    and correct aspect ratios
    title       : figure suptitle
    lo_pct      : lower percentile for vmin (default 50.0 — sets black point
                  at the median, which is typically background for sparse images)
    hi_pct      : upper percentile for vmax (default 95.0)
    print_stats : if True, print a table of key percentiles and display limits
                  for each projection before showing the figure (default True)
    figsize     : figure size; defaults to (12, 4)

    Returns
    -------
    matplotlib Figure — caller may further customise or save it.
    """
    n_z, n_y, n_x = volume.shape
    vx, vy, vz    = voxel_size_um

    vol_f = volume.astype(np.float32)
    xy = vol_f.max(axis=0)               # (n_y, n_x)
    zx = vol_f.max(axis=1)               # (n_z, n_x)
    zy = vol_f.max(axis=2)               # (n_z, n_y)

    projections   = [("XY (z-max)", xy), ("ZX (y-max)", zx), ("ZY (x-max)", zy)]
    stat_pcts     = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    display_limits = []

    if print_stats:
        header = f"{'Projection':<14}" + "".join(f"  p{p:>3}" for p in stat_pcts)
        header += "   vmin    vmax"
        print(header)
        print("-" * len(header))

    for label, proj in projections:
        vals = np.percentile(proj, stat_pcts)
        vmin, vmax = compute_display_limits(proj, lo_pct, hi_pct)
        display_limits.append((vmin, vmax))
        if print_stats:
            row = f"{label:<14}" + "".join(f"  {int(v):>4}" for v in vals)
            row += f"   {int(vmin):>4}    {int(vmax):>4}"
            print(row)

    if print_stats:
        print(f"\n  Display contrast: vmin = p{lo_pct:.0f}, vmax = p{hi_pct:.0f}  (per panel)")

    # Physical extents in µm
    ext_xy = [0, n_x * vx, 0, n_y * vy]
    ext_zx = [0, n_x * vx, 0, n_z * vz]
    ext_zy = [0, n_y * vy, 0, n_z * vz]

    if figsize is None:
        figsize = (12, 4)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (label, proj), extent, (xlabel, ylabel), (vmin, vmax) in zip(
        axes,
        projections,
        [ext_xy, ext_zx, ext_zy],
        [("X (µm)", "Y (µm)"), ("X (µm)", "Z (µm)"), ("Y (µm)", "Z (µm)")],
        display_limits,
    ):
        ax.imshow(proj, cmap="gray", origin="lower", aspect="auto",
                  extent=extent, interpolation="nearest",
                  vmin=vmin, vmax=vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
