# MERci/plots/view_intensity_stats_plots.py
"""Plotting for ``notebooks/after_imaging/04_view_intensity_stats.ipynb`` (see
:mod:`MERci.analysis.view_intensity_stats` for the data-loading side)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ..analysis.stage_z import positions_to_grid_indices

# Wavelength -> plot color mapping (matches the per-function dict in
# visualization.py's shutter-sequence plot -- not shared as a module-level
# constant there, so duplicated here rather than reaching into that
# function's internals).
_WL_COLOR = {
    405: "#9467bd",
    488: "#1f77b4",
    560: "#ff7f0e",
    650: "#2ca02c",
    750: "#d62728",
}
_DEFAULT_LINE_COLOR = "#7f7f7f"


def plot_z_profiles(stats_df: pd.DataFrame, round_id: int, color_nm: float, bead_z: float = 0.0, bead_color: float = 488.0) -> None:
    """
    Z-profile plot for one bit (round_id x color_nm).
    One line per FOV; all lines share the same color, alpha = 0.5.
    """
    if "z" not in stats_df.columns or "color" not in stats_df.columns:
        print("No z/color information in stats (frame table not found).")
        return

    data = stats_df[
        (stats_df["round_id"] == round_id)
        & (stats_df["color"].round() == round(color_nm))
        & (stats_df["z"] != bead_z)
    ].copy()

    if data.empty:
        print(f"No data for round {round_id}, color {color_nm:.0f} nm.")
        return

    line_color = _WL_COLOR.get(int(round(color_nm)), _DEFAULT_LINE_COLOR)

    fig, ax = plt.subplots(figsize=(7, 4))
    for _, fov_data in data.groupby("fov_id"):
        fov_data = fov_data.sort_values("z")
        ax.plot(fov_data["z"], fov_data["median"],
                color=line_color, alpha=0.5, linewidth=0.9)

    ax.set_xlabel("Z position (µm)")
    ax.set_ylabel("Median pixel intensity")
    ax.set_title(f"Round {round_id}  |  {color_nm:.0f} nm  —  Z-profile  "
                 f"({data['fov_id'].nunique()} FOVs)")
    fig.tight_layout()
    plt.show()


def plot_fov_heatmap(stats_df: pd.DataFrame, meta, round_id: int, color_nm: float, bead_z: float = 0.0, bead_color: float = 488.0) -> None:
    """
    Heatmap of per-FOV median intensity (median over z) for one bit.

    Stage positions are mapped to integer (x_idx, y_idx) grid indices
    (:func:`MERci.analysis.stage_z.positions_to_grid_indices`). The heatmap
    rows correspond to y_idx (increasing downward, matching the stage
    coordinate convention).
    """
    if "z" not in stats_df.columns or "color" not in stats_df.columns:
        print("No z/color information in stats (frame table not found).")
        return

    data = stats_df[
        (stats_df["round_id"] == round_id)
        & (stats_df["color"].round() == round(color_nm))
        & (stats_df["z"] != bead_z)
    ]

    if data.empty:
        print(f"No data for round {round_id}, color {color_nm:.0f} nm.")
        return

    # Per-FOV median across z positions
    fov_medians = data.groupby("fov_id")["median"].median()
    fov_ids = sorted(fov_medians.index.tolist())

    grid = positions_to_grid_indices(fov_ids, meta)
    n_x = max(xi for xi, _ in grid.values()) + 1
    n_y = max(yi for _, yi in grid.values()) + 1

    matrix = np.full((n_y, n_x), np.nan)
    for fov_id, intensity in fov_medians.items():
        xi, yi = grid[fov_id]
        matrix[yi, xi] = intensity

    fig, ax = plt.subplots(figsize=(max(5, n_x * 0.4 + 1.5),
                                    max(4, n_y * 0.4 + 1.5)))
    im = ax.imshow(matrix, cmap="viridis", origin="upper",
                   interpolation="nearest",
                   vmin=np.nanpercentile(matrix, 2),
                   vmax=np.nanpercentile(matrix, 98))
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Median pixel intensity")

    ax.set_title(f"Round {round_id}  |  {color_nm:.0f} nm  —  FOV intensity map  "
                 f"({len(fov_ids)} FOVs)")
    ax.set_xlabel("X grid index  (increasing stage X →)")
    ax.set_ylabel("Y grid index  (increasing stage Y ↓)")

    # Annotate each cell with its FOV id
    if n_x * n_y <= 200:   # only annotate if grid is not too large
        fov_at = {(xi, yi): fov_id for fov_id, (xi, yi) in grid.items()}
        for (xi, yi), fov_id in fov_at.items():
            intensity = matrix[yi, xi]
            txt_color = "white" if intensity < np.nanmedian(matrix) else "black"
            ax.text(xi, yi, str(fov_id), ha="center", va="center",
                    fontsize=6, color=txt_color)

    fig.tight_layout()
    plt.show()
