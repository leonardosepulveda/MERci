# MERci/visualization.py
"""
Unified plotting utilities for both acquisition planning and analysis.

Acquisition
-----------
visualize_shutter_sequence  – bar chart of the per-frame colour sequence

Analysis
--------
plot_fov_layout             – scatter plot of FOV stage positions
plot_stats_over_rounds      – mean signal per round with error bars
plot_spatial_uniformity     – colour-coded scatter for the last round
display_mosaic              – show a saved mosaic PNG in a notebook
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Acquisition: shutter sequence ─────────────────────────────────────────────

def visualize_shutter_sequence(
    frame_table,                   # pd.DataFrame
    title:     Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Draw a vertical bar chart of the per-frame colour/z sequence.

    Each horizontal bar represents one camera frame.  Bars are placed in
    columns corresponding to hardware channels, with blank frames in an
    extra column.  Horizontal separators mark z-group boundaries.
    Z annotations appear to the right.

    Parameters
    ----------
    frame_table : DataFrame with columns ``["color", "channel", "z"]``
    title       : plot title (default generated automatically)
    save_path   : if given, the figure is saved to this path (300 dpi)
    """
    _WAVELENGTH_COLOUR = {
        405.0: "#9467bd",   # purple
        488.0: "#1f77b4",   # blue
        560.0: "#ff7f0e",   # orange
        650.0: "#2ca02c",   # green
        750.0: "#d62728",   # red
    }
    _BLANK_COLOUR  = "black"
    _DEFAULT_COLOUR = "#7f7f7f"
    _BLANK_X       = -1

    df = frame_table.copy().reset_index().rename(columns={"index": "frame"})
    if df.empty:
        log.warning("Frame table is empty – nothing to plot.")
        return

    active_channels = sorted(df["channel"].dropna().unique())
    all_x           = [_BLANK_X] + [int(c) for c in active_channels]

    df = df.sort_values("frame")

    fig, ax = plt.subplots(figsize=(5, 8))

    # ── Draw bars ────────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        frame = int(row["frame"])
        ch    = row["channel"]
        wav   = row["color"]

        if pd.isna(ch):
            x_ctr, face = _BLANK_X, _BLANK_COLOUR
        else:
            x_ctr = int(ch)
            face  = _WAVELENGTH_COLOUR.get(float(wav), _DEFAULT_COLOUR)

        ax.barh(
            y=frame, width=0.8, left=x_ctr - 0.4, height=1.0,
            align="center", color=face, edgecolor="k", linewidth=0.2,
        )

    # ── z-group separators ────────────────────────────────────────────────────
    prev_frame = prev_z = None
    for _, row in df.iterrows():
        f = int(row["frame"])
        z = row["z"]
        if prev_frame is not None and z != prev_z:
            ax.axhline(
                y=(prev_frame + f) / 2.0,
                color="0.7", linestyle="--", linewidth=0.5, zorder=0,
            )
        prev_frame, prev_z = f, z

    # ── Z annotations ─────────────────────────────────────────────────────────
    right_x = max(all_x) + 1.2 if all_x else 0.5
    for z_val, f0 in df.groupby("z", sort=True)["frame"].min().items():
        if pd.isna(z_val):
            continue
        ax.text(right_x, f0, f"z={z_val:g}", fontsize=8, va="center", ha="left")

    # ── Labels & ticks ────────────────────────────────────────────────────────
    ax.set_ylabel("Frame")
    ax.set_xlabel("Channel / blank")
    ax.set_title(title or "Shutter sequence")
    ax.set_xticks(all_x)
    ax.set_xticklabels(["blank"] + [str(int(c)) for c in active_channels])
    ax.set_xlim(min(all_x) - 1, max(all_x) + 2.5)
    ax.set_ylim(df["frame"].min() - 1, df["frame"].max() + 1)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ── Acquisition: FOV layout ───────────────────────────────────────────────────

def plot_fov_layout(
    metadata,                       # ExperimentMetadata
    highlight_fov_ids=None,         # optional list[int]
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Scatter plot of all FOV stage positions.

    Parameters
    ----------
    metadata          : ExperimentMetadata instance
    highlight_fov_ids : optional list of FOV ids to mark in a different colour
    title             : plot title
    save_path         : if given, figure is saved here (300 dpi)
    """
    xs = [v.position[0] for v in metadata.fovs.values()]
    ys = [v.position[1] for v in metadata.fovs.values()]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, s=8, color="steelblue", label="FOV")

    if highlight_fov_ids:
        hxs = [metadata.fovs[i].position[0] for i in highlight_fov_ids
               if i in metadata.fovs]
        hys = [metadata.fovs[i].position[1] for i in highlight_fov_ids
               if i in metadata.fovs]
        ax.scatter(hxs, hys, s=30, color="red", zorder=5, label="highlighted")
        ax.legend(fontsize=8)

    ax.set_xlabel("Stage X (µm)")
    ax.set_ylabel("Stage Y (µm)")
    ax.set_title(title or f"FOV layout  ({metadata.n_fovs} FOVs)")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ── Analysis: round-level mean signal ─────────────────────────────────────────

def plot_stats_over_rounds(
    stats_df:      pd.DataFrame,
    frame_idx:     int = 0,
    title:         Optional[str] = None,
    save_path:     Optional[Path] = None,
) -> None:
    """
    Plot mean ± std of the per-frame signal for each imaging round.

    Parameters
    ----------
    stats_df   : aggregated DataFrame with columns
                 ``["round_id", "frame", "mean", "std", ...]``
    frame_idx  : which frame index to plot
    title      : plot title
    save_path  : if given, figure is saved here (300 dpi)
    """
    frame0 = stats_df[stats_df["frame"] == frame_idx].copy()
    by_round = (
        frame0.groupby("round_id")["mean"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_signal", "std": "std_signal"})
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(
        by_round.index,
        by_round["mean_signal"],
        yerr=by_round["std_signal"],
        fmt="o-", capsize=4,
    )
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Mean intensity (frame {frame_idx})")
    ax.set_title(title or f"Signal per round  (frame {frame_idx})")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_spatial_uniformity(
    stats_df:   pd.DataFrame,
    round_id,
    frame_idx:  int = 0,
    title:      Optional[str] = None,
    save_path:  Optional[Path] = None,
) -> None:
    """
    Colour-coded scatter of mean signal across FOV positions for one round.

    Parameters
    ----------
    stats_df   : aggregated DataFrame with columns
                 ``["round_id", "frame", "mean", "position_x", "position_y"]``
    round_id   : which round to show
    frame_idx  : which frame index to use
    title      : plot title
    save_path  : if given, figure is saved here (300 dpi)
    """
    data = stats_df[
        (stats_df["round_id"] == round_id) & (stats_df["frame"] == frame_idx)
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        data["position_x"], data["position_y"],
        c=data["mean"], cmap="viridis", s=50,
    )
    plt.colorbar(sc, ax=ax, label="Mean intensity")
    ax.set_xlabel("Stage X (µm)")
    ax.set_ylabel("Stage Y (µm)")
    ax.set_title(title or f"Spatial uniformity – round {round_id}")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ── Analysis: mosaic display ──────────────────────────────────────────────────

def display_mosaic(
    mosaic_path: Path,
    round_id:    int,
    width:       int = 700,
) -> None:
    """
    Show a saved mosaic PNG as an inline image in a Jupyter notebook.

    Parameters
    ----------
    mosaic_path : path to the mosaic PNG file
    round_id    : shown in the caption
    width       : display width in pixels
    """
    try:
        from IPython.display import Image as IPImage, display
        print(f"\nRound {round_id:03d} mosaic:")
        display(IPImage(str(mosaic_path), width=width))
    except ImportError:
        log.warning("IPython not available; cannot display mosaic inline.")