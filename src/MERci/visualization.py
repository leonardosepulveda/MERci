# MERci/visualization.py
"""
Unified plotting utilities for both acquisition planning and analysis.

Acquisition
-----------
visualize_shutter_sequence  – frame-vs-z trajectory, markers coloured by laser

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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Acquisition: shutter sequence ─────────────────────────────────────────────

def visualize_shutter_sequence(
    frame_table,                   # pd.DataFrame
    title:     Optional[str] = None,
    save_path: Optional[Path] = None,
    style:     str = "dot",
) -> None:
    """
    Plot the per-frame z trajectory of an imaging round.

    The x-axis is the camera frame number and the y-axis is the z position
    (µm).  A grey baseline traces the objective's z path across frames, and a
    marker at each frame is coloured by the laser acquired at that frame.

    The shape of the trajectory makes the acquisition order obvious: an
    interleaved scan shows a rising staircase (all colours per z before
    stepping), a sequential scan shows full z-sweeps per colour, and a
    progressive z-return shows the descending tail back to ``bead_z``.

    Parameters
    ----------
    frame_table : DataFrame with columns ``["color", "channel", "z"]``
    title       : plot title (default generated automatically)
    save_path   : if given, the figure is saved to this path (300 dpi)
    style       : ``"dot"`` *(default)* — circular markers whose size and the
                  figure width auto-scale with the number of frames so dense
                  rounds do not overlap (blank frames drawn hollow); or
                  ``"line"`` — a thin vertical tick of the laser colour centred
                  on each ``(frame, z)`` point, which stays legible at any
                  frame count (blank frames drawn in grey).
    """
    if style not in ("dot", "line"):
        raise ValueError(f"Unknown style {style!r}. Use 'dot' or 'line'.")

    _WAVELENGTH_COLOUR = {
        405.0: "#9467bd",   # purple
        488.0: "#1f77b4",   # blue
        560.0: "#ff7f0e",   # orange
        650.0: "#2ca02c",   # green
        750.0: "#d62728",   # red
    }
    _BLANK_FACE     = "white"
    _BLANK_EDGE     = "0.6"
    _DEFAULT_COLOUR = "#7f7f7f"

    df = frame_table.copy().reset_index().rename(columns={"index": "frame"})
    if df.empty:
        log.warning("Frame table is empty – nothing to plot.")
        return

    df = df.sort_values("frame")
    frames = df["frame"].to_numpy()
    z_vals = df["z"].astype(float).to_numpy()
    n      = len(df)

    # Per-frame laser colour (None for blank frames).
    colours = [
        None if pd.isna(ch) else _WAVELENGTH_COLOUR.get(float(wav), _DEFAULT_COLOUR)
        for wav, ch in zip(df["color"], df["channel"])
    ]

    # Scale the figure width and marker size with the number of frames so dense
    # rounds (hundreds of frames) stay readable instead of overlapping.
    fig_w     = float(np.clip(0.09 * n, 12.0, 22.0))
    marker_s  = float(np.clip(4000.0 / max(n, 1), 5.0, 50.0))

    fig, ax = plt.subplots(figsize=(fig_w, 4))

    # Grey baseline tracing the z path across frames.
    ax.plot(frames, z_vals, color="grey", alpha=0.5, linewidth=1.0, zorder=1)

    if style == "dot":
        faces = [c if c is not None else _BLANK_FACE for c in colours]
        edges = [c if c is not None else _BLANK_EDGE for c in colours]
        ax.scatter(
            frames, z_vals,
            facecolors=faces, edgecolors=edges,
            s=marker_s, linewidths=0.6, zorder=2,
        )
    else:  # "line": thin vertical tick centred on each point
        line_cols = [c if c is not None else _BLANK_EDGE for c in colours]
        ax.scatter(
            frames, z_vals,
            marker="|", c=line_cols, s=120, linewidths=1.3, zorder=2,
        )

    # Legend: one entry per laser present, plus blank if any.
    present = sorted({float(w) for w, c in zip(df["color"], df["channel"])
                      if not pd.isna(c)})
    handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=_WAVELENGTH_COLOUR.get(w, _DEFAULT_COLOUR),
               markeredgecolor=_WAVELENGTH_COLOUR.get(w, _DEFAULT_COLOUR),
               markersize=7, label=f"{int(w)} nm")
        for w in present
    ]
    if df["channel"].isna().any():
        handles.append(
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=_BLANK_FACE, markeredgecolor=_BLANK_EDGE,
                   markersize=7, label="blank")
        )
    if handles:
        ax.legend(handles=handles, title="Laser", fontsize=8, loc="best")

    ax.set_xlabel("Frame")
    ax.set_ylabel("z position (µm)")
    ax.set_title(title or "Shutter sequence")
    ax.margins(x=0.01)

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