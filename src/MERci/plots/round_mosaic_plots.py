# MERci/plots/round_mosaic_plots.py
"""Plotting for ``notebooks/during_imaging/round_mosaics.ipynb`` (see
:mod:`MERci.live_round_mosaic` for the logic that builds the canvases)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image


def _downsample_for_preview(canvas: np.ndarray, max_px: int) -> np.ndarray:
    """
    Long side capped at *max_px* -- PIL's own resize, confirmed directly to
    cost well under 0.1s even for a real ~10600x6700 px canvas (vs. ~6s for
    matplotlib to ``imshow``+draw that array at full resolution). Returns
    the canvas unchanged if it's already small enough (e.g. early in a
    round, before many FOVs are placed).
    """
    scale = min(1.0, max_px / max(canvas.shape))
    if scale >= 1.0:
        return canvas
    new_w, new_h = max(1, int(canvas.shape[1] * scale)), max(1, int(canvas.shape[0] * scale))
    return np.array(Image.fromarray(canvas).resize((new_w, new_h)))


def show_round_mosaic(
    round_id: int,
    canvases: Dict[float, np.ndarray],
    label,
    mosaic_paths: Dict[float, Path],
    live_preview_max_px: int,
    save_full_res: bool = True,
) -> None:
    """
    Redraw the on-screen figure from a DOWNSAMPLED preview
    (:func:`_downsample_for_preview` -- the full-resolution array is
    expensive for matplotlib to draw at real production scale, confirmed
    directly, which would otherwise dominate wall-clock time and make
    per-tile "live" redraws arrive only every several seconds).

    *save_full_res* controls whether every color's current FULL-RESOLUTION
    canvas also gets (re)written to *mosaic_paths* this call -- also
    expensive at real scale (confirmed directly), so the caller
    (``LiveRoundMosaicBuilder.build_round_mosaic``'s ``maybe_redraw``)
    throttles this far more coarsely than the cheap on-screen preview.
    """
    colors = sorted(canvases)
    fig, axes = plt.subplots(1, max(len(colors), 1), figsize=(6 * max(len(colors), 1), 6), squeeze=False)
    for ax, color_nm in zip(axes[0], colors):
        ax.imshow(_downsample_for_preview(canvases[color_nm], live_preview_max_px), cmap="gray")
        ax.set_title(f"round {label} — {color_nm:.0f} nm")
        ax.axis("off")
    fig.tight_layout()
    clear_output(wait=True)
    display(fig)
    plt.close(fig)
    if save_full_res:
        for color_nm, canvas in canvases.items():
            Image.fromarray(canvas).save(str(mosaic_paths[color_nm]))
