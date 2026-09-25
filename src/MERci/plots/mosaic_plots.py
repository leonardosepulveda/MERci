# MERci/plots/mosaic_plots.py
"""Plots for the Steve-mosaic tissue-boundary workflow (``acquisition.mosaic``)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

from ..acquisition.mosaic import (
    MosaicCanvas, MosaicSegmentation, SteveTile, _classify_tiles_by_signal,
    assemble_mosaic_canvas, estimate_bimodal_threshold, normalize_tiles_for_display,
)


def plot_tile_intensity_histograms(
    tiles:           List[SteveTile],
    bins:            int = 200,
    ax=None,
    color:           tuple = (0.7, 0.7, 0.7),
    alpha:           float = 0.5,
    show_threshold:  bool = True,
) -> Tuple[object, Optional[float]]:
    """
    Overlay one log-intensity histogram per tile (thin lines) and a combined
    histogram (solid), to spot outlier tiles and pick a segmentation
    threshold by eye.

    The combined histogram weights "empty" and "signal" tiles 50/50
    (:func:`_classify_tiles_by_signal`) instead of pooling pixels: when most
    tiles are empty, pooling shrinks the tissue peak below the 5% prominence
    :func:`estimate_bimodal_threshold` needs, and no threshold is found.

    All histograms share the same ``log10`` bin edges and are density-
    normalized, so their shapes compare directly. If the combined histogram is
    bimodal, the valley between its two main peaks
    (:func:`estimate_bimodal_threshold`) is drawn and returned, for use as
    ``THRESHOLD`` instead of Otsu.

    Parameters
    ----------
    tiles : from :func:`load_steve_mosaic` (or a subset)
    bins : number of bins over the full log10(intensity) range
    ax : optional matplotlib Axes to draw into
    color, alpha : line style for the per-tile histograms
    show_threshold : draw the estimated threshold as a labelled vertical line

    Returns
    -------
    (ax, threshold) : the Axes, and the threshold in linear intensity (``None``
        if the histogram is not clearly bimodal).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Clip at 1 (not 0) so log10 stays finite for zero/saturated-low pixels.
    log_images = [np.log10(np.clip(t.image, 1, None).astype(np.float64)) for t in tiles]
    lo = min(float(li.min()) for li in log_images)
    hi = max(float(li.max()) for li in log_images)
    bin_edges = np.linspace(lo, hi, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for li in log_images:
        counts, _ = np.histogram(li, bins=bin_edges, density=True)
        ax.plot(bin_centers, counts, "-", color=color, alpha=alpha, lw=1.0)

    signal_mask = _classify_tiles_by_signal(log_images)
    if signal_mask is not None:
        empty_pixels = np.concatenate(
            [log_images[i].ravel() for i in range(len(tiles)) if not signal_mask[i]])
        signal_pixels = np.concatenate(
            [log_images[i].ravel() for i in range(len(tiles)) if signal_mask[i]])
        empty_counts, _ = np.histogram(empty_pixels, bins=bin_edges, density=True)
        signal_counts, _ = np.histogram(signal_pixels, bins=bin_edges, density=True)
        combined_counts = 0.5 * empty_counts + 0.5 * signal_counts
        combined_label = (f"combined (balanced: {int((~signal_mask).sum())} empty / "
                           f"{int(signal_mask.sum())} signal tile(s))")
    else:
        all_pixels = np.concatenate([li.ravel() for li in log_images])
        combined_counts, _ = np.histogram(all_pixels, bins=bin_edges, density=True)
        combined_label = "all tiles combined"
    ax.plot(bin_centers, combined_counts, "-", color="black", lw=1.8, label=combined_label)

    threshold = estimate_bimodal_threshold(bin_centers, combined_counts)
    if show_threshold and threshold is not None:
        log_threshold = np.log10(threshold)
        ax.axvline(log_threshold, color="crimson", linestyle="--", lw=1.5,
                   label=f"estimated threshold = {threshold:.0f}")
        ymax = ax.get_ylim()[1]
        ax.text(log_threshold, ymax * 0.97, f"  {threshold:.0f}",
                color="crimson", va="top", ha="left")

    ax.set_xlabel("log10(pixel intensity)")
    ax.set_ylabel("density")
    ax.set_title(f"Per-tile intensity histograms ({len(tiles)} tile(s))")
    ax.legend(loc="upper right", fontsize=8)
    return ax, threshold


def plot_mosaic_segmentation(canvas: MosaicCanvas, segmentation: MosaicSegmentation, ax=None):
    """
    Overlay detected tissue (green) / hole (red) polygons on the mosaic canvas,
    for the notebook's interactive threshold-tuning review step.

    Parameters
    ----------
    canvas : from :func:`assemble_mosaic_canvas`.
    segmentation : from :func:`segment_mosaic_tissue`.
    ax : optional existing matplotlib Axes to draw into (creates one if omitted).

    Returns
    -------
    The matplotlib Axes drawn into.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))

    covered_vals = canvas.image[canvas.covered]
    ax.imshow(
        canvas.image, cmap="gray",
        vmin=np.percentile(covered_vals, 1), vmax=np.percentile(covered_vals, 99),
    )

    def _plot_ring(ring, color: str, lw: float, linestyle: str = "-"):
        xs, ys = ring.xy
        cols = [(x - canvas.origin_um[0]) / canvas.pixel_size_um for x in xs]
        rows = [(y - canvas.origin_um[1]) / canvas.pixel_size_um for y in ys]
        ax.plot(cols, rows, linestyle, color=color, lw=lw)

    def _plot_poly(poly: Polygon, color: str):
        if poly.geom_type != "Polygon":
            return
        _plot_ring(poly.exterior, color, lw=1.2)
        # Interior rings = islands inside a hole (a true donut/annulus) --
        # drawn dashed in the same color so they read as "carved out of the
        # hole, still imaged" rather than another hole of their own.
        for interior in poly.interiors:
            _plot_ring(interior, color, lw=1.0, linestyle="--")

    for poly in segmentation.tissue_polygons:
        _plot_poly(poly, "lime")
    for poly in segmentation.hole_polygons:
        _plot_poly(poly, "red")

    ax.set_title(
        f"{len(segmentation.tissue_polygons)} tissue piece(s), "
        f"{len(segmentation.hole_polygons)} hole(s)  (threshold={segmentation.threshold:.0f})"
    )
    return ax


def plot_objective_alignment_check(
    tiles_all:         List[SteveTile],
    verify_objectives: Optional[List[str]] = None,
    working_pixel_um:  float = 5.0,
    low_percentile:    float = 1.0,
    high_percentile:   float = 99.0,
    ax=None,
):
    """
    Composite two (or more) objectives together, independently normalized
    for display, as a visual corroboration that :func:`load_steve_mosaic`'s
    per-objective ``.msc``-recorded ``(x_offset, y_offset)`` calibration
    shift is actually landing the objectives' real tissue content in the
    same place -- not just a numerically-claimed correction. Typical use:
    a low-mag scan plus a handful of high-mag alignment/reference tiles
    deliberately overlapping it -- if the shift is correct, the two
    objectives' shared tissue features line up continuously across the
    overlap; if not, there is a visible discontinuity/offset.

    This is a DIAGNOSTIC composite only, never the canvas used for real
    tissue segmentation (:func:`segment_mosaic_tissue` needs
    :func:`assemble_mosaic_canvas`'s real-intensity canvas).

    Parameters
    ----------
    tiles_all : every tile from :func:`load_steve_mosaic` (all objectives)
        -- already includes each tile's real, ``.msc``-corrected stage
        position, so no additional shift needs to be applied here.
    verify_objectives : which ``objective_name``(s) to composite; ``None``
        (default) composites every objective present in *tiles_all*.
    working_pixel_um : forwarded to :func:`assemble_mosaic_canvas`.
    low_percentile, high_percentile : forwarded to
        :func:`normalize_tiles_for_display`.
    ax : existing matplotlib Axes to draw into; ``None`` creates a new figure.

    Returns
    -------
    (ax, canvas) : the Axes drawn into, and the assembled (display-
    normalized) :class:`MosaicCanvas` -- e.g. to crop/zoom further on a
    specific overlap region.
    """
    import matplotlib.pyplot as plt

    tiles = tiles_all if verify_objectives is None else [
        t for t in tiles_all if t.objective_name in verify_objectives
    ]
    if not tiles:
        raise ValueError(f"No tiles match verify_objectives={verify_objectives!r}")

    normalized = normalize_tiles_for_display(tiles, low_percentile, high_percentile)
    canvas = assemble_mosaic_canvas(normalized, working_pixel_um=working_pixel_um)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas.image, cmap="gray", vmin=0.0, vmax=1.0)
    objectives_present = sorted({t.objective_name for t in tiles})
    ax.set_title(
        f"Objective alignment check: {', '.join(objectives_present)} "
        "(each independently normalized for display)", fontsize=11,
    )
    return ax, canvas
