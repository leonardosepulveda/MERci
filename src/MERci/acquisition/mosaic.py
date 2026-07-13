# MERci/acquisition/mosaic.py
"""
Derive tissue-boundary / hole polygons automatically from a Steve low-mag
mosaic, instead of drawing ``boundary_positions*.txt``/``hole*.txt`` by hand.

Typical workflow (see ``02_create_positions_from_mosaic.ipynb``)
------------------------------------------------------------------
1. ``load_steve_mosaic`` – read a Steve ``.msc`` manifest + its ``.stv`` tile
   pickles (each tile already carries its own stage position and pixel size,
   so no separate calibration step is needed).
2. ``filter_tiles_by_objective`` – drop tiles shot with a different
   objective than the main scan (e.g. high-mag alignment/reference FOVs
   mixed into a low-mag mosaic) -- required if the mosaic mixes objectives,
   since ``assemble_mosaic_canvas`` refuses to paste mismatched pixel sizes
   into one canvas.
3. ``assemble_mosaic_canvas`` – paste all (same-objective) tiles into one
   flattened image in real stage-micron coordinates.
4. ``plot_tile_intensity_histograms`` – overlay every tile's log-space
   intensity histogram, to pick a fixed segmentation threshold by eye when
   Otsu doesn't separate tissue from background well on a given sample.
5. ``segment_mosaic_tissue`` – threshold + clean up the canvas into tissue
   and hole polygons, in the same ``x_um, y_um`` coordinate space that
   ``boundary_positions.txt``/``hole*.txt`` already use.
6. ``plot_mosaic_segmentation`` – overlay the detected polygons on the
   canvas for a visual sanity check before committing to them (thresholding
   parameters are re-run interactively until the overlay looks right).
7. ``save_boundary_from_mosaic`` – write the polygons out in the exact
   filename convention ``positions.discover_boundary_files``/
   ``load_hole_polygons`` already expect, so
   ``02_create_positions_from_boundaries.ipynb`` picks them up unchanged.

Steve file formats (reverse-engineered from ``storm_control.steve``, not
otherwise documented)
------------------------------------------------------------------------
* ``<name>.msc`` – plain text, one comma-separated record per line. An
  ``objective,<name>,<um_per_pix>,<x_offset>,<y_offset>`` line per configured
  objective (informational only -- not used here, see below) and an
  ``image,<filename>`` line per saved tile.
* ``<name>_<id>.stv`` – a ``pickle.dump`` of the tile's ``ImageItem.__dict__``
  (minus its Qt graphics item). The keys used here: ``numpy_data`` (the raw,
  already-oriented camera frame), ``x_um``/``y_um`` (stage position of the
  frame's *center*), and ``magnification``. The real per-tile pixel size is
  derived from the tile's own ``x_um``/``x_pix`` ratio and ``magnification``
  rather than trusting the ``.msc`` objective line (which is rounded to 2
  decimal places for display) or a hard-coded ``storm_control.steve.coord
  .Point.pixels_to_um`` value (which is a mutable class attribute, not a
  universal constant).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage import filters, measure, morphology

from MERci.common.io import save_positions_array

_MSC_IMAGE_PREFIX = "image,"


@dataclass
class SteveTile:
    """One Steve mosaic tile, in real stage-micron units.

    Attributes
    ----------
    image : np.ndarray
        Raw camera frame (2D, as saved by Steve -- already flip/transpose
        oriented to match the stage axes).
    x_um, y_um : float
        Stage position of the frame's center.
    pixel_size_um : float
        Real-world size of one ``image`` pixel, derived from the tile's own
        ``magnification``/``x_pix`` fields (not assumed from the objective
        name).
    objective_name : str
        The objective this tile was acquired with (e.g. ``"10x"``). A mosaic
        can contain tiles shot with more than one objective -- e.g. a few
        alignment/reference FOVs taken at high magnification alongside the
        low-mag scan -- see :func:`filter_tiles_by_objective`.
    """
    image:          np.ndarray
    x_um:           float
    y_um:           float
    pixel_size_um:  float
    objective_name: str


@dataclass
class MosaicCanvas:
    """A Steve mosaic flattened into one image, in stage-micron coordinates.

    Attributes
    ----------
    image : np.ndarray
        Mean-pooled intensity per canvas pixel (0 where no tile covers it).
    covered : np.ndarray
        Boolean mask, ``True`` where at least one tile contributed.
    origin_um : (float, float)
        Stage position (x_um, y_um) of the canvas's ``[0, 0]`` pixel corner.
    pixel_size_um : float
        Real-world size of one canvas pixel (after any working-resolution
        downsampling -- see ``assemble_mosaic_canvas``).
    """
    image:         np.ndarray
    covered:       np.ndarray
    origin_um:     Tuple[float, float]
    pixel_size_um: float

    def to_um(self, row: float, col: float) -> Tuple[float, float]:
        """Convert a (row, col) canvas-pixel coordinate to (x_um, y_um)."""
        x0, y0 = self.origin_um
        return (x0 + col * self.pixel_size_um, y0 + row * self.pixel_size_um)


@dataclass
class MosaicSegmentation:
    """Tissue/hole polygons detected in a :class:`MosaicCanvas`, plus the
    intermediate mask and threshold used -- kept around so
    :func:`plot_mosaic_segmentation` can show exactly what was thresholded.
    """
    tissue_polygons: List[Polygon]
    hole_polygons:   List[Polygon]
    mask:            np.ndarray
    threshold:       float


def load_steve_mosaic(msc_path: Path) -> List[SteveTile]:
    """
    Load every tile referenced by a Steve ``.msc`` mosaic manifest.

    Parameters
    ----------
    msc_path : path to the ``<name>.msc`` manifest (tiles are expected as
        sibling ``.stv`` files in the same directory, as Steve saves them).

    Returns
    -------
    List of :class:`SteveTile`, in the order listed in the manifest.
    """
    msc_path = Path(msc_path)
    directory = msc_path.parent

    tile_files = []
    with msc_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(_MSC_IMAGE_PREFIX):
                tile_files.append(line[len(_MSC_IMAGE_PREFIX):])

    tiles = []
    for fname in tile_files:
        with open(directory / fname, "rb") as fh:
            d = pickle.load(fh)

        # pixels_to_um is a mutable class attribute in storm_control, not a
        # fixed constant -- recover it from this tile's own x_um/x_pix ratio
        # instead of assuming a value.
        pixels_to_um = d["x_um"] / d["x_pix"] if d["x_pix"] else 1.0
        pixel_size_um = pixels_to_um / d["magnification"]

        tiles.append(SteveTile(
            image=d["numpy_data"],
            x_um=d["x_um"],
            y_um=d["y_um"],
            pixel_size_um=pixel_size_um,
            objective_name=d["objective_name"],
        ))

    return tiles


def filter_tiles_by_objective(
    tiles:      List[SteveTile],
    objective:  Optional[str] = None,
) -> List[SteveTile]:
    """
    Keep only the tiles shot with one objective, dropping the rest.

    A Steve mosaic can mix in a handful of tiles shot at a different
    objective than the main low-mag scan -- e.g. alignment/reference FOVs
    used to register a 60x objective against the 10x mosaic. Those tiles
    have a different real pixel size and must not be pasted into the same
    flattened canvas as the rest (see :func:`assemble_mosaic_canvas`, which
    raises rather than silently mixing scales).

    Parameters
    ----------
    tiles : from :func:`load_steve_mosaic`.
    objective : which objective's tiles to keep; ``None`` = auto-pick
        whichever objective the most tiles share (prints nothing itself --
        the caller should log the counts/decision; see the notebook for the
        printed breakdown this is paired with).

    Returns
    -------
    The filtered tile list (all sharing one ``objective_name``).
    """
    if not tiles:
        raise ValueError("No tiles to filter.")

    if objective is None:
        counts: dict = {}
        for t in tiles:
            counts[t.objective_name] = counts.get(t.objective_name, 0) + 1
        objective = max(counts, key=counts.get)

    return [t for t in tiles if t.objective_name == objective]


def assemble_mosaic_canvas(
    tiles:           List[SteveTile],
    working_pixel_um: float = 5.0,
) -> MosaicCanvas:
    """
    Paste every tile into one flattened image in stage-micron coordinates.

    Tiles are assumed non-overlapping (a raster-scanned Steve mosaic); where
    tiles do overlap slightly, overlapping pixels are averaged rather than
    the last tile winning, so a seam is never fully opaque either way.

    Parameters
    ----------
    tiles : tiles from :func:`load_steve_mosaic`, all sharing one objective
        (hence one real pixel size) -- see :func:`filter_tiles_by_objective`
        if the mosaic mixes objectives.
    working_pixel_um : target canvas pixel size, in microns. Tiles are
        downsampled (by the nearest integer factor to this target) before
        pasting, since full camera resolution (e.g. 2304x2304 per tile) is
        unnecessary for tissue-scale thresholding and would make the full
        mosaic canvas very large. 5 um/pixel keeps sub-mm tissue features
        while keeping a multi-mm mosaic's canvas a few thousand pixels across.

    Raises
    ------
    ValueError
        If the tiles don't all share the same pixel size (within 1%) --
        e.g. a mosaic that mixes a low-mag scan with a few high-mag
        alignment FOVs. Pasting mismatched-scale tiles into one canvas using
        a single pixel size would silently misplace/mis-scale whichever
        tiles don't match; filter to one objective first instead of
        overriding this check.

    Returns
    -------
    MosaicCanvas
    """
    if not tiles:
        raise ValueError("No tiles to assemble.")

    pixel_sizes = {round(t.pixel_size_um, 4) for t in tiles}
    if len(pixel_sizes) > 1:
        by_objective: dict = {}
        for t in tiles:
            by_objective.setdefault(t.objective_name, []).append(t.pixel_size_um)
        breakdown = ", ".join(
            f"{obj!r}: {len(sizes)} tile(s) @ {sizes[0]:.4f} um/px"
            for obj, sizes in by_objective.items()
        )
        raise ValueError(
            f"Tiles have {len(pixel_sizes)} different pixel sizes -- this mosaic "
            f"mixes objectives ({breakdown}). Filter to one objective first, e.g. "
            f"`tiles = filter_tiles_by_objective(tiles, objective='10x')`."
        )

    tile_pixel_um = tiles[0].pixel_size_um
    downsample = max(1, round(working_pixel_um / tile_pixel_um))
    canvas_pixel_um = tile_pixel_um * downsample

    tile_h, tile_w = tiles[0].image.shape
    half_w_um = tile_w * tile_pixel_um / 2
    half_h_um = tile_h * tile_pixel_um / 2

    x_min = min(t.x_um for t in tiles) - half_w_um
    x_max = max(t.x_um for t in tiles) + half_w_um
    y_min = min(t.y_um for t in tiles) - half_h_um
    y_max = max(t.y_um for t in tiles) + half_h_um

    canvas_w = int(np.ceil((x_max - x_min) / canvas_pixel_um)) + 2
    canvas_h = int(np.ceil((y_max - y_min) / canvas_pixel_um)) + 2

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for t in tiles:
        img = t.image[::downsample, ::downsample].astype(np.float32)
        h, w = img.shape
        x0_um = t.x_um - (tile_w * tile_pixel_um) / 2
        y0_um = t.y_um - (tile_h * tile_pixel_um) / 2
        col0 = int(round((x0_um - x_min) / canvas_pixel_um))
        row0 = int(round((y0_um - y_min) / canvas_pixel_um))
        row1, col1 = row0 + h, col0 + w
        # Clip in case rounding pushes a tile fractionally outside the canvas.
        row0c, col0c = max(row0, 0), max(col0, 0)
        row1c, col1c = min(row1, canvas_h), min(col1, canvas_w)
        img = img[row0c - row0: row0c - row0 + (row1c - row0c),
                  col0c - col0: col0c - col0 + (col1c - col0c)]
        canvas[row0c:row1c, col0c:col1c] += img
        weight[row0c:row1c, col0c:col1c] += 1

    covered = weight > 0
    canvas[covered] /= weight[covered]

    return MosaicCanvas(
        image=canvas, covered=covered,
        origin_um=(x_min, y_min), pixel_size_um=canvas_pixel_um,
    )


def segment_mosaic_tissue(
    canvas:              MosaicCanvas,
    threshold:           Optional[float] = None,
    smooth_sigma_um:     float = 10.0,
    close_radius_um:     float = 50.0,
    open_radius_um:      float = 15.0,
    margin_um:           float = 75.0,
    min_tissue_area_um2: float = 1000.0,
    min_hole_area_um2:   float = 500.0,
    simplify_tol_um:     float = 15.0,
) -> MosaicSegmentation:
    """
    Threshold a mosaic canvas into tissue and hole polygons.

    Pipeline (each step's purpose, since a single global threshold on the
    raw canvas is too noisy on real Steve mosaics -- illumination
    vignetting and tile seams otherwise fragment one tissue mass into
    hundreds of tiny disjoint specks):

    1. Gaussian-smooth the canvas (``smooth_sigma_um``) to suppress
       per-pixel/vignetting noise before thresholding.
    2. Otsu-threshold the smoothed canvas (or use ``threshold`` if given).
    3. Morphological closing (``close_radius_um``) bridges small real gaps
       between adjacent bits of the same tissue piece.
    4. Morphological opening (``open_radius_um``) removes small noise specks
       that closing alone would keep.
    5. Dilate outward by ``margin_um`` -- mimics the safety margin a person
       drawing a boundary by hand would naturally include, so the FOV grid
       doesn't just barely clip the true tissue edge.
    6. Fill enclosed background regions to find the tissue's own holes,
       label connected components of both the tissue and the holes, and
       trace each labelled region's contour with marching squares
       (``skimage.measure.find_contours``), converting canvas-pixel
       coordinates to stage microns via ``canvas.to_um``.
    7. Drop components below ``min_tissue_area_um2``/``min_hole_area_um2``
       and simplify each polygon by ``simplify_tol_um`` (marching squares
       otherwise produces one vertex per canvas pixel of perimeter).

    Parameters
    ----------
    canvas : from :func:`assemble_mosaic_canvas`.
    threshold : intensity threshold; ``None`` = Otsu on the smoothed canvas.
    smooth_sigma_um, close_radius_um, open_radius_um, margin_um :
        morphology parameters, in microns (converted to canvas pixels
        internally via ``canvas.pixel_size_um``).
    min_tissue_area_um2, min_hole_area_um2 : drop components smaller than
        this (um^2) -- filters residual noise specks after morphology.
    simplify_tol_um : Shapely ``simplify`` tolerance, in microns.

    Returns
    -------
    MosaicSegmentation
    """
    px = canvas.pixel_size_um
    smoothed = filters.gaussian(canvas.image, sigma=smooth_sigma_um / px, preserve_range=True)

    if threshold is None:
        threshold = float(filters.threshold_otsu(smoothed[canvas.covered]))
    mask = (smoothed > threshold) & canvas.covered

    close_radius_px = max(1, round(close_radius_um / px))
    open_radius_px = max(1, round(open_radius_um / px))
    margin_px = max(0, round(margin_um / px))

    closed = morphology.closing(mask, morphology.disk(close_radius_px))
    opened = morphology.opening(closed, morphology.disk(open_radius_px))
    mask_final = morphology.dilation(opened, morphology.disk(margin_px)) if margin_px else opened

    filled = ndimage.binary_fill_holes(mask_final)
    holes_mask = filled & ~mask_final

    tissue_labels, n_tissue = ndimage.label(filled)
    hole_labels, n_holes = ndimage.label(holes_mask)

    def _contour_polygon(label_img: np.ndarray, label_id: int, min_area_px: float) -> Optional[Polygon]:
        component = (label_img == label_id)
        if component.sum() < min_area_px:
            return None
        # Pad so a region touching the canvas edge still yields a closed contour.
        padded = np.pad(component, 1, mode="constant", constant_values=False)
        contours = measure.find_contours(padded.astype(float), 0.5)
        if not contours:
            return None
        contour = max(contours, key=len) - 1  # undo the padding offset
        xy = np.array([canvas.to_um(r, c) for r, c in contour])
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if (not poly.is_empty and poly.area > 0) else None

    min_tissue_area_px = min_tissue_area_um2 / (px * px)
    min_hole_area_px = min_hole_area_um2 / (px * px)

    tissue_polygons = [
        p.simplify(simplify_tol_um) for lid in range(1, n_tissue + 1)
        if (p := _contour_polygon(tissue_labels, lid, min_tissue_area_px)) is not None
    ]
    hole_polygons = [
        p.simplify(simplify_tol_um) for lid in range(1, n_holes + 1)
        if (p := _contour_polygon(hole_labels, lid, min_hole_area_px)) is not None
    ]

    return MosaicSegmentation(
        tissue_polygons=tissue_polygons, hole_polygons=hole_polygons,
        mask=mask_final, threshold=threshold,
    )


def _estimate_bimodal_threshold(bin_centers_log: np.ndarray, counts: np.ndarray) -> Optional[float]:
    """
    Estimate a separating threshold between two modes of a (log-space)
    density histogram, as the valley between its two most prominent peaks.

    Returns the threshold in **linear** intensity units (``10 **
    valley_log10``), or ``None`` if fewer than two prominent peaks are found
    (e.g. a genuinely unimodal sample) -- callers should fall back to Otsu
    in that case rather than plot a misleading line.
    """
    from scipy.signal import find_peaks

    peaks, props = find_peaks(counts, prominence=counts.max() * 0.05)
    if len(peaks) < 2:
        return None

    top2 = sorted(peaks[np.argsort(props["prominences"])[::-1][:2]])
    lo_idx, hi_idx = top2
    valley_idx = lo_idx + int(np.argmin(counts[lo_idx:hi_idx + 1]))
    return float(10 ** bin_centers_log[valley_idx])


def plot_tile_intensity_histograms(
    tiles:           List[SteveTile],
    bins:            int = 200,
    ax=None,
    color:           tuple = (0.7, 0.7, 0.7),
    alpha:           float = 0.25,
    show_threshold:  bool = True,
) -> Tuple[object, Optional[float]]:
    """
    Overlay one log-space pixel-intensity histogram per tile (thin gray
    lines), plus a solid combined histogram pooling every tile's pixels
    together -- lets an outlier tile (a different objective, a debris/bubble
    FOV, ...) stand out, and helps pick a fixed segmentation threshold by eye
    instead of trusting Otsu blindly.

    Every histogram (per-tile and combined) is computed over the same
    ``log10`` bin edges (spanning the full range across all tiles) so the
    overlaid shapes are directly comparable, and all are density-normalized
    so tiles don't need to be the same pixel count to compare shapes.

    When the combined histogram is clearly bimodal, the valley between its
    two most prominent peaks is estimated (:func:`_estimate_bimodal_threshold`),
    drawn as a vertical line labelled with the threshold in linear intensity
    units, and returned -- so it can be used directly as ``THRESHOLD`` in the
    segmentation cell instead of Otsu's often-biased pick (see
    :func:`segment_mosaic_tissue`'s docstring for why Otsu can be biased when
    one class vastly outnumbers the other in pixel count).

    Parameters
    ----------
    tiles : from :func:`load_steve_mosaic` (or a filtered subset).
    bins : number of bins across the full log10(intensity) range.
    ax : optional existing matplotlib Axes to draw into.
    color, alpha : shared line style for every tile's (thin) histogram.
    show_threshold : draw the estimated valley threshold as a vertical line
        with a text label, if a clearly bimodal shape is found.

    Returns
    -------
    (ax, threshold) : the matplotlib Axes drawn into, and the estimated
        linear-space threshold (``None`` if no clearly bimodal shape found).
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

    all_pixels = np.concatenate([li.ravel() for li in log_images])
    combined_counts, _ = np.histogram(all_pixels, bins=bin_edges, density=True)
    ax.plot(bin_centers, combined_counts, "-", color="black", lw=1.8,
            label="all tiles combined")

    threshold = _estimate_bimodal_threshold(bin_centers, combined_counts)
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

    def _plot_poly(poly: Polygon, color: str):
        if poly.geom_type != "Polygon":
            return
        xs, ys = poly.exterior.xy
        cols = [(x - canvas.origin_um[0]) / canvas.pixel_size_um for x in xs]
        rows = [(y - canvas.origin_um[1]) / canvas.pixel_size_um for y in ys]
        ax.plot(cols, rows, "-", color=color, lw=1.2)

    for poly in segmentation.tissue_polygons:
        _plot_poly(poly, "lime")
    for poly in segmentation.hole_polygons:
        _plot_poly(poly, "red")

    ax.set_title(
        f"{len(segmentation.tissue_polygons)} tissue piece(s), "
        f"{len(segmentation.hole_polygons)} hole(s)  (threshold={segmentation.threshold:.0f})"
    )
    return ax


def save_boundary_from_mosaic(segmentation: MosaicSegmentation, positions_dir: Path) -> List[str]:
    """
    Write ``segmentation``'s polygons as ``boundary_positions*.txt``/``hole*.txt``,
    in the exact convention :func:`MERci.acquisition.positions.discover_boundary_files`
    and :func:`MERci.acquisition.positions.load_hole_polygons` expect -- so
    ``02_create_positions_from_boundaries.ipynb`` picks them up unchanged.

    A single detected tissue polygon is written as the legacy
    ``boundary_positions.txt``; several disjoint tissue polygons (e.g. genuinely
    separate tissue fragments) are written as ``boundary_positions_{b}.txt``
    (the "single" layout -- one tissue, several boundary pieces), ordered
    left-to-right then top-to-bottom by centroid so the resulting boundary
    numbering reads in a stable, predictable order.

    Holes are global in the existing pipeline (applied to every boundary
    alike), so every detected hole is written out regardless of which tissue
    polygon it sits inside.

    Parameters
    ----------
    segmentation : from :func:`segment_mosaic_tissue`.
    positions_dir : directory to write into (typically ``SAMPLE_DIR/positions``).

    Returns
    -------
    List of filenames written.
    """
    positions_dir = Path(positions_dir)
    positions_dir.mkdir(parents=True, exist_ok=True)

    if not segmentation.tissue_polygons:
        raise ValueError("No tissue polygons in this segmentation -- nothing to write.")

    tissue_sorted = sorted(
        segmentation.tissue_polygons,
        key=lambda p: (p.centroid.x, p.centroid.y),
    )

    written = []
    if len(tissue_sorted) == 1:
        fname = "boundary_positions.txt"
        save_positions_array(np.array(tissue_sorted[0].exterior.coords), positions_dir / fname)
        written.append(fname)
    else:
        for b, poly in enumerate(tissue_sorted, start=1):
            fname = f"boundary_positions_{b}.txt"
            save_positions_array(np.array(poly.exterior.coords), positions_dir / fname)
            written.append(fname)

    for n, poly in enumerate(segmentation.hole_polygons, start=1):
        fname = f"hole{n}.txt"
        save_positions_array(np.array(poly.exterior.coords), positions_dir / fname)
        written.append(fname)

    return written
