# MERci/acquisition/mosaic.py
"""
Derive tissue-boundary / hole polygons automatically from a Steve low-mag
mosaic, instead of drawing ``boundary_positions*.txt``/``hole*.txt`` by hand.

Typical workflow (see ``02_create_boundary_from_mosaic.ipynb``)
------------------------------------------------------------------
1. ``load_steve_mosaic`` – read a Steve ``.msc`` manifest + its ``.stv`` tile
   pickles (each tile already carries its own stage position, pixel size,
   and display stacking order, so no separate calibration step is needed).
2. ``assemble_mosaic_canvas`` – paste every tile into one flattened image in
   real stage-micron coordinates. Tiles can mix objectives/pixel sizes and
   deliberately overlap (e.g. a few high-mag alignment FOVs over a low-mag
   scan); each is resampled using its own native pixel size, and overlaps
   are resolved by Steve's own stacking order (topmost tile wins), not
   averaged. Use ``filter_tiles_by_objective`` first only if some tiles
   should be dropped entirely rather than composited (e.g. genuinely
   unwanted debris/bubble frames).
3. ``plot_tile_intensity_histograms`` – overlay every tile's log-space
   intensity histogram, to pick a fixed segmentation threshold by eye when
   Otsu doesn't separate tissue from background well on a given sample.
4. ``segment_mosaic_tissue`` – threshold + clean up the canvas into tissue
   and hole polygons, in the same ``x_um, y_um`` coordinate space that
   ``boundary_positions.txt``/``hole*.txt`` already use.
5. ``plot_mosaic_segmentation`` – overlay the detected polygons on the
   canvas for a visual sanity check before committing to them (thresholding
   parameters are re-run interactively until the overlay looks right).
6. ``save_boundary_from_mosaic`` – write the polygons out in the exact
   filename convention ``positions.discover_boundary_files``/
   ``load_hole_polygons`` already expect, so
   ``02_create_positions_from_boundaries.ipynb`` picks them up unchanged.

Steve file formats (reverse-engineered from ``storm_control.steve``, not
otherwise documented)
------------------------------------------------------------------------
* ``<name>.msc`` – plain text, one comma-separated record per line. An
  ``objective,<name>,<um_per_pix>,<x_offset>,<y_offset>`` line per configured
  objective and an ``image,<filename>`` line per saved tile. The per-
  objective ``(x_offset, y_offset)`` -- Steve's own record of that
  objective's real parfocal/parcentric misalignment relative to whichever
  objective it treats as this session's stage-position reference (always
  ``0.00, 0.00`` on every real ``.msc`` file seen so far) -- IS used, by
  :func:`load_steve_mosaic` (added after confirming directly, on real data,
  that a previously-uncorrected mosaic/high-mag-alignment-tile discrepancy
  exactly matched this already-recorded-but-ignored value); the
  ``um_per_pix`` field in the same line is still NOT used for pixel size
  (see below -- a separate, unrelated field in the same line).
* ``<name>_<id>.stv`` – a ``pickle.dump`` of the tile's ``ImageItem.__dict__``
  (minus its Qt graphics item). The keys used here: ``numpy_data`` (the raw,
  already-oriented camera frame), ``x_um``/``y_um`` (stage position of the
  frame's *center*), and ``magnification``. The real per-tile pixel size is
  derived from the tile's own ``x_um``/``x_pix`` ratio and ``magnification``
  rather than trusting the ``.msc`` objective line's ``um_per_pix`` (which is
  rounded to 2 decimal places for display) or a hard-coded
  ``storm_control.steve.coord.Point.pixels_to_um`` value (which is a mutable
  class attribute, not a universal constant). On a shared microscope where
  objectives are physically removed/reinstalled between users, the
  ``(x_offset, y_offset)`` above is NOT a fixed hardware constant either --
  confirmed directly by comparing the same 10x-vs-60x pair's recorded offset
  across several real experiment sessions on the identical scope: correct,
  but different, values each time. It must always be read fresh from each
  experiment's own ``.msc`` file.
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

_MSC_IMAGE_PREFIX     = "image,"
_MSC_OBJECTIVE_PREFIX = "objective,"


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
        alignment/reference FOVs taken at high magnification, deliberately
        overlapping the low-mag scan -- see :func:`assemble_mosaic_canvas`
        (composites mixed-scale/overlapping tiles directly) and
        :func:`filter_tiles_by_objective` (drops one objective entirely,
        for when some tiles are genuinely unwanted rather than a different
        magnification of real tissue).
    zvalue : float
        Steve's own display stacking order for this tile (higher = drawn on
        top in Steve itself). Increases monotonically with acquisition
        order in practice. Used by :func:`assemble_mosaic_canvas` to decide,
        pixel-by-pixel, which tile "wins" where tiles overlap.
    """
    image:          np.ndarray
    x_um:           float
    y_um:           float
    pixel_size_um:  float
    objective_name: str
    zvalue:         float


@dataclass
class MosaicCanvas:
    """A Steve mosaic flattened into one image, in stage-micron coordinates.

    Attributes
    ----------
    image : np.ndarray
        Per-canvas-pixel intensity (0 where no tile covers it). Where more
        than one tile covers a pixel, the value comes from whichever tile
        has the highest ``zvalue`` there (topmost wins -- not averaged; see
        ``assemble_mosaic_canvas``).
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


def _parse_objective_offsets(msc_path: Path) -> "dict[str, Tuple[float, float]]":
    """
    Parse every ``objective,<name>,<um_per_pix>,<x_offset>,<y_offset>`` line
    of a Steve ``.msc`` manifest into ``{name: (x_offset, y_offset)}``.

    This offset is Steve's own record of the physical parfocal/parcentric
    misalignment between that objective's optical axis and whichever
    objective Steve treats as its stage-position reference for this session
    (confirmed directly across several real experiments: the reference
    objective's own line always reads ``0.00, 0.00``). On a shared
    microscope where objectives are physically removed/reinstalled between
    users, this offset is NOT a fixed hardware constant -- confirmed
    directly by comparing the same 10x-vs-60x objective pair's recorded
    offset across several real sessions on the identical scope (ST2):
    correct, but different, values each time. Reading it fresh from each
    experiment's own ``.msc`` file (never hardcoded) is therefore required.
    """
    offsets: "dict[str, Tuple[float, float]]" = {}
    with msc_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith(_MSC_OBJECTIVE_PREFIX):
                continue
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < 5:
                continue
            _, name, _um_per_pix, x_offset, y_offset = fields[:5]
            try:
                offsets[name] = (float(x_offset), float(y_offset))
            except ValueError:
                continue
    return offsets


def load_steve_mosaic(msc_path: Path) -> List[SteveTile]:
    """
    Load every tile referenced by a Steve ``.msc`` mosaic manifest.

    Every tile's ``x_um``/``y_um`` has its own objective's recorded
    ``(x_offset, y_offset)`` (the ``.msc`` file's ``objective,...`` line,
    see :func:`_parse_objective_offsets`) added before being returned, so
    tiles shot with different objectives land in one consistent stage-
    position frame without a separate manual calibration/correction step --
    a real, previously-uncorrected discrepancy between a mosaic's low-mag
    scanning objective and its high-mag alignment tiles was confirmed
    directly on real data (see ``notebooks/tests/fix_mosaic_shift_missing_
    fovs.ipynb`` and its ``prompt_history`` for the investigation) to
    exactly match this already-recorded-but-previously-unused offset. An
    objective with no matching ``objective,`` line (or a manifest with none
    at all) gets ``(0, 0)`` -- unchanged from the previous, uncorrected
    behavior.

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

    objective_offsets = _parse_objective_offsets(msc_path)

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

        x_offset, y_offset = objective_offsets.get(d["objective_name"], (0.0, 0.0))

        tiles.append(SteveTile(
            image=d["numpy_data"],
            x_um=d["x_um"] + x_offset,
            y_um=d["y_um"] + y_offset,
            pixel_size_um=pixel_size_um,
            objective_name=d["objective_name"],
            zvalue=d["zvalue"],
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
    tiles:            List[SteveTile],
    working_pixel_um: float = 5.0,
) -> MosaicCanvas:
    """
    Paste every tile into one flattened image in stage-micron coordinates.

    Tiles can mix objectives/pixel sizes and can deliberately overlap (e.g. a
    handful of high-mag alignment/reference FOVs overlaid on a low-mag scan):
    each tile is independently resampled to ``working_pixel_um`` using its
    *own* native pixel size (rather than assuming every tile shares one), and
    tiles are painted in ascending ``zvalue`` order -- Steve's own display
    stacking order -- so wherever tiles overlap, the pixel comes from
    whichever tile is topmost there (painted last, so it overwrites), never
    an average of the overlapping tiles. This matches how Steve itself
    displays the mosaic. If some tiles should be excluded entirely rather
    than composited (e.g. genuinely bad/debris frames), drop them from
    *tiles* first -- see :func:`filter_tiles_by_objective`.

    Parameters
    ----------
    tiles : tiles from :func:`load_steve_mosaic` (or a filtered subset).
    working_pixel_um : target canvas pixel size, in microns. Each tile is
        downsampled (by the nearest integer factor to this target, computed
        from that tile's own pixel size) before pasting, since full camera
        resolution (e.g. 2304x2304 per tile) is unnecessary for tissue-scale
        thresholding and would make the full mosaic canvas very large. Note
        this means each tile's *actual* resampled pixel size is only
        approximately ``working_pixel_um`` (whichever exact multiple of its
        own native pixel size is closest) -- a small (sub-pixel-scale)
        misalignment between differently-scaled tiles is possible as a
        result, which is acceptable for tissue-scale boundary detection but
        not for precision registration.

    Returns
    -------
    MosaicCanvas
    """
    if not tiles:
        raise ValueError("No tiles to assemble.")

    canvas_pixel_um = working_pixel_um

    x_min = min(t.x_um - (t.image.shape[1] * t.pixel_size_um) / 2 for t in tiles)
    x_max = max(t.x_um + (t.image.shape[1] * t.pixel_size_um) / 2 for t in tiles)
    y_min = min(t.y_um - (t.image.shape[0] * t.pixel_size_um) / 2 for t in tiles)
    y_max = max(t.y_um + (t.image.shape[0] * t.pixel_size_um) / 2 for t in tiles)

    canvas_w = int(np.ceil((x_max - x_min) / canvas_pixel_um)) + 2
    canvas_h = int(np.ceil((y_max - y_min) / canvas_pixel_um)) + 2

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    covered = np.zeros((canvas_h, canvas_w), dtype=bool)

    # Ascending zvalue: paint lowest first, highest (topmost) last, so ties
    # in coverage are resolved by simple overwrite -- topmost wins.
    for t in sorted(tiles, key=lambda t: t.zvalue):
        downsample = max(1, round(working_pixel_um / t.pixel_size_um))
        img = t.image[::downsample, ::downsample].astype(np.float32)
        tile_h, tile_w = t.image.shape
        h, w = img.shape
        x0_um = t.x_um - (tile_w * t.pixel_size_um) / 2
        y0_um = t.y_um - (tile_h * t.pixel_size_um) / 2
        col0 = int(round((x0_um - x_min) / canvas_pixel_um))
        row0 = int(round((y0_um - y_min) / canvas_pixel_um))
        row1, col1 = row0 + h, col0 + w
        # Clip in case rounding pushes a tile fractionally outside the canvas.
        row0c, col0c = max(row0, 0), max(col0, 0)
        row1c, col1c = min(row1, canvas_h), min(col1, canvas_w)
        img = img[row0c - row0: row0c - row0 + (row1c - row0c),
                  col0c - col0: col0c - col0 + (col1c - col0c)]
        canvas[row0c:row1c, col0c:col1c] = img
        covered[row0c:row1c, col0c:col1c] = True

    return MosaicCanvas(
        image=canvas, covered=covered,
        origin_um=(x_min, y_min), pixel_size_um=canvas_pixel_um,
    )


def segment_mosaic_tissue(
    canvas:               MosaicCanvas,
    threshold:            Optional[float] = None,
    smooth_sigma_um:      float = 10.0,
    close_radius_um:      float = 50.0,
    open_radius_um:       float = 15.0,
    margin_um:            float = 75.0,
    min_tissue_area_um2:  float = 1000.0,
    min_hole_area_um2:    float = 500.0,
    min_island_area_um2:  float = 1000.0,
    simplify_tol_um:      float = 15.0,
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
       trace each labelled region's contour(s) with marching squares
       (``skimage.measure.find_contours``), converting canvas-pixel
       coordinates to stage microns via ``canvas.to_um``. A hole component
       can itself enclose a real tissue **island** (a true donut/annulus
       shape, e.g. a ring of tissue around an empty center that itself has
       a tissue clump in the middle) -- marching squares then returns more
       than one contour for that one hole component: the outer boundary,
       plus one per island. Each island becomes an **interior ring** of the
       hole polygon (``shapely.geometry.Polygon(exterior, holes=[...])``),
       so the island area is correctly excluded *from* the hole (i.e. still
       imaged) instead of being silently swallowed into a solid disk.
    7. Drop components below ``min_tissue_area_um2``/``min_hole_area_um2``,
       drop islands below ``min_island_area_um2``, and simplify each
       polygon by ``simplify_tol_um`` (marching squares otherwise produces
       one vertex per canvas pixel of perimeter).

    Parameters
    ----------
    canvas : from :func:`assemble_mosaic_canvas`.
    threshold : intensity threshold; ``None`` = Otsu on the smoothed canvas.
    smooth_sigma_um, close_radius_um, open_radius_um, margin_um :
        morphology parameters, in microns (converted to canvas pixels
        internally via ``canvas.pixel_size_um``).
    min_tissue_area_um2, min_hole_area_um2 : drop components smaller than
        this (um^2) -- filters residual noise specks after morphology.
    min_island_area_um2 : drop a hole's interior island (see step 6 above)
        smaller than this (um^2) -- filters noise specks inside a hole from
        becoming spurious interior rings; a genuine tissue island is
        typically well above this.
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

    def _contour_polygon(
        label_img:              np.ndarray,
        label_id:                int,
        min_area_px:             float,
        min_interior_area_um2:   Optional[float] = None,
    ) -> Optional[Polygon]:
        component = (label_img == label_id)
        if component.sum() < min_area_px:
            return None
        # Pad so a region touching the canvas edge still yields a closed contour.
        padded = np.pad(component, 1, mode="constant", constant_values=False)
        contours = measure.find_contours(padded.astype(float), 0.5)
        if not contours:
            return None

        rings = []
        for contour in contours:
            xy = np.array([canvas.to_um(r, c) for r, c in (contour - 1)])
            if len(xy) < 4:
                continue
            ring = Polygon(xy)
            rings.append(ring if ring.is_valid else ring.buffer(0))
        if not rings:
            return None

        # The largest ring is the exterior. When min_interior_area_um2 is
        # given (hole components only), any other sufficiently large ring is
        # a real interior island -- see step 6 of this function's docstring.
        ext_idx = max(range(len(rings)), key=lambda i: rings[i].area)
        exterior = rings[ext_idx]
        interiors = []
        if min_interior_area_um2 is not None:
            interiors = [
                list(rings[i].exterior.coords)
                for i in range(len(rings))
                if i != ext_idx and rings[i].area >= min_interior_area_um2
            ]

        poly = Polygon(exterior.exterior.coords, interiors) if interiors else exterior
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
        if (p := _contour_polygon(hole_labels, lid, min_hole_area_px,
                                  min_interior_area_um2=min_island_area_um2)) is not None
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


def _classify_tiles_by_signal(
    log_images: List[np.ndarray], percentile: float = 99.0
) -> Optional[np.ndarray]:
    """
    Split tiles into "empty" (background-only) vs. "signal" (real tissue
    present) groups, from each tile's own upper-``percentile`` log-intensity
    -- a per-TILE summary statistic (one number per tile), not a per-pixel
    one, so the split isn't swamped by however many purely-empty tiles
    happen to be in the mosaic (see :func:`plot_tile_intensity_histograms`
    for why that swamping matters). 99th percentile: high enough to ignore
    an empty tile's own noise floor, low enough that a tile whose real
    tissue only covers a small fraction of its area still registers as
    elevated relative to a genuinely empty tile.

    Splitting on this small (one-value-per-tile) array with Otsu is far more
    reliable than looking for two modes in the full pooled-pixel histogram:
    it isn't diluted by the fact that most pixels, even in a tissue tile,
    are still background.

    Returns
    -------
    A boolean array (one entry per tile, True = classified as "signal"), or
    ``None`` if the per-tile statistic itself has no separable structure
    (e.g. every tile looks the same -- all empty, all tissue, or too
    uniform a sample for this split to be meaningful) -- callers should
    fall back to pooling all pixels together in that case.
    """
    from skimage.filters import threshold_otsu

    tile_stat = np.array([np.percentile(li, percentile) for li in log_images])
    if tile_stat.min() == tile_stat.max():
        return None
    try:
        split = threshold_otsu(tile_stat)
    except ValueError:
        return None
    signal_mask = tile_stat >= split
    if signal_mask.all() or not signal_mask.any():
        return None
    return signal_mask


def plot_tile_intensity_histograms(
    tiles:           List[SteveTile],
    bins:            int = 200,
    ax=None,
    color:           tuple = (0.7, 0.7, 0.7),
    alpha:           float = 0.5,
    show_threshold:  bool = True,
) -> Tuple[object, Optional[float]]:
    """
    Overlay one log-space pixel-intensity histogram per tile (thin gray
    lines), plus a solid combined histogram, weighted 50/50 between
    "empty" and "signal" tiles (:func:`_classify_tiles_by_signal`) rather
    than pooled by raw pixel count -- lets an outlier tile (a different
    objective, a debris/bubble FOV, ...) stand out, and helps pick a fixed
    segmentation threshold by eye instead of trusting Otsu blindly.

    Confirmed directly on a real dataset where most FOVs are tissue-free:
    pooling by raw pixel count let the (much more numerous) empty tiles'
    background peak swamp the real tissue peak down to ~3% of the combined
    histogram's max density -- under the 5% prominence cutoff
    :func:`_estimate_bimodal_threshold` requires, so it always returned
    ``None`` even though the tissue peak is clearly real (visible in the
    per-tile lines). Weighting the two classes equally instead of by pixel
    count fixes this regardless of how lopsided the empty/signal tile split
    is, since the two classes always contribute equal weight to the
    combined curve.

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
    polygon it sits inside. A hole that has interior rings (a real tissue
    island inside it -- a true donut/annulus, see :func:`segment_mosaic_tissue`)
    is written as ``hole{n}.txt`` (the outer boundary) plus one
    ``hole{n}_island{m}.txt`` companion file per island, the convention
    :func:`MERci.acquisition.positions.load_hole_polygons` reassembles back
    into one polygon with interior rings.

    Parameters
    ----------
    segmentation : from :func:`segment_mosaic_tissue`.
    positions_dir : directory to write into (typically
        ``SAMPLE_DIR/positions/boundaries/from_mosaic``).

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
        for m, interior in enumerate(poly.interiors, start=1):
            island_fname = f"hole{n}_island{m}.txt"
            save_positions_array(np.array(interior.coords), positions_dir / island_fname)
            written.append(island_fname)

    return written


def save_mosaic_canvas(canvas: MosaicCanvas, path: Path) -> None:
    """
    Save a :class:`MosaicCanvas` to a single compressed ``.npz`` file.

    Generalizes the ad-hoc ``image``/``covered``/``origin_um``/``pixel_size_um``
    round-trip that ``02_create_positions_from_boundaries.ipynb``'s own local
    cache cell already builds by hand -- one shared implementation, reused by
    that per-experiment cache and by bundled example canvases under
    ``MERci/data/mosaic_canvas_examples/`` (see :func:`load_mosaic_canvas`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=canvas.image, covered=canvas.covered,
        origin_um=np.array(canvas.origin_um), pixel_size_um=canvas.pixel_size_um,
    )


def load_mosaic_canvas(path: Path) -> MosaicCanvas:
    """Load a :class:`MosaicCanvas` written by :func:`save_mosaic_canvas`."""
    npz = np.load(Path(path))
    return MosaicCanvas(
        image=npz["image"], covered=npz["covered"],
        origin_um=tuple(npz["origin_um"]), pixel_size_um=float(npz["pixel_size_um"]),
    )
