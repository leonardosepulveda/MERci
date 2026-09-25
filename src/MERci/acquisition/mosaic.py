# MERci/acquisition/mosaic.py
"""
Derive tissue-boundary / hole polygons from a Steve low-mag mosaic instead
of drawing ``boundary_positions*.txt``/``hole*.txt`` by hand.

Workflow (``02_create_boundary_from_mosaic.ipynb``)
---------------------------------------------------
1. ``load_steve_mosaic`` -- read a Steve ``.msc`` manifest and its ``.stv``
   tiles (each carries its stage position, pixel size and stacking order).
2. ``assemble_mosaic_canvas`` -- paste the tiles into one image in stage
   microns. Tiles may mix objectives and overlap: each is resampled at its
   own pixel size and the topmost (Steve's stacking order) wins. Drop tiles
   from the list first (e.g. by ``objective_name``) to exclude them.
3. ``plots.mosaic_plots.plot_tile_intensity_histograms`` -- per-tile
   log-intensity histograms, to pick a threshold by eye when Otsu fails.
4. ``segment_mosaic_tissue`` -- threshold and clean up the canvas into
   tissue and hole polygons, in the ``x_um, y_um`` space the boundary and
   hole files use.
5. ``plots.mosaic_plots.plot_mosaic_segmentation`` -- overlay the polygons
   on the canvas to check them before saving.
6. ``save_boundary_from_mosaic`` -- write the polygons with the filenames
   ``positions.discover_boundary_files``/``load_hole_polygons`` expect.

Steve file formats (from ``storm_control.steve``; not documented elsewhere)
---------------------------------------------------------------------------
* ``<name>.msc`` -- text, one comma-separated record per line:
  ``objective,<name>,<um_per_pix>,<x_offset>,<y_offset>`` per objective and
  ``image,<filename>`` per tile. The per-objective ``(x_offset, y_offset)``
  (that objective's misalignment relative to the reference objective, which
  records ``0.00, 0.00``) is applied by :func:`load_steve_mosaic`. It is not
  a hardware constant (it changes when objectives are reinstalled), so it
  is always read from each experiment's own ``.msc``. ``um_per_pix`` is not
  used (rounded for display).
* ``<name>_<id>.stv`` -- a pickled ``ImageItem.__dict__``. Keys used:
  ``numpy_data`` (already-oriented frame), ``x_um``/``y_um`` (stage
  position of the frame centre) and ``magnification``. Pixel size comes
  from the tile's own ``x_um``/``x_pix`` and ``magnification``, not from
  ``um_per_pix`` or ``storm_control``'s mutable ``pixels_to_um``.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
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
        (composites mixed-scale/overlapping tiles directly).
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

    def to_px(self, x_um, y_um) -> Tuple[np.ndarray, np.ndarray]:
        """Stage (x_um, y_um), scalars or arrays, to canvas pixels as ``(col, row)``
        -- x then y, the order matplotlib plots in. Inverse of :meth:`to_um`."""
        x0, y0 = self.origin_um
        return ((np.asarray(x_um) - x0) / self.pixel_size_um,
                (np.asarray(y_um) - y0) / self.pixel_size_um)


@dataclass
class MosaicSegmentation:
    """Tissue/hole polygons detected in a :class:`MosaicCanvas`, plus the
    intermediate mask and threshold used -- kept around so
    :func:`MERci.plots.mosaic_plots.plot_mosaic_segmentation` can show exactly what was thresholded.
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
    (the reference objective's own line always reads ``0.00, 0.00``). On a
    shared microscope where objectives are physically removed/reinstalled
    between users, this offset is NOT a fixed hardware constant -- the same
    objective pair can record different, still-correct values across
    sessions. Reading it fresh from each experiment's own ``.msc`` file
    (never hardcoded) is therefore required.
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
    position frame without a separate manual calibration step. An objective
    with no matching ``objective,`` line (or a manifest with none at all)
    gets ``(0, 0)``.

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

    # Print exactly what offset was applied to each objective's tiles --
    # explicit, not just claimed, so a caller can directly confirm from this
    # output (before ever looking at the assembled canvas) that the returned
    # tiles' x_um/y_um already include the .msc-recorded shift, rather than
    # trusting it silently. Printed per objective actually present (not per
    # tile -- a mosaic can have 100+ tiles of one objective), including
    # objectives with no matching ``objective,`` line in the .msc (shown as
    # "no .msc line" rather than a silent (0.00, 0.00)), since that's the
    # other real way this ends up applying no shift.
    for name in sorted({t.objective_name for t in tiles}):
        n = sum(1 for t in tiles if t.objective_name == name)
        x_off, y_off = objective_offsets.get(name, (0.0, 0.0))
        source = ".msc-recorded" if name in objective_offsets else "no .msc line for this objective -- defaulted"
        print(f"load_steve_mosaic: '{name}' tiles ({n}) shifted by "
              f"(x={x_off:+.2f}, y={y_off:+.2f}) um [{source}]")

    return tiles


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
    *tiles* first.

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
    boundary_max_deviation_um: float = 1.0,
    near_fragment_max_distance_um: float = 0.0,
    min_near_fragment_area_um2:    float = 0.0,
) -> MosaicSegmentation:
    """
    Threshold a mosaic canvas into tissue and hole polygons.

    A single threshold on the raw canvas breaks one tissue into many specks
    (vignetting, tile seams), hence the pipeline:

    1. Gaussian-smooth (``smooth_sigma_um``).
    2. Threshold: Otsu on the smoothed canvas, or *threshold* if given.
    3. Close (``close_radius_um``) to bridge small gaps within a piece.
    4. Open (``open_radius_um``) to remove noise specks.
    5. Dilate by ``margin_um``, the safety margin a hand-drawn boundary has.
    6. Fill enclosed background to find holes, label tissue and hole
       components, and trace contours (``skimage.measure.find_contours``,
       converted with ``canvas.to_um``). A tissue island inside a hole becomes
       an interior ring of the hole polygon, so it is still imaged.
    7. Drop tissue below ``min_tissue_area_um2`` (except recovered fragments,
       below), holes below ``min_hole_area_um2`` and islands below
       ``min_island_area_um2``. Simplify each polygon to within
       ``boundary_max_deviation_um`` of its traced contour.

    **Fragment recovery** (``near_fragment_max_distance_um > 0``; default 0 =
    off). Small real fragments and dust specks have similar areas, so lowering
    ``min_tissue_area_um2`` brings back ~10 specks per real fragment. Distance
    separates them better: a small component is kept if it lies within
    ``near_fragment_max_distance_um`` of a kept (>= ``min_tissue_area_um2``)
    component and its area is at least ``min_near_fragment_area_um2``.

    Parameters
    ----------
    canvas : from :func:`assemble_mosaic_canvas`
    threshold : intensity threshold; ``None`` = Otsu
    smooth_sigma_um, close_radius_um, open_radius_um, margin_um : morphology
        sizes in µm (converted with ``canvas.pixel_size_um``)
    min_tissue_area_um2, min_hole_area_um2 : drop smaller components (µm²)
    min_island_area_um2 : drop smaller islands inside holes (µm²)
    boundary_max_deviation_um : Douglas-Peucker tolerance for Shapely
        ``simplify`` (a max deviation, not a segment length)
    near_fragment_max_distance_um : 0 = no fragment recovery; > 0 = recovery
        distance (µm)
    min_near_fragment_area_um2 : minimum area (µm²) of a recovered fragment
        (default 0 = no minimum)

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

    # Which tissue label ids to keep, before contour tracing -- normally
    # just the ones clearing min_tissue_area_um2, optionally expanded by the
    # near-fragment recovery pass described above.
    min_tissue_area_px = min_tissue_area_um2 / (px * px)
    tissue_areas_px = (
        ndimage.sum(np.ones_like(tissue_labels), tissue_labels, index=np.arange(1, n_tissue + 1))
        if n_tissue else np.array([])
    )
    primary_ids = set(int(i) for i in np.arange(1, n_tissue + 1)[tissue_areas_px >= min_tissue_area_px])

    if near_fragment_max_distance_um > 0 and n_tissue and primary_ids:
        primary_mask = np.isin(tissue_labels, list(primary_ids))
        dist_to_primary_px = ndimage.distance_transform_edt(~primary_mask)
        min_dist_px_per_label = ndimage.minimum(
            dist_to_primary_px, tissue_labels, index=np.arange(1, n_tissue + 1)
        )
        min_near_fragment_area_px = min_near_fragment_area_um2 / (px * px)
        recovered_ids = {
            int(lid) for lid, area_px, dist_px in zip(
                np.arange(1, n_tissue + 1), tissue_areas_px, np.atleast_1d(min_dist_px_per_label)
            )
            if lid not in primary_ids
            and area_px >= min_near_fragment_area_px
            and dist_px * px <= near_fragment_max_distance_um
        }
    else:
        recovered_ids = set()

    keep_tissue_ids = primary_ids | recovered_ids

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

    min_hole_area_px = min_hole_area_um2 / (px * px)

    # min_area_px=0 below: inclusion was already decided above (keep_tissue_ids
    # is primary_ids | recovered_ids), so _contour_polygon's own area gate
    # would otherwise re-exclude every recovered sub-threshold fragment.
    tissue_polygons = [
        p.simplify(boundary_max_deviation_um) for lid in sorted(keep_tissue_ids)
        if (p := _contour_polygon(tissue_labels, lid, 0)) is not None
    ]
    hole_polygons = [
        p.simplify(boundary_max_deviation_um) for lid in range(1, n_holes + 1)
        if (p := _contour_polygon(hole_labels, lid, min_hole_area_px,
                                  min_interior_area_um2=min_island_area_um2)) is not None
    ]

    return MosaicSegmentation(
        tissue_polygons=tissue_polygons, hole_polygons=hole_polygons,
        mask=mask_final, threshold=threshold,
    )


def estimate_bimodal_threshold(bin_centers_log: np.ndarray, counts: np.ndarray) -> Optional[float]:
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
    happen to be in the mosaic (see :func:`MERci.plots.mosaic_plots.plot_tile_intensity_histograms`
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


def normalize_tiles_for_display(
    tiles:           List[SteveTile],
    low_percentile:  float = 1.0,
    high_percentile: float = 99.0,
) -> List[SteveTile]:
    """
    Return a copy of *tiles* with each objective's own pixel values
    independently rescaled onto a shared ``[0, 1]`` display range.

    For DISPLAY/verification only -- e.g. compositing a low-mag scan
    together with its high-mag alignment tiles onto one canvas so both are
    actually visible at once (see :func:`MERci.plots.mosaic_plots.plot_objective_alignment_check`).
    A shared-percentile stretch over raw values from mixed objectives is
    otherwise unreadable: real per-objective exposure/gain differences mean
    the high-mag patch saturates (or the low-mag one vanishes) under a
    single shared range. Never use the result for real segmentation --
    :func:`segment_mosaic_tissue` needs each tile's real intensity values,
    not a display-normalized copy.

    Parameters
    ----------
    tiles : from :func:`load_steve_mosaic` (or a filtered/composed subset).
    low_percentile, high_percentile : percentile bounds -- computed by
        pooling every tile sharing one ``objective_name`` together, then
        mapped to ``[0, 1]`` -- so tiles of the same objective stay mutually
        comparable while different objectives each get their own range.

    Returns
    -------
    New list of :class:`SteveTile`, same order, each with a rescaled
    ``.image`` (float64, clipped to ``[0, 1]``).
    """
    from dataclasses import replace

    by_objective: dict = {}
    for t in tiles:
        by_objective.setdefault(t.objective_name, []).append(t)

    bounds = {}
    for objective, obj_tiles in by_objective.items():
        pooled = np.concatenate([t.image.ravel() for t in obj_tiles])
        lo, hi = np.percentile(pooled, [low_percentile, high_percentile])
        bounds[objective] = (lo, hi if hi > lo else lo + 1.0)

    normalized = []
    for t in tiles:
        lo, hi = bounds[t.objective_name]
        img = np.clip((t.image.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
        normalized.append(replace(t, image=img))
    return normalized


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


def load_or_build_mosaic_canvas_cached(
    msc_path: Path, keep_objectives: List[str], working_pixel_um: float, cache_dir: Path,
) -> Tuple[MosaicCanvas, bool, Optional[int]]:
    """
    Load *msc_path* as a :class:`MosaicCanvas`, reusing
    ``cache_dir/mosaic_canvas.npz`` (:func:`save_mosaic_canvas`/
    :func:`load_mosaic_canvas`) if it's still valid for the given
    *msc_path*/*keep_objectives*/*working_pixel_um* (a
    ``mosaic_canvas_params.json`` sidecar records what the cache was built
    from, including *msc_path*'s own mtime), else assembles it fresh via
    :func:`load_steve_mosaic`/:func:`assemble_mosaic_canvas` and caches the
    result for next time.

    Returns ``(canvas, was_cached, n_tiles)`` -- *n_tiles* is the number of
    tiles assembled, or None when the
    cache was reused (no fresh assembly, so nothing to count).
    """
    msc_path = Path(msc_path)
    cache_dir = Path(cache_dir)
    cache_npz = cache_dir / "mosaic_canvas.npz"
    cache_params = cache_dir / "mosaic_canvas_params.json"
    params_now = {
        "msc_path": str(msc_path), "msc_mtime": msc_path.stat().st_mtime,
        "keep_objectives": keep_objectives, "working_pixel_um": working_pixel_um,
    }

    cached_ok = False
    if cache_npz.exists() and cache_params.exists():
        with open(cache_params) as fh:
            cached_ok = json.load(fh) == params_now

    if cached_ok:
        return load_mosaic_canvas(cache_npz), True, None

    tiles_all = load_steve_mosaic(msc_path)
    tiles = [t for t in tiles_all if t.objective_name in keep_objectives]
    canvas = assemble_mosaic_canvas(tiles, working_pixel_um=working_pixel_um)
    save_mosaic_canvas(canvas, cache_npz)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_params, "w") as fh:
        json.dump(params_now, fh)
    return canvas, False, len(tiles)


def load_mosaic_canvas(path: Path) -> MosaicCanvas:
    """Load a :class:`MosaicCanvas` written by :func:`save_mosaic_canvas`."""
    npz = np.load(Path(path))
    return MosaicCanvas(
        image=npz["image"], covered=npz["covered"],
        origin_um=tuple(npz["origin_um"]), pixel_size_um=float(npz["pixel_size_um"]),
    )


def __getattr__(name):
    # Plotting moved to MERci.plots.mosaic_plots; keep old imports working
    # (e.g. notebooks already exported into an experiment folder).
    if name in ('plot_tile_intensity_histograms', 'plot_mosaic_segmentation', 'plot_objective_alignment_check'):
        import importlib
        return getattr(importlib.import_module('MERci.plots.mosaic_plots'), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
