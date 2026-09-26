# MERci/acquisition/positions.py
"""
Generate FOV grids and optimised scanning paths for stage-based MERFISH imaging.

Typical workflow
----------------
1. Define the tissue boundary and any excluded regions (holes) or whitelisted
   regions (subsets).
2. ``create_grid_positions`` – build a regular ``(H, W, 2)`` grid centred on the
   midpoint of the boundary bounding box; the traversal axis (columns for
   direction="vertical", rows for "horizontal") is forced even for a short
   scan-path return leg, the other axis odd for a centred cell (see its docstring).
3. ``generate_scanning_path`` – order grid points in a boustrophedon pattern.
4. ``load_hole_polygons``/``load_subset_polygons`` – load polygon masks for
   excluded (hole) or whitelisted-only (subset) areas. Holes may come as one
   file per hole (``hole{n}.txt``) or all in one combined file (see
   ``load_hole_polygons``'s docstring). ``load_hole_polygons_all_sources``
   additionally merges in hand-drawn holes from ``boundaries/manual/`` even
   when the boundary itself came from ``boundaries/from_mosaic/``.
5. ``filter_scanning_path`` – keep FOVs whose camera frame overlaps the boundary;
   remove FOVs whose camera frame overlaps any hole; if any subsets are given,
   also remove FOVs that don't overlap the subset area.
6. ``close_scanning_path`` – move the "return" points to the end of the path.
7. ``get_path_stats`` – inspect total travel distance and largest single step.
8. ``MERci.common.io.save_positions_array`` – write positions.txt.
"""
from __future__ import annotations

import csv
import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union
from shapely.strtree import STRtree

log = logging.getLogger(__name__)


# ── Grid construction ─────────────────────────────────────────────────────────

def spaced_coords(
    center: float,
    d_min:  float,
    d_max:  float,
    step:   float,
    even:   bool,
) -> np.ndarray:
    """
    Return evenly-spaced 1-D coordinates centred exactly on *center*.

    The count is the smallest number of the requested parity such that the
    coordinates span at least [d_min, d_max]: odd when *even* is ``False``
    (the traditional behaviour -- guarantees one coordinate exactly at
    *center*), or even when *even* is ``True`` (no coordinate falls exactly
    on *center*; the two innermost coordinates straddle it symmetrically).

    ``center + (arange(n) - (n - 1) / 2) * step`` is used for both parities:
    for odd n, ``(n - 1) / 2`` is an integer index offset (a true centre
    point); for even n it's a half-integer, placing points symmetrically
    on either side of *center* with none exactly on it.
    """
    span = d_max - d_min
    n    = int(np.ceil(span / step))
    if even:
        if n % 2 != 0:
            n += 1                       # ensure even
    else:
        if n % 2 == 0:
            n += 1                       # ensure odd
    return center + (np.arange(n) - (n - 1) / 2.0) * step


def create_grid_positions(
    boundary_polygon: Polygon,
    step_size:        float,
    direction:        str = "vertical",
    offset:           Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a regular 2-D grid centred on the midpoint of *boundary_polygon*'s
    bounding box (shifted by *offset*), large enough to cover its full
    bounding box.

    One axis is forced ODD (guarantees a cell exactly at the boundary's
    bounding-box midpoint when ``offset=(0, 0)``) and the other EVEN, chosen
    from *direction* so that :func:`generate_scanning_path` (called with the
    SAME *direction*) starts and ends its boustrophedon snake in the same
    row/column, instead of the opposite corner:

    * ``"vertical"``   (snake column-by-column): columns (W) EVEN, rows (H)
      ODD -- the snake starts and ends in the same row.
    * ``"horizontal"`` (snake row-by-row): rows (H) EVEN, columns (W) ODD --
      the snake starts and ends in the same column.

    Why: with direction="vertical", the snake alternates traversal
    direction column-by-column (see `generate_scanning_path`); if W is odd,
    the first and last column share the same parity and therefore the same
    sub-direction, so the path starts at one grid corner and ends at the
    *opposite* corner -- a long "return" leg back to the start of the next
    round. Forcing W even makes the first and last column have opposite
    parity/sub-direction instead, so both ends of the snake land in the
    same row -- a short return leg. (Symmetric reasoning applies to rows
    when direction="horizontal".) This parity rule is a hard constraint,
    unaffected by *offset* -- only where the grid's phase sits relative to
    the polygon shifts, never which axis is forced odd/even.

    Parameters
    ----------
    boundary_polygon : Shapely Polygon of the tissue boundary
    step_size        : distance between adjacent grid points (µm)
    direction        : ``"vertical"`` or ``"horizontal"`` -- must match the
                       *direction* passed to :func:`generate_scanning_path`
                       for the short-return property above to hold.
    offset           : ``(dx, dy)`` shift applied to the bounding-box
                       midpoint before building the grid (µm). ``(0, 0)``
                       (default) reproduces the original centred behaviour
                       exactly. Only offsets within one *step_size* period
                       produce a distinct grid phase relative to the fixed
                       polygon -- see :func:`optimize_grid_offset`, which
                       searches this space to minimise wasted (non-tissue)
                       imaged area, travel length, and FOV count.

    Returns
    -------
    grid : ``(H, W, 2)`` array of ``(x, y)`` coordinates
    xs   : 1-D x-coordinates (length W)
    ys   : 1-D y-coordinates (length H)
    """
    xmin, ymin, xmax, ymax = boundary_polygon.bounds
    cx = (xmin + xmax) / 2.0 + offset[0]
    cy = (ymin + ymax) / 2.0 + offset[1]

    # A shifted centre can sit closer to one bbox edge than the other --
    # sizing each axis's point count from the ORIGINAL (pre-shift) span
    # alone (as this used to) can then leave the outermost point short of
    # the far edge, a real gap confirmed on a real boundary (see
    # optimize_grid_offset's own docstring). Fixed by sizing each axis
    # from the LARGER of its two post-shift half-spans (radius) instead --
    # exactly reduces to the original span when offset=(0, 0) (both halves
    # equal span/2), so this is a no-op at the default offset.
    rx = max(cx - xmin, xmax - cx)
    ry = max(cy - ymin, ymax - cy)

    if direction == "vertical":
        xs = spaced_coords(cx, cx - rx, cx + rx, step_size, even=True)
        ys = spaced_coords(cy, cy - ry, cy + ry, step_size, even=False)
    elif direction == "horizontal":
        xs = spaced_coords(cx, cx - rx, cx + rx, step_size, even=False)
        ys = spaced_coords(cy, cy - ry, cy + ry, step_size, even=True)
    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    Xg, Yg = np.meshgrid(xs, ys)
    grid   = np.stack([Xg, Yg], axis=-1)   # (H, W, 2)
    return grid, xs, ys


# ── Scanning path ─────────────────────────────────────────────────────────────

def generate_scanning_path(
    grid:      np.ndarray,
    direction: str = "vertical",
) -> np.ndarray:
    """
    Order grid points in a boustrophedon (snake) pattern.

    Pass the SAME *direction* used to build *grid* via
    :func:`create_grid_positions` -- that function forces the traversal axis
    (columns for "vertical", rows for "horizontal") to an even count
    precisely so this snake's start and end land in the same row/column
    (short return leg) rather than opposite corners (see its docstring).

    Parameters
    ----------
    grid      : ``(H, W, 2)`` array as returned by :func:`create_grid_positions`
    direction : ``"vertical"`` (snake column-by-column, left → right) or
                ``"horizontal"`` (snake row-by-row, top → bottom)

    Returns
    -------
    ``(N, 2)`` array of ordered ``(x, y)`` stage coordinates.
    """
    H, W, _ = grid.shape
    path:    List = []

    if direction == "vertical":
        for j in range(W):
            rows = range(H - 1, -1, -1) if j % 2 == 0 else range(0, H)
            for i in rows:
                path.append(grid[i, j])

    elif direction == "horizontal":
        for i in range(H - 1, -1, -1):
            strip = H - 1 - i
            cols  = range(0, W) if strip % 2 == 0 else range(W - 1, -1, -1)
            for j in cols:
                path.append(grid[i, j])

    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    return np.array(path)


# ── Hole polygons ──────────────────────────────────────────────────────────────

def _read_xy_file(path: Path) -> List[Tuple[float, float]]:
    """Read a comma-separated ``x,y`` file (one vertex per line) into a list."""
    coords = []
    with Path(path).open() as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) >= 2:
                try:
                    coords.append((float(row[0]), float(row[1])))
                except ValueError:
                    pass   # skip header-like lines
    return coords


def _polygon_from_file(path: Path, islands=None) -> Optional[Polygon]:
    """Polygon from an ``x,y`` file (with optional interior *islands*), or
    None with a warning if it has < 3 points or is empty/invalid."""
    coords = _read_xy_file(path)
    if len(coords) < 3:
        log.warning("%s has fewer than 3 valid points – skipping.", path.name)
        return None
    poly = Polygon(coords, islands) if islands else Polygon(coords)
    if poly.is_empty or not poly.is_valid:
        log.warning("%s produced an invalid Shapely polygon – skipping.", path.name)
        return None
    return poly


def _split_by_distance_gap(
    coords: List[Tuple[float, float]],
) -> List[List[Tuple[float, float]]]:
    """
    Split a flat, click-ordered list of ``(x, y)`` points into separate
    polygon vertex groups, wherever the distance to the next point is a
    statistical outlier relative to the typical vertex-to-vertex spacing.

    Steve dumps every hole's vertices into one file, one polygon at a time,
    with no separator between holes. Within one hole, consecutive vertices
    are close together (they trace its outline); the jump from a hole's
    last vertex to the next hole's first vertex is much larger -- that jump
    is what this function detects, using a modified z-score (Iglewicz &
    Hoaglin) on the consecutive-distance array so no fixed µm threshold is
    needed.

    Returns
    -------
    List of point-list groups, in file order. A single hole (no outlier
    gaps) returns one group containing every point.
    """
    if len(coords) < 2:
        return [list(coords)] if coords else []

    pts   = np.asarray(coords, dtype=float)
    dists = np.hypot(*np.diff(pts, axis=0).T)

    median = np.median(dists)
    mad    = np.median(np.abs(dists - median))
    scale  = mad if mad > 0 else (np.std(dists) or median or 1.0)

    modified_z = 0.6745 * (dists - median) / scale
    gap_after  = np.flatnonzero(modified_z > 3.5)   # index i means gap after point i

    groups, start = [], 0
    for i in gap_after:
        groups.append(coords[start:i + 1])
        start = i + 1
    groups.append(coords[start:])
    return groups


def _load_combined_hole_polygons(path: Path) -> List[Polygon]:
    """
    Load hole polygons from a single combined file (see *combined_filename*
    on :func:`load_hole_polygons`): every hole's vertices, back to back, no
    separator -- split via :func:`_split_by_distance_gap`.
    """
    coords = _read_xy_file(path)
    polygons: List[Polygon] = []

    for n, group in enumerate(_split_by_distance_gap(coords), start=1):
        if len(group) < 3:
            log.warning(
                "Hole %d in %s has fewer than 3 valid points – skipping.",
                n, path.name,
            )
            continue

        poly = Polygon(group)
        if poly.is_empty or not poly.is_valid:
            log.warning(
                "Hole %d in %s produced an invalid Shapely polygon – skipping.",
                n, path.name,
            )
            continue

        polygons.append(poly)

    return polygons


def load_hole_polygons(
    hole_dir: Path,
    pattern:  str = "hole*.txt",
    combined_filename: str = "holes.txt",
) -> List[Polygon]:
    """
    Load exclusion-region polygons from a directory.

    Two input conventions, checked in this order:

    1. **Combined** -- ``{combined_filename}`` (default ``holes.txt``): every
       hole's vertices back to back in one comma-separated ``x,y`` file, no
       separator between holes (this is what Steve exports when several
       holes are drawn in one session). Split into separate holes by
       :func:`_split_by_distance_gap`. Takes priority when present; islands
       are not supported in this format.
    2. **Per-file** (unchanged) -- each ``hole{n}.txt`` file matching
       *pattern* must be a comma-separated ``x,y`` file (one vertex per
       line); files with fewer than three vertices are skipped. A hole can
       optionally have one or more companion ``hole{n}_island{m}.txt`` files
       (same format) -- these become interior rings of the hole polygon, for
       a hole that is really a donut/annulus around genuine tissue (e.g.
       auto-derived by ``MERci.acquisition.mosaic.segment_mosaic_tissue``/
       ``save_boundary_from_mosaic``): the island area is then correctly
       excluded *from* the hole (i.e. still imaged), rather than swallowed
       whole into a solid disk. Island companion files are never treated as
       holes in their own right.

    Returns
    -------
    List of valid Shapely :class:`~shapely.geometry.Polygon` objects (with
    interior rings where per-file island companion files were found).
    """
    hole_dir = Path(hole_dir)

    combined_path = hole_dir / combined_filename
    if combined_path.exists():
        return _load_combined_hole_polygons(combined_path)

    hole_re  = re.compile(r"^hole(\d+)\.txt$", re.IGNORECASE)
    polygons: List[Polygon] = []

    for path in sorted(hole_dir.glob(pattern)):
        m = hole_re.match(path.name)
        if not m:
            continue   # e.g. a hole{n}_island{m}.txt companion file -- not its own hole
        hole_id = m.group(1)

        islands = []
        for island_path in sorted(hole_dir.glob(f"hole{hole_id}_island*.txt")):
            island_coords = _read_xy_file(island_path)
            if len(island_coords) < 3:
                log.warning(
                    "%s has fewer than 3 valid points – skipping this island.",
                    island_path.name,
                )
                continue
            islands.append(island_coords)

        poly = _polygon_from_file(path, islands)
        if poly is not None:
            polygons.append(poly)

    return polygons


def load_hole_polygons_all_sources(
    positions_dir: Path,
    boundary_dir:  Path,
    pattern:            str = "hole*.txt",
    combined_filename:  str = "holes.txt",
) -> List[Polygon]:
    """
    Load hole polygons the way :func:`load_hole_polygons` does, plus also
    from the sibling ``positions_dir/boundaries/manual/`` source when it
    isn't already *boundary_dir* itself.

    Holes are already "global" within one source directory -- every hole
    file there applies to every boundary alike (see :func:`load_hole_polygons`).
    This extends that across sources too: boundaries auto-derived from a
    mosaic (``positions_dir/boundaries/from_mosaic/``) can still be combined
    with hand-drawn supplementary holes dropped in
    ``positions_dir/boundaries/manual/``, without redrawing them into the
    mosaic-derived directory or re-running mosaic segmentation. When
    *boundary_dir* already IS the manual source (or manual has no hole
    files), this is exactly :func:`load_hole_polygons`'s result -- no holes
    are ever double-loaded.

    Parameters
    ----------
    positions_dir : ``SAMPLE_DIR/positions`` (holds ``boundaries/manual/``
                     and ``boundaries/from_mosaic/``).
    boundary_dir  : the resolved boundary source directory in use (from
                     :func:`resolve_boundaries_source_dir`/:func:`resolve_boundary_dir`).
    pattern, combined_filename : forwarded to :func:`load_hole_polygons`.

    Returns
    -------
    List of valid Shapely :class:`~shapely.geometry.Polygon` objects, from
    *boundary_dir* followed by ``manual/`` (if distinct and present).
    """
    boundary_dir = Path(boundary_dir)
    polygons = load_hole_polygons(boundary_dir, pattern=pattern, combined_filename=combined_filename)

    manual_dir = Path(positions_dir) / "boundaries" / "manual"
    if manual_dir.is_dir() and manual_dir.resolve() != boundary_dir.resolve():
        polygons += load_hole_polygons(manual_dir, pattern=pattern, combined_filename=combined_filename)

    return polygons


def load_subset_polygons(
    boundary_dir: Path,
    pattern:      str = "subset*.txt",
) -> List[Polygon]:
    """
    Load whitelist region polygons from a directory -- the inverse of
    :func:`load_hole_polygons`: FOVs are kept only if they overlap one of
    these, instead of being dropped when a hole fully contains them.

    Each ``subset{n}.txt`` file matching *pattern* must be a comma-separated
    ``x,y`` file (one vertex per line, same format as ``boundary_positions*
    .txt``/``hole*.txt``); files with fewer than three vertices are skipped.
    Naming convention: ``subset1.txt``, ``subset2.txt``, etc. Like holes,
    subsets are global -- :func:`filter_scanning_path` applies the union of
    every loaded subset to every boundary alike. No result (the default) means
    no subset restriction at all, so existing layouts without any
    ``subset*.txt`` file are entirely unaffected.

    Returns
    -------
    List of valid Shapely :class:`~shapely.geometry.Polygon` objects.
    """
    boundary_dir = Path(boundary_dir)
    subset_re    = re.compile(r"^subset(\d+)\.txt$", re.IGNORECASE)
    polygons: List[Polygon] = []

    for path in sorted(boundary_dir.glob(pattern)):
        if not subset_re.match(path.name):
            continue

        poly = _polygon_from_file(path)
        if poly is not None:
            polygons.append(poly)

    return polygons


# ── Path filtering ─────────────────────────────────────────────────────────────

def filter_scanning_path(
    coords:           np.ndarray,
    boundary_polygon: Polygon,
    hole_polygons:    List[Polygon],
    fov_size_um:      float,
    min_coverage_fraction: float = 0.0,
    subset_polygons:  Optional[List[Polygon]] = None,
) -> np.ndarray:
    """
    Keep the FOVs that should be imaged for one tissue boundary.

    Each FOV is a square of side *fov_size_um* centred on its stage coordinate.
    With the default ``min_coverage_fraction=0.0`` a FOV is kept when its
    square overlaps *boundary_polygon* at all and no hole fully contains it (a
    FOV partly over a hole still images tissue, so it stays).

    With ``min_coverage_fraction > 0`` both rules are replaced by one: keep the
    FOV only if (overlap with the boundary minus every hole) / FOV area is at
    least that fraction. This drops real low-coverage tissue from the plan,
    unlike :func:`optimize_grid_offset`'s ``n_low_coverage_fovs``, which only
    ranks grid phases.

    *subset_polygons* (optional whitelist, see :func:`load_subset_polygons`)
    adds one more requirement: the square must also intersect their union.

    Parameters
    ----------
    coords           : ``(N, 2)`` candidate stage coordinates
    boundary_polygon : outer tissue boundary
    hole_polygons    : Shapely Polygons to exclude
    fov_size_um      : FOV side length (``pixel_size_um × image_size_px``)
    min_coverage_fraction : 0 = any-overlap rule; > 0 = tissue-fraction rule
    subset_polygons  : optional whitelist region(s)

    Returns
    -------
    ``(M, 2)`` array of accepted coordinates, in their original order.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    boxes  = _fov_boxes(coords, fov_size_um)

    if min_coverage_fraction <= 0.0:
        keep = shapely.intersects(boxes, boundary_polygon)
        for hole in hole_polygons:
            keep &= ~shapely.contains(hole, boxes)
    else:
        coverage = shapely.area(shapely.intersection(boxes, _effective_tissue(boundary_polygon, hole_polygons)))
        keep = coverage / (fov_size_um * fov_size_um) >= min_coverage_fraction
    if subset_polygons:
        keep &= shapely.intersects(boxes, unary_union(subset_polygons))
    return coords[keep]


def _fov_boxes(coords: np.ndarray, fov_size_um: float) -> np.ndarray:
    """Array of shapely squares (side *fov_size_um*) centred on each ``(x, y)`` in *coords*."""
    half = fov_size_um / 2.0
    x, y = coords[:, 0], coords[:, 1]
    return shapely.box(x - half, y - half, x + half, y + half)


def _effective_tissue(boundary_polygon, hole_polygons, subset_polygons=None):
    """*boundary_polygon* minus every hole, intersected with the subset union if given."""
    tissue = boundary_polygon.difference(unary_union(hole_polygons)) if hole_polygons else boundary_polygon
    if subset_polygons:
        tissue = tissue.intersection(unary_union(subset_polygons))
    return tissue


# ── Loop closure ───────────────────────────────────────────────────────────────

def _grid_indices(
    coords:    np.ndarray,
    step_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map continuous coordinates to zero-based integer grid indices."""
    eps = 1e-9
    x0  = coords[:, 0].min()
    y0  = coords[:, 1].min()
    ix  = ((coords[:, 0] - x0) / step_size + eps).astype(int)
    iy  = ((coords[:, 1] - y0) / step_size + eps).astype(int)
    return ix, iy


def _side_indices(
    coords: np.ndarray,
    ix:     np.ndarray,
    iy:     np.ndarray,
    side:   str,
) -> np.ndarray:
    """Return path indices of points on *side* of the grid."""
    side = side.lower()
    if side not in {"top", "bottom", "left", "right"}:
        raise ValueError("side must be 'top', 'bottom', 'left', or 'right'")

    all_idx   = np.arange(coords.shape[0])
    selected: List[int] = []

    if side in {"top", "bottom"}:
        for col in np.unique(ix):
            mask    = ix == col
            col_idx = all_idx[mask]
            col_iy  = iy[mask]
            k       = np.argmax(col_iy) if side == "top" else np.argmin(col_iy)
            selected.append(int(col_idx[k]))
    else:
        for row in np.unique(iy):
            mask    = iy == row
            row_idx = all_idx[mask]
            row_ix  = ix[mask]
            k       = np.argmax(row_ix) if side == "right" else np.argmin(row_ix)
            selected.append(int(row_idx[k]))

    return np.array(sorted(set(selected)), dtype=int)


def close_scanning_path(
    coords:      np.ndarray,
    step_size:   float,
    return_side: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder the path so that the points on *return_side* appear at the end,
    making it easy to start the next round near the original starting position.

    The very first point (index 0) is never relocated.

    Parameters
    ----------
    coords      : ``(N, 2)`` ordered stage coordinates
    step_size   : grid spacing in µm (used to compute grid indices)
    return_side : ``"top"``, ``"bottom"``, ``"left"``, or ``"right"``

    Returns
    -------
    new_coords : ``(N, 2)`` reordered array
    side_idxs  : original indices of the points that were moved
    """
    coords = np.asarray(coords, dtype=float)
    N      = coords.shape[0]

    ix, iy    = _grid_indices(coords, step_size)
    side_idxs = _side_indices(coords, ix, iy, return_side)
    side_idxs = side_idxs[side_idxs != 0]   # never move the starting point

    stay_mask            = np.ones(N, dtype=bool)
    stay_mask[side_idxs] = False
    stay_idxs            = np.arange(N)[stay_mask]

    new_order = np.concatenate([stay_idxs, side_idxs[::-1]])
    return coords[new_order], side_idxs


def determine_return_side(
    coords:     np.ndarray,
    fixed_axis: str,
    step_size:  float,
) -> Tuple[str, str]:
    """
    Pick which of the two natural cross-axis sides :func:`close_scanning_path`
    should move to the end, automatically, from the raw path's own
    start/end geometry -- instead of a hardcoded guess.

    Only meaningful for the natural ``(fixed_axis, return_side)`` pairing
    that keeps :func:`close_scanning_path`'s quantised ``_side_indices``
    selection valid for a single-axis-adaptive irregular grid (see
    :func:`build_irregular_boundary_path`): ``fixed_axis="y"`` ->
    ``"left"``/``"right"``, ``fixed_axis="x"`` -> ``"top"``/``"bottom"``.
    The reverse pairing groups by the cross axis, which is not a shared
    lattice across bands, and is not supported here.

    Checks which side of ITS OWN row/column the raw (pre-closure) path's
    start point and end point each sit on (reusing ``_side_indices``, the
    same primitive :func:`close_scanning_path` itself uses) -- if both
    agree on one side, that side is returned directly, since the path
    already naturally begins/ends there (no new jump needed to close it).
    If they disagree (e.g. an odd fixed-axis position count, or a boundary
    shape that breaks the same-side guarantee), falls back to actually
    trying both sides and keeping whichever gives the smaller
    ``max_step_um`` -- an empirical tie-break, not a second guess. Verified
    (in the local, unshipped irregular-grid return-path test notebook)
    to never pick the worse of the two natural sides, on a real benchmark.

    Parameters
    ----------
    coords     : ``(N, 2)`` ordered stage coordinates (pre-closure)
    fixed_axis : ``"y"`` or ``"x"`` -- which axis is the regular lattice
                (see :func:`build_irregular_bands`)
    step_size  : grid spacing (µm)

    Returns
    -------
    (side, reason) : the chosen ``return_side``, and ``"start/end agree"``
        or ``"fallback: shorter max_step_um"`` for how it was decided.
    """
    if fixed_axis not in ("x", "y"):
        raise ValueError("fixed_axis must be 'x' or 'y'")

    side_a, side_b = {"y": ("left", "right"), "x": ("top", "bottom")}[fixed_axis]
    ix, iy = _grid_indices(coords, step_size)
    idxs_a = set(_side_indices(coords, ix, iy, side_a).tolist())
    idxs_b = set(_side_indices(coords, ix, iy, side_b).tolist())

    def _side_of(idx):
        in_a, in_b = idx in idxs_a, idx in idxs_b
        if in_a and not in_b:
            return side_a
        if in_b and not in_a:
            return side_b
        return None   # on neither/both -- ambiguous (e.g. a single-point row/column)

    start_side, end_side = _side_of(0), _side_of(len(coords) - 1)
    if start_side is not None and start_side == end_side:
        return start_side, "start/end agree"

    best_side, best_max_step = None, None
    for side in (side_a, side_b):
        closed, _ = close_scanning_path(coords, step_size, return_side=side)
        _, max_step_um = get_path_stats(closed)
        if best_max_step is None or max_step_um < best_max_step:
            best_side, best_max_step = side, max_step_um
    return best_side, "fallback: shorter max_step_um"


# ── Path statistics ────────────────────────────────────────────────────────────

def get_path_stats(coords: np.ndarray) -> Tuple[float, float]:
    """
    Return ``(total_length_um, max_step_um)`` for an ordered path.

    Parameters
    ----------
    coords : ``(N, 2)`` array of stage coordinates (µm)

    Returns
    -------
    total_length : sum of Euclidean step distances
    max_step     : largest single step distance
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        return 0.0, 0.0
    dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return float(dists.sum()), float(dists.max())


# ── Exterior-FOV detection ────────────────────────────────────────────────────

def find_exterior_fovs(
    positions:          Dict[int, Tuple[float, float]],
    step_size:           float,
    connectivity:        str   = "8",
    tolerance_fraction:  float = 0.25,
) -> Set[int]:
    """
    FOVs on the exterior of an imaged grid: the outer perimeter and the edges
    of any holes, i.e. every FOV with at least one grid-neighbour position
    that is not imaged.

    Neighbours are looked up in real stage coordinates with a KD-tree, not by
    snapping to one shared ``(row, col)`` grid: in a multi-boundary layout each
    tissue piece has its own grid phase (see :func:`create_grid_positions`), so
    a shared snap could misjudge adjacency where pieces meet.

    Parameters
    ----------
    positions          : {fov_id: (x, y)} in µm. Use one round's own imaged
                         FOVs, not the whole positions file, so transit-only
                         FOVs never enter the result.
    step_size          : grid step (µm), e.g. ``ExperimentConfig.step_size_um``
    connectivity       : "4" (N/S/E/W) or "8" (default, adds diagonals; also
                         catches FOVs at diagonal-only notches and corners)
    tolerance_fraction : how close (fraction of *step_size*) a FOV must be to
                         a neighbour position to count as present

    Returns
    -------
    Set of exterior FOV ids.
    """
    if connectivity not in ("4", "8"):
        raise ValueError(f"connectivity must be '4' or '8', got {connectivity!r}")
    if not positions:
        return set()

    fov_ids, tree = _position_tree(positions)
    coords = tree.data
    tol    = tolerance_fraction * step_size

    offsets = [(step_size, 0.0), (-step_size, 0.0), (0.0, step_size), (0.0, -step_size)]
    if connectivity == "8":
        offsets += [
            (step_size, step_size), (step_size, -step_size),
            (-step_size, step_size), (-step_size, -step_size),
        ]

    # (n_fovs, n_offsets) distance from each candidate neighbour position to its nearest FOV.
    dist, _ = tree.query(coords[:, None, :] + np.asarray(offsets)[None, :, :])
    return {fov_ids[i] for i in np.flatnonzero((dist > tol).any(axis=1))}


def median_nn_distance(coords) -> float:
    """Median nearest-neighbour distance between *coords* (``(N, 2)``) -- the grid step."""
    from scipy.spatial import cKDTree
    coords = np.asarray(coords, dtype=float)
    dists, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(dists[:, 1]))


def _position_tree(positions: Dict[int, Tuple[float, float]]):
    """``(fov_ids, cKDTree)`` over *positions*' coordinates, cached by content
    (find_grid_neighbor is called thousands of times on the same positions)."""
    return _position_tree_cached(tuple((f, float(x), float(y)) for f, (x, y) in positions.items()))


@functools.lru_cache(maxsize=8)
def _position_tree_cached(items: Tuple[Tuple[int, float, float], ...]):
    from scipy.spatial import cKDTree
    fov_ids = [f for f, _, _ in items]
    return fov_ids, cKDTree(np.array([(x, y) for _, x, y in items], dtype=float))


def find_grid_neighbor(
    fov_id:             int,
    positions:          Dict[int, Tuple[float, float]],
    direction:          str,
    step_size:          float,
    tolerance_fraction: float = 0.25,
) -> Optional[int]:
    """
    Find the real FOV id at *fov_id*'s 4-connected grid neighbour in the
    given *direction*, or ``None`` if that neighbour was not imaged (e.g.
    *fov_id* is on the grid's exterior -- see :func:`find_exterior_fovs`).

    Same KD-tree "is there a real FOV near this candidate position" approach
    as :func:`find_exterior_fovs` -- queries the actual stage coordinates
    rather than snapping onto a shared integer grid index, so it stays
    correct for any single tissue-piece's own FOV grid regardless of other
    pieces' phase (see that function's docstring for why).

    Parameters
    ----------
    fov_id     : the anchor FOV
    positions  : {fov_id: (x, y)} stage coordinates (µm) of the FOVs to
                 search among (scope this to one round's own real imaged
                 FOVs, matching :func:`find_exterior_fovs`)
    direction  : ``"right"`` (+x), ``"left"`` (-x), ``"up"`` (+y), or
                 ``"down"`` (-y)
    step_size  : grid step size (µm)
    tolerance_fraction : match tolerance for "is a neighbour actually
                 present", as a fraction of step_size

    Returns
    -------
    The neighbour's FOV id, or ``None`` if no real FOV sits there.
    """
    offsets = {
        "right": (step_size, 0.0), "left": (-step_size, 0.0),
        "up":    (0.0, step_size), "down": (0.0, -step_size),
    }
    if direction not in offsets:
        raise ValueError(f"direction must be one of {list(offsets)}, got {direction!r}")
    if fov_id not in positions:
        raise KeyError(f"fov_id {fov_id} not in positions.")

    fov_ids, tree = _position_tree(positions)
    x, y   = positions[fov_id]
    dx, dy = offsets[direction]
    dist, idx = tree.query([x + dx, y + dy])
    if dist > tolerance_fraction * step_size:
        return None
    return fov_ids[int(idx)]


def find_3x3_block(
    fov_ids:            Sequence[int],
    positions:          Dict[int, Tuple[float, float]],
    step_size:          float,
    tolerance_fraction: float = 0.25,
) -> Optional[Dict[str, int]]:
    """
    First FOV in *fov_ids* with all 8 grid neighbours imaged (see
    :func:`find_grid_neighbor`), as ``{"center", "up", "down", "left",
    "right", "up_left", "up_right", "down_left", "down_right": fov_id}``;
    ``None`` if no FOV has a complete neighbourhood. A diagonal is looked up
    from the vertical neighbour first, then from the horizontal one.
    """
    def nb(fov, direction):
        return find_grid_neighbor(fov, positions, direction, step_size, tolerance_fraction)

    def diag(fov_a, dir_a, fov_b, dir_b):
        d = nb(fov_a, dir_a)
        return d if d is not None else nb(fov_b, dir_b)

    for center in fov_ids:
        up, down, left, right = (nb(center, d) for d in ("up", "down", "left", "right"))
        if None in (up, down, left, right):
            continue
        diagonals = {
            "up_left":    diag(up, "left", left, "up"),
            "up_right":   diag(up, "right", right, "up"),
            "down_left":  diag(down, "left", left, "down"),
            "down_right": diag(down, "right", right, "down"),
        }
        if None in diagonals.values():
            continue
        return {"center": center, "up": up, "down": down, "left": left, "right": right, **diagonals}
    return None


# ── Multi-tissue / multi-boundary discovery ─────────────────────────────────────

@dataclass
class BoundarySpec:
    """One tissue-boundary input file and where it sits in the acquisition order.

    Attributes
    ----------
    tissue : int
        Tissue section index (1-based). Always 1 for the single-tissue and legacy
        layouts.
    boundary : int
        Boundary index within the tissue (1-based).
    path : Path
        The ``*boundary_positions*.txt`` file for this boundary.
    label : str
        Short segment label used in output filenames: ``"T{t}B{b}"`` for the
        multi-tissue layout, ``"B{b}"`` for a single tissue with several
        boundaries, and ``""`` for the legacy single-boundary layout.
    """

    tissue:   int
    boundary: int
    path:     Path
    label:    str


_MULTI_BOUNDARY_RE  = re.compile(r"^tissue_(\d+)_boundary_positions_(\d+)\.txt$", re.IGNORECASE)
_SINGLE_BOUNDARY_RE = re.compile(r"^boundary_positions_(\d+)\.txt$", re.IGNORECASE)


def discover_boundary_files(positions_dir: Path) -> Tuple[List[BoundarySpec], str]:
    """
    Auto-detect the tissue/boundary layout from the filenames in *positions_dir*.

    Three layouts are recognised, in priority order:

    * **multi**  – ``tissue_{t}_boundary_positions_{b}.txt`` (several tissue
      sections, each possibly split across several boundary files). Labels
      ``T{t}B{b}``.
    * **single** – ``boundary_positions_{b}.txt`` (one tissue, several
      boundaries). Labels ``B{b}``.
    * **legacy** – a lone ``boundary_positions.txt`` (one boundary). Label ``""``.

    Boundaries are returned in acquisition order: sorted by tissue, then boundary.
    This global order defines the ``transit_k`` numbering used by the caller
    (``transit_k`` connects boundary *k* to boundary *k+1*, wrapping the last back
    to the first).

    Parameters
    ----------
    positions_dir : directory holding the boundary files

    Returns
    -------
    (specs, mode) : (list of BoundarySpec, str)
        *mode* is ``"multi"``, ``"single"`` or ``"legacy"``.

    Raises
    ------
    FileNotFoundError
        if no boundary file of any recognised layout is present.
    """
    positions_dir = Path(positions_dir)

    multi:  List[BoundarySpec] = []
    single: List[BoundarySpec] = []
    for p in sorted(positions_dir.glob("*.txt")):
        m = _MULTI_BOUNDARY_RE.match(p.name)
        if m:
            t, b = int(m.group(1)), int(m.group(2))
            multi.append(BoundarySpec(t, b, p, f"T{t}B{b}"))
            continue
        s = _SINGLE_BOUNDARY_RE.match(p.name)
        if s:
            b = int(s.group(1))
            single.append(BoundarySpec(1, b, p, f"B{b}"))

    if multi:
        multi.sort(key=lambda s: (s.tissue, s.boundary))
        return multi, "multi"

    if single:
        single.sort(key=lambda s: s.boundary)
        return single, "single"

    legacy = positions_dir / "boundary_positions.txt"
    if legacy.exists():
        return [BoundarySpec(1, 1, legacy, "")], "legacy"

    raise FileNotFoundError(
        f"No boundary files found in {positions_dir}. Expected one of: "
        f"'tissue_<t>_boundary_positions_<b>.txt', 'boundary_positions_<b>.txt', "
        f"or 'boundary_positions.txt'."
    )


@dataclass
class BoundaryGroup:
    """One acquisition-order "boundary" segment, possibly merging more than
    one physical boundary file of the same tissue.

    Attributes
    ----------
    tissue : int
        Which tissue this segment belongs to.
    label : str
        Segment label used in output filenames -- ``"T{t}"``/``""`` for a
        merged "legacy" segment (see below), or the source boundary's own
        label (``"T{t}B{b}"``/``"B{b}"``/``""``) when not merged.
    boundary_indices : Tuple[int, ...]
        Indices into the ``boundaries`` list (as returned by
        :func:`discover_boundary_files`) that this segment covers, in order.
        Length 1 unless merged under ``"legacy"`` mode (see
        :func:`group_boundaries_by_path_mode`).
    """
    tissue:           int
    label:            str
    boundary_indices: Tuple[int, ...]


def group_boundaries_by_path_mode(
    boundaries:       Sequence[BoundarySpec],
    mode:             str,
    tissue_path_mode: Callable[[int], str],
) -> List[BoundaryGroup]:
    """
    Group each tissue's boundaries into acquisition-order segments.

    A tissue with more than one boundary becomes ONE segment (label ``"T{t}"``
    in multi mode, else ``""``) when its path mode is ``"legacy"`` or
    ``"union"``; otherwise each boundary is its own segment. The two merged
    modes differ only in how notebook 02 builds the segment's FOVs, which
    this function never touches.

    The single source of truth for this grouping: notebook 02 (positions
    files) and :func:`MERci.acquisition.dave.create_round_info_multitissue`
    (round_info rows) both call it, so they always agree. Transit segments
    between groups are added by the callers.

    Parameters
    ----------
    boundaries       : from :func:`discover_boundary_files`, sorted by tissue
    mode             : ``"multi"``, ``"single"`` or ``"legacy"`` (same call)
    tissue_path_mode : tissue index -> ``"legacy"``, ``"transit"`` or ``"union"``

    Returns
    -------
    List of :class:`BoundaryGroup`, in acquisition order.
    """
    groups: List[BoundaryGroup] = []
    for t, i, j in _tissue_runs(boundaries):
        if tissue_path_mode(t) in ("legacy", "union") and j - i > 1:
            label = f"T{t}" if mode == "multi" else ""
            groups.append(BoundaryGroup(t, label, tuple(range(i, j))))
        else:   # "transit", or a single boundary: one segment each
            for k in range(i, j):
                groups.append(BoundaryGroup(t, boundaries[k].label, (k,)))
    return groups


def _tissue_runs(boundaries: Sequence[BoundarySpec]):
    """Yield ``(tissue, i, j)`` for each run ``boundaries[i:j]`` of one tissue."""
    i = 0
    while i < len(boundaries):
        t = boundaries[i].tissue
        j = i
        while j < len(boundaries) and boundaries[j].tissue == t:
            j += 1
        yield t, i, j
        i = j


def merge_union_tissue_boundaries(
    boundaries:        List[BoundarySpec],
    boundary_polygons: List[Polygon],
    mode:               str,
    tissue_path_mode:   Callable[[int], str],
) -> Tuple[List[BoundarySpec], List[Polygon]]:
    """
    Collapse each ``"union"``-path-mode tissue's own multiple boundary
    pieces into ONE pseudo-boundary (their union polygon) up front.

    A tissue with several disjoint boundary pieces (e.g. one tissue section
    split by a real gap on the coverslip) normally gets a separate,
    independently-phased FOV grid per piece ("legacy"/"transit" modes,
    concatenated or transit-bridged). ``"union"`` mode instead images every
    piece with ONE shared grid and ONE continuous boustrophedon snake,
    which naturally threads through the gap between pieces (FOVs over the
    gap are simply dropped by the normal boundary-overlap filter, just as
    they would be for any other non-tissue area) instead of stitching
    separate per-piece paths together.

    Calling this BEFORE any other per-boundary grouping/grid/plotting logic
    (in particular, before :func:`group_boundaries_by_path_mode`) collapses
    a "union" tissue's *boundaries*/*boundary_polygons* entries down to
    exactly one each -- so every other multi-boundary code path, which
    already only has to handle "one boundary per tissue", needs no changes
    at all: :func:`create_grid_positions`/:func:`generate_scanning_path`/
    :func:`filter_scanning_path` are all Shapely Polygon/MultiPolygon-
    agnostic, so building the usual single-boundary FOV path (e.g. via
    :func:`build_boundary_path`/:func:`optimize_grid_offset`) against the
    merged (Multi)Polygon just works.

    Parameters
    ----------
    boundaries        : from :func:`discover_boundary_files` (sorted by
                        tissue, then boundary)
    boundary_polygons : each entry's own polygon, same order (one
                        :func:`load_boundary_polygon` call per entry)
    mode              : ``"multi"``, ``"single"`` or ``"legacy"`` (from the
                        same discovery call) -- selects the merged pseudo-
                        boundary's label (``"T{t}"`` for multi, ``""``
                        otherwise), matching :func:`group_boundaries_by_path_mode`'s
                        own merged-segment label convention exactly
    tissue_path_mode  : tissue index -> ``"legacy"``, ``"transit"`` or ``"union"``

    Returns
    -------
    (boundaries, boundary_polygons) : same shapes/order contract as the
        inputs -- each "union" tissue's own run of >1 entries replaced by
        one merged entry (``boundary=1``, ``path`` = its first piece's own
        file, ``label`` matching the convention above); every other tissue
        (or a "union" tissue with only one boundary piece to begin with)
        passes through unchanged.
    """
    merged_boundaries: List[BoundarySpec] = []
    merged_polygons:   List[Polygon]      = []
    for t, i, j in _tissue_runs(boundaries):
        if tissue_path_mode(t) == "union" and j - i > 1:
            label = f"T{t}" if mode == "multi" else ""
            merged_boundaries.append(BoundarySpec(t, 1, boundaries[i].path, label))
            merged_polygons.append(unary_union(boundary_polygons[i:j]))
        else:
            merged_boundaries.extend(boundaries[i:j])
            merged_polygons.extend(boundary_polygons[i:j])
    return merged_boundaries, merged_polygons


def has_boundary_files(positions_dir: Path) -> bool:
    """Return ``True`` if *positions_dir* holds boundary files of any layout.

    Checks for the multi (``tissue_<t>_boundary_positions_<b>.txt``), single
    (``boundary_positions_<b>.txt``) or legacy (``boundary_positions.txt``)
    naming — i.e. whether :func:`discover_boundary_files` would succeed.
    """
    positions_dir = Path(positions_dir)
    if not positions_dir.is_dir():
        return False
    for p in positions_dir.glob("*.txt"):
        if _MULTI_BOUNDARY_RE.match(p.name) or _SINGLE_BOUNDARY_RE.match(p.name):
            return True
    return (positions_dir / "boundary_positions.txt").exists()


def resolve_boundaries_source_dir(
    positions_dir: Path,
    source:        Optional[str] = None,
) -> Tuple[Path, str]:
    """
    Resolve which ``positions/boundaries/<source>/`` subfolder to read tissue
    boundaries from.

    Boundary inputs live under two possible sources: ``manual`` (hand-drawn)
    or ``from_mosaic`` (auto-derived by ``02_create_boundary_from_mosaic
    .ipynb``) — both write/read the same ``boundary_positions*.txt``/
    ``hole*.txt`` file convention, just from different subfolders, so this
    is the single place that decides which one a downstream notebook
    (``02_create_positions_from_boundaries.ipynb``, ``03_create_round_info
    .ipynb``) actually uses, keeping both in agreement without needing to
    pass state between separate notebook runs.

    Parameters
    ----------
    positions_dir : ``SAMPLE_DIR/positions``.
    source : ``"from_mosaic"``, ``"manual"``, or ``None`` to auto-detect --
        prefers ``from_mosaic`` if it has boundary files, else ``manual``,
        else ``manual`` again (as the target for :func:`resolve_boundary_dir`'s
        example-data fallback, since a hand-drawn-style example set belongs
        with the manual source).

    Returns
    -------
    (source_dir, source) : the resolved ``positions/boundaries/<source>``
        directory and which source string was used.
    """
    positions_dir = Path(positions_dir)
    boundaries_root = positions_dir / "boundaries"

    if source is not None:
        return boundaries_root / source, source

    for candidate in ("from_mosaic", "manual"):
        candidate_dir = boundaries_root / candidate
        if has_boundary_files(candidate_dir):
            return candidate_dir, candidate

    return boundaries_root / "manual", "manual"


def resolve_boundary_dir(
    primary_dir:    Path,
    example_root:   Optional[Path] = None,
    example_layout: Optional[str]  = None,
) -> Tuple[Path, bool]:
    """
    Pick the directory to read tissue boundaries from, with an example fallback.

    Returns *primary_dir* when it already contains boundary files. Otherwise —
    handy when the experiment's ``positions/`` folder is still empty — falls back
    to a bundled example dataset ``example_root/example_layout`` (e.g. the
    ``MERci/data/positions/examples/{legacy,single,multi}`` sets), so the notebook
    can be run and tested before any real boundaries are drawn.

    Parameters
    ----------
    primary_dir    : the experiment's ``positions/`` directory (preferred)
    example_root   : directory holding the example layout subfolders; if ``None``
                     no fallback is attempted
    example_layout : which example subfolder to use (``"legacy"``, ``"single"``
                     or ``"multi"``)

    Returns
    -------
    (boundary_dir, used_example) : (Path, bool)
        *used_example* is ``True`` when the example fallback was selected.

    Raises
    ------
    FileNotFoundError
        if *primary_dir* has no boundary files and no usable example fallback
        is available.
    """
    primary_dir = Path(primary_dir)
    if has_boundary_files(primary_dir):
        return primary_dir, False

    if example_root is not None and example_layout is not None:
        example_dir = Path(example_root) / example_layout
        if has_boundary_files(example_dir):
            return example_dir, True

    raise FileNotFoundError(
        f"No boundary files in {primary_dir}"
        + (
            f" and no example dataset at {Path(example_root) / example_layout}"
            if example_root is not None and example_layout is not None
            else " (and no example fallback was configured)"
        )
        + ". Add boundary files, or point example_root/example_layout at a bundled "
          "example set (e.g. MERci/data/positions/examples/{legacy,single,multi})."
    )


def load_boundary_polygon(path: Path) -> Polygon:
    """
    Load a tissue-boundary polygon from a comma-separated ``x,y`` file.

    Lines that cannot be parsed as two floats (e.g. headers or ``#`` comments)
    are skipped. Requires at least three valid vertices.

    Parameters
    ----------
    path : path to the boundary ``.txt`` file

    Returns
    -------
    shapely.geometry.Polygon
    """
    path = Path(path)
    coords = _read_xy_file(path)
    if len(coords) < 3:
        raise ValueError(f"{path} has fewer than 3 valid (x, y) vertices.")
    return Polygon(coords)


# ── Transit path between boundaries ──────────────────────────────────────────────

def create_transit_path(
    point_a:        np.ndarray,
    point_b:        np.ndarray,
    step_size:      float,
    spacing_factor: float = 2.0,
) -> np.ndarray:
    """
    Build the transit FOV path from *point_a* to *point_b*.

    Transit FOVs move the stage smoothly between two tissue boundaries. The path
    starts at *point_a* (the last FOV of one boundary), ends at *point_b* (the
    first FOV of the next), and places intermediate FOVs along the straight line
    between them, spaced about ``spacing_factor × step_size`` apart.

    The intermediate count is ``round(dist / spacing)`` so both endpoints are hit
    exactly and the realised spacing is as close to the target as an integer
    number of equal steps allows. When the two points are closer than one spacing,
    only the two endpoints are returned.

    Parameters
    ----------
    point_a, point_b : ``(2,)`` ``(x, y)`` endpoints (µm)
    step_size        : grid spacing (µm)
    spacing_factor   : target transit spacing as a multiple of *step_size*
                       (default 2.0 → transit FOVs every two grid steps)

    Returns
    -------
    ``(M, 2)`` array of transit coordinates, endpoints included (``M >= 2``).
    """
    a = np.asarray(point_a, dtype=float).reshape(2)
    b = np.asarray(point_b, dtype=float).reshape(2)
    spacing = spacing_factor * step_size
    dist    = float(np.linalg.norm(b - a))

    n_intervals = int(round(dist / spacing)) if spacing > 0 else 0
    n_intervals = max(1, n_intervals)                 # at least the two endpoints

    ts  = np.linspace(0.0, 1.0, n_intervals + 1)      # includes 0 (A) and 1 (B)
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]
    return pts


# ── Per-boundary FOV path ────────────────────────────────────────────────────────

def build_boundary_path(
    boundary_polygon: Polygon,
    hole_polygons:    List[Polygon],
    step_size:        float,
    fov_size_um:      float,
    direction:        str            = "vertical",
    return_side:      Optional[str]  = None,
    min_coverage_fraction: float     = 0.0,
    subset_polygons:  Optional[List[Polygon]] = None,
    offset:           Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Build the ordered FOV path for a single boundary.

    Convenience wrapper that runs the standard per-boundary pipeline:
    :func:`create_grid_positions` → :func:`generate_scanning_path` →
    :func:`filter_scanning_path`, optionally followed by
    :func:`close_scanning_path`.

    Parameters
    ----------
    boundary_polygon : the tissue boundary for this segment -- a Polygon or
                       MultiPolygon (e.g. from :func:`merge_union_tissue_
                       boundaries`, for a "union"-mode tissue's several
                       disjoint pieces sharing one grid); every Shapely call
                       this function makes is Multi-geometry-agnostic
    hole_polygons    : exclusion polygons (applied to this boundary)
    step_size        : grid spacing (µm)
    fov_size_um      : camera FOV side length (µm)
    direction        : boustrophedon direction for
                       :func:`generate_scanning_path`
    return_side      : if given, :func:`close_scanning_path` moves that side's
                       points to the end; if ``None`` (default) the raw snake
                       order is kept — preferred in the multi-boundary layout,
                       where the transit segments handle travel between boundaries
    min_coverage_fraction : forwarded to :func:`filter_scanning_path` --
                       ``0.0`` (default) keeps every boundary-overlapping FOV
                       as before; > 0 additionally drops real low-coverage
                       tissue instead of just imaging it (see that
                       function's docstring).
    subset_polygons  : forwarded to :func:`filter_scanning_path` -- optional
                       whitelist region(s); ``None``/empty (default) keeps
                       every boundary-overlapping FOV as before.
    offset           : grid phase, forwarded to :func:`create_grid_positions`

    Returns
    -------
    ``(M, 2)`` ordered stage coordinates for this boundary.
    """
    grid, _, _ = create_grid_positions(boundary_polygon, step_size, direction=direction, offset=offset)
    path       = generate_scanning_path(grid, direction=direction)
    filtered   = filter_scanning_path(path, boundary_polygon, hole_polygons, fov_size_um,
                                       min_coverage_fraction=min_coverage_fraction,
                                       subset_polygons=subset_polygons)
    if return_side is not None and len(filtered) > 1:
        filtered, _ = close_scanning_path(filtered, step_size, return_side=return_side)
    return filtered


# ── Grid offset optimization ───────────────────────────────────────────────────

@dataclass
class GridOffsetCandidate:
    """One evaluated grid phase and the metrics of the path it produces."""
    offset:              Tuple[float, float]
    n_fovs:               int
    waste_area_um2:       float
    total_length_um:      float
    max_step_um:          float
    n_low_coverage_fovs:  int


@dataclass
class GridOffsetResult:
    """Winner (by *priority*) plus every candidate evaluated, for inspection."""
    coords:      np.ndarray
    best:        GridOffsetCandidate
    candidates:  List[GridOffsetCandidate]


def optimize_grid_offset(
    boundary_polygon: Polygon,
    hole_polygons:    List[Polygon],
    step_size:        float,
    fov_size_um:      float,
    direction:        str                 = "vertical",
    return_side:      Optional[str]       = None,
    n_samples:        int                 = 9,
    priority:         Tuple[str, ...]     = ("n_fovs", "waste_area_um2", "total_length_um", "n_low_coverage_fovs"),
    low_coverage_fraction: float          = 0.5,
    min_coverage_fraction: float          = 0.0,
    subset_polygons:  Optional[List[Polygon]] = None,
) -> GridOffsetResult:
    """
    Search the grid's phase (its offset within one *step_size* period) for the
    best one by *priority*, keeping *step_size* and *direction* fixed so
    :func:`create_grid_positions`'s parity rule for a short return leg holds.

    Offsets one full step apart give the same lattice, so each axis samples
    ``[-step_size/2, step_size/2)``.

    Metrics per candidate (all minimised):

    * ``n_fovs`` -- the direct proxy for imaging time (each FOV costs about the
      same time whatever it contains). First in the default *priority*: on a
      real benchmark every ordering traded off only a few FOVs against
      ``n_low_coverage_fovs``, so the simplest objective leads.
    * ``waste_area_um2`` -- imaged area that is not tissue (holes count as
      non-tissue, even under FOVs :func:`filter_scanning_path` keeps).
    * ``total_length_um`` -- scan travel length, measured after
      :func:`close_scanning_path` when *return_side* is given.
    * ``n_low_coverage_fovs`` -- FOVs whose tissue fraction is below
      *low_coverage_fraction*. Counts wasted time rather than wasted area, so
      it can disagree with ``waste_area_um2``. Ranking only, never drops a FOV.

    Parameters
    ----------
    boundary_polygon : the tissue boundary for this segment
    hole_polygons    : exclusion polygons for this boundary
    step_size        : grid spacing (µm)
    fov_size_um      : FOV side length (µm)
    direction        : boustrophedon direction, forwarded to
                       :func:`create_grid_positions`/:func:`generate_scanning_path`
    return_side      : forwarded to :func:`close_scanning_path`
    n_samples        : offsets per axis (``n_samples**2`` candidates); an odd
                       value includes the centred ``(0, 0)`` grid
    priority         : ``GridOffsetCandidate`` field names, most important
                       first, for lexicographic ranking
    low_coverage_fraction : threshold for ``n_low_coverage_fovs`` (default 0.5)
    min_coverage_fraction : forwarded to :func:`filter_scanning_path`; > 0
                       actually drops low-coverage FOVs from every candidate.
                       Answers a different question from
                       *low_coverage_fraction* (exclude vs. report), so the two
                       are not meant to share a value.
    subset_polygons  : forwarded to :func:`filter_scanning_path`, and also
                       intersected into the tissue the waste/coverage metrics
                       are measured against

    Returns
    -------
    :class:`GridOffsetResult` -- ``coords`` is the winner's ``(M, 2)`` path
    (closed if *return_side* was given, as in :func:`build_boundary_path`);
    ``candidates`` holds every offset's metrics.
    """
    effective_tissue = _effective_tissue(boundary_polygon, hole_polygons, subset_polygons)
    fov_area = fov_size_um * fov_size_um
    offsets = np.linspace(-step_size / 2.0, step_size / 2.0, n_samples, endpoint=False)

    candidates: List[GridOffsetCandidate] = []
    paths: Dict[Tuple[float, float], np.ndarray] = {}
    for dx in offsets:
        for dy in offsets:
            offset = (float(dx), float(dy))
            filtered = build_boundary_path(
                boundary_polygon, hole_polygons, step_size, fov_size_um,
                direction=direction, return_side=return_side,
                min_coverage_fraction=min_coverage_fraction,
                subset_polygons=subset_polygons, offset=offset,
            )
            paths[offset] = filtered

            boxes    = _fov_boxes(filtered, fov_size_um)
            overlaps = shapely.area(shapely.intersection(boxes, effective_tissue))
            waste    = shapely.area(boxes) - overlaps
            total_length_um, max_step_um = get_path_stats(filtered)
            candidates.append(GridOffsetCandidate(
                offset              = offset,
                n_fovs              = len(filtered),
                # cumsum = sequential sum, so ties rank exactly as a plain loop would
                waste_area_um2       = float(np.cumsum(waste)[-1]) if len(waste) else 0.0,
                total_length_um      = total_length_um,
                max_step_um          = max_step_um,
                n_low_coverage_fovs  = int(np.count_nonzero(overlaps / fov_area < low_coverage_fraction)),
            ))

    candidates.sort(key=lambda c: tuple(getattr(c, field) for field in priority))
    best = candidates[0]
    return GridOffsetResult(coords=paths[best.offset], best=best, candidates=candidates)


def build_boundary_path_optimized(
    boundary_polygon: Polygon,
    hole_polygons:    List[Polygon],
    step_size:        float,
    fov_size_um:      float,
    direction:        str             = "vertical",
    return_side:      Optional[str]   = None,
    n_samples:        int             = 9,
    priority:         Tuple[str, ...] = ("n_fovs", "waste_area_um2", "total_length_um", "n_low_coverage_fovs"),
    low_coverage_fraction: float      = 0.5,
    min_coverage_fraction: float      = 0.0,
    subset_polygons:  Optional[List[Polygon]] = None,
) -> np.ndarray:
    """
    Drop-in replacement for :func:`build_boundary_path` that additionally
    searches the grid's phase via :func:`optimize_grid_offset` -- same
    ``(M, 2)`` coordinate array contract, offset search parameters only.

    Use :func:`optimize_grid_offset` directly instead when the per-candidate
    diagnostic metrics are needed (e.g. to print/plot the evaluated spread).
    """
    result = optimize_grid_offset(
        boundary_polygon, hole_polygons, step_size, fov_size_um,
        direction=direction, return_side=return_side,
        n_samples=n_samples, priority=priority,
        low_coverage_fraction=low_coverage_fraction,
        min_coverage_fraction=min_coverage_fraction,
        subset_polygons=subset_polygons,
    )
    return result.coords


# ── Irregular (single-axis-adaptive) grid ──────────────────────────────────────
#
# Alternative to the regular grid above: the fixed axis stays on one lattice
# (so adjacent rows/columns share a phase and overlap), and the cross axis is
# rebuilt per fixed-axis position to fit that row/column's own tissue
# extent. Fewer wasted FOVs on irregular tissue, weaker cross-axis overlap.
# Validated against the regular grid in local (unshipped) test notebooks.
#
# Workflow:
#   1. build_irregular_bands            - cross-axis positions per band
#   2. fix_overlap_clusters             - REQUIRED per band before ordering
#   3. generate_irregular_scanning_path - boustrophedon order
#   4. filter_scanning_path             - same filter as the regular grid
#   5. patch_uncovered_gaps             - REQUIRED coverage guarantee
#   6. determine_return_side + close_scanning_path - optional loop closure
#   (build_irregular_boundary_path / optimize_irregular_grid wrap 1-6.)

def build_irregular_bands(
    boundary_polygon: Polygon,
    hole_polygons:    List[Polygon],
    step_size:        float,
    fov_size_um:      float,
    fixed_axis:       str   = "y",
    min_width_frac:   float = 0.1,
    fixed_offset:     float = 0.0,
    force_parity:     bool  = True,
) -> List[Tuple[float, np.ndarray]]:
    """
    Build per-band (fixed_axis regular-lattice) cross-axis position lists.

    One axis (*fixed_axis*) is a single regular lattice spanning the whole
    boundary's bounding box, built with :func:`spaced_coords` -- same as
    :func:`create_grid_positions`'s own traversal axis. For each fixed-axis
    lattice position, the tissue is intersected with a strip one FOV tall/
    wide centred on it; this can split into several disjoint pieces (e.g.
    either side of a hole) -- each piece gets its OWN cross-axis lattice,
    centred on and spanning just that piece's own extent
    (:func:`spaced_coords` with ``even=False``, one centred piece per
    contiguous strip of tissue).

    Returns a list of ``(fixed_value, cross_array_ascending)``, one entry
    per fixed-axis lattice position, in ASCENDING ``fixed_value`` order --
    the exact contract :func:`generate_irregular_scanning_path` expects. A
    band with no real tissue in its strip is still included, with an empty
    cross array (so band INDEX still lines up with the fixed-axis lattice
    position for that function's alternation logic; empty bands simply
    contribute no points).

    A band with multiple disjoint pieces can produce two adjacent points
    (the last of one piece, the first of the next) much closer together
    than *step_size* whenever the gap between pieces is narrower than
    *step_size* -- each piece's lattice is centred independently, with no
    awareness of its neighbour's phase. Call :func:`fix_overlap_clusters` on
    each band's cross array before building the scanning path to correct
    this (see that function's own docstring); it is not applied here so
    that this function's own output stays inspectable on its own (e.g. for
    diagnosing exactly which bands are affected).

    Parameters
    ----------
    boundary_polygon : the tissue boundary
    hole_polygons    : exclusion polygons
    step_size        : lattice spacing (µm) for the fixed axis and for each
                       piece's own cross-axis lattice
    fov_size_um      : camera FOV side length (µm) -- sets the strip height/
                       width and the minimum-piece-width cutoff below
    fixed_axis       : ``"y"`` (rows regular, columns/x adaptive per row) or
                       ``"x"`` (columns regular, rows/y adaptive per column)
    min_width_frac   : drop any local tissue piece narrower than
                       ``min_width_frac * fov_size_um`` along the cross axis
    fixed_offset     : shift the fixed axis's lattice phase away from the
                       boundary bbox's own midpoint (µm) -- see
                       :func:`optimize_irregular_grid`, which searches this
    force_parity     : force the fixed axis to an EVEN position count (the
                       same short-return-leg parity rule
                       :func:`create_grid_positions` applies to its own
                       traversal axis -- see its docstring). Default
                       ``True``; ``False`` uses the plain smallest-span-
                       covering count instead, whatever parity results.

    Returns
    -------
    List of ``(fixed_value, cross_array_ascending)``.
    """
    if fixed_axis not in ("x", "y"):
        raise ValueError("fixed_axis must be 'x' or 'y'")

    tissue = (boundary_polygon.difference(unary_union(hole_polygons))
              if hole_polygons else boundary_polygon)
    xmin, ymin, xmax, ymax = boundary_polygon.bounds
    half_h = fov_size_um / 2.0
    min_width_um = min_width_frac * fov_size_um

    if fixed_axis == "y":
        fixed_min, fixed_max, cross_min, cross_max = ymin, ymax, xmin, xmax
    else:
        fixed_min, fixed_max, cross_min, cross_max = xmin, xmax, ymin, ymax

    fixed_center = (fixed_min + fixed_max) / 2.0 + fixed_offset
    if force_parity:
        fixed_positions = spaced_coords(fixed_center, fixed_min, fixed_max, step_size, even=True)
    else:
        span = fixed_max - fixed_min
        n = max(1, int(np.ceil(span / step_size)))
        fixed_positions = fixed_center + (np.arange(n) - (n - 1) / 2.0) * step_size

    bands = []
    for f in fixed_positions:
        lo_f, hi_f = f - half_h, f + half_h
        strip = (shapely_box(cross_min - 1.0, lo_f, cross_max + 1.0, hi_f) if fixed_axis == "y"
                 else shapely_box(lo_f, cross_min - 1.0, hi_f, cross_max + 1.0))
        inter = tissue.intersection(strip)
        cross_vals: List[float] = []
        if not inter.is_empty:
            pieces = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
            pieces.sort(key=lambda p: p.bounds[0] if fixed_axis == "y" else p.bounds[1])
            for piece in pieces:
                if piece.is_empty:
                    continue
                pxmin, pymin, pxmax, pymax = piece.bounds
                lo, hi = (pxmin, pxmax) if fixed_axis == "y" else (pymin, pymax)
                if (hi - lo) < min_width_um:
                    continue
                piece_positions = spaced_coords((lo + hi) / 2.0, lo, hi, step_size, even=False)
                cross_vals.extend(piece_positions.tolist())
        bands.append((float(f), np.array(sorted(cross_vals))))
    return bands


def fix_overlap_clusters(
    cross_vals:       np.ndarray,
    fov_size_um:      float,
    step_size:        float,
    bad_overlap_frac: float = 0.3,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Re-space one band's cross-axis positions to remove overlap clusters at
    tissue-piece boundaries. REQUIRED after :func:`build_irregular_bands` and
    before :func:`generate_irregular_scanning_path`.

    When a band's tissue splits into disjoint pieces, each piece gets its own
    centred lattice, so the last FOV of one piece and the first of the next can
    sit much closer than *step_size* (up to ~full overlap on a real benchmark),
    while gaps within a piece are exactly *step_size*.

    A **sub-band** is a maximal run of points between the band's ends or a true
    gap (``overlap_frac <= 0``: footprints that don't touch, i.e. a real hole,
    never re-spaced across). If a sub-band has any gap with ``overlap_frac >
    bad_overlap_frac``, ALL its points are re-spaced evenly between its first
    and last position (moving only the two points at the bad gap does nothing
    for a 2-point cluster). Its point count becomes
    ``ceil(span / step_size) + 1``, so spacing never exceeds *step_size*.
    Use ``ceil``, not ``round``: rounding down leaves real uncovered tissue.

    A no-op on a band without bad gaps (e.g. a rectangle with no holes).

    Parameters
    ----------
    cross_vals       : one band's cross-axis positions (any order; sorted
                       internally), one entry of a :func:`build_irregular_bands`
                       result
    fov_size_um      : FOV side length (µm)
    step_size        : nominal lattice spacing (µm), used to judge bad gaps and
                       to size each re-spaced sub-band
    bad_overlap_frac : overlap fraction above which a gap is bad (default 0.3,
                       well above a typical ~0.10 design overlap)

    Returns
    -------
    (new_cross_vals, applied) : corrected, sorted positions, and
        ``(lo_idx, hi_idx, n_before, n_after)`` for each re-spaced sub-band
        (indices into the ORIGINAL sorted *cross_vals*); empty if none.
    """
    cross_vals = np.array(sorted(cross_vals), dtype=float)
    n = len(cross_vals)
    if n < 2:
        return cross_vals.copy(), []

    gaps = np.diff(cross_vals)
    overlap_frac = np.clip((fov_size_um - gaps) / fov_size_um, 0.0, None)
    disjoint = overlap_frac <= 0.0   # a TRUE gap -- never redistribute across this
    is_bad   = overlap_frac > bad_overlap_frac

    sub_ranges = []
    start = 0
    for i, d in enumerate(disjoint):
        if d:
            sub_ranges.append((start, i))
            start = i + 1
    sub_ranges.append((start, n - 1))

    fixed:   List[float] = []
    applied: List[Tuple[int, int, int, int]] = []
    for lo, hi in sub_ranges:
        seg_bad = is_bad[lo:hi] if hi > lo else np.array([], dtype=bool)
        if hi == lo or not seg_bad.any():
            fixed.extend(cross_vals[lo:hi + 1].tolist())
            continue
        span = cross_vals[hi] - cross_vals[lo]
        n_before = hi - lo + 1
        expected_n = max(2, int(np.ceil(span / step_size)) + 1)
        n_after = min(n_before, expected_n)
        fixed.extend(np.linspace(cross_vals[lo], cross_vals[hi], n_after).tolist())
        applied.append((lo, hi, n_before, n_after))
    return np.array(sorted(fixed)), applied


def generate_irregular_scanning_path(
    bands:      List[Tuple[float, np.ndarray]],
    fixed_axis: str,
) -> np.ndarray:
    """
    Order per-band cross-axis positions into a boustrophedon path.

    Generalises :func:`generate_scanning_path` to variable-length bands
    instead of a fixed-width ``(H, W)`` grid, replicating that function's
    own traversal/alternation loop structure exactly -- the two agree
    exactly whenever every band shares the same cross-axis positions (a
    degenerate rectangular tissue -- see the boustrophedon-return-path test
    notebook's own regression check).

    Parameters
    ----------
    bands      : from :func:`build_irregular_bands` (optionally passed
                through :func:`fix_overlap_clusters` first), in ASCENDING
                fixed-axis order
    fixed_axis : ``"y"`` (mirrors ``generate_scanning_path(direction=
                "horizontal")``) or ``"x"`` (mirrors ``direction="vertical"``)

    Returns
    -------
    ``(N, 2)`` array of ordered ``(x, y)`` stage coordinates.
    """
    if fixed_axis not in ("x", "y"):
        raise ValueError("fixed_axis must be 'x' or 'y'")

    path: List[Tuple[float, float]] = []
    n_bands = len(bands)
    if fixed_axis == "y":
        for strip, i in enumerate(range(n_bands - 1, -1, -1)):
            fixed_val, cross_vals = bands[i]
            ordered = cross_vals if strip % 2 == 0 else cross_vals[::-1]
            for c in ordered:
                path.append((c, fixed_val))
    else:
        for j in range(n_bands):
            fixed_val, cross_vals = bands[j]
            ordered = cross_vals[::-1] if j % 2 == 0 else cross_vals
            for c in ordered:
                path.append((fixed_val, c))
    return np.array(path) if path else np.empty((0, 2))


def patch_uncovered_gaps(
    coords:         np.ndarray,
    tissue_polygon: Polygon,
    step_size:      float,
    fov_size_um:    float,
    eps_um2:        float = 1.0,
    max_iters:      int   = 5,
) -> Tuple[np.ndarray, int]:
    """
    Add FOVs to close any tissue left uncovered by *coords*. REQUIRED after
    :func:`build_irregular_boundary_path`.

    A second line of defence behind :func:`fix_overlap_clusters`'s ``ceil``
    sizing: measure the uncovered area (*tissue_polygon* minus the union of all
    FOV squares), tile each uncovered piece's bounding box with
    *step_size*-spaced FOVs (same :func:`spaced_coords` pitch as the grid), and
    repeat up to *max_iters* times until nothing real is left. A fast no-op
    when coverage is already complete.

    The regular grid (:func:`build_boundary_path`/
    :func:`build_boundary_path_optimized`) doesn't need this:
    :func:`create_grid_positions` sizes each axis from the larger post-shift
    half-span, so a searched offset can't leave the far edge short.

    Parameters
    ----------
    coords         : ``(N, 2)`` FOV centres (µm)
    tissue_polygon : the region to cover, with holes (and any subset) already
                     applied -- the same "effective tissue" as
                     :func:`find_fully_redundant_fovs`
    step_size      : lattice spacing (µm) for the patch tiling
    fov_size_um    : FOV side length (µm)
    eps_um2        : residual area (µm²) treated as floating-point noise
    max_iters      : cap on patch/re-check passes

    Returns
    -------
    (patched_coords, n_added) : *coords* with any new FOVs appended at the end
        (never reordered or removed), and how many were added.
    """
    half    = fov_size_um / 2.0
    current = np.asarray(coords, dtype=float)
    added:  List[Tuple[float, float]] = []

    for _ in range(max_iters):
        boxes     = [shapely_box(x - half, y - half, x + half, y + half) for x, y in current]
        covered   = unary_union(boxes) if len(boxes) else None
        uncovered = tissue_polygon.difference(covered) if covered is not None else tissue_polygon
        if uncovered.is_empty or uncovered.area < eps_um2:
            break

        pieces  = list(uncovered.geoms) if hasattr(uncovered, "geoms") else [uncovered]
        new_pts: List[Tuple[float, float]] = []
        for piece in pieces:
            if piece.area < eps_um2:
                continue
            pxmin, pymin, pxmax, pymax = piece.bounds
            xs = spaced_coords((pxmin + pxmax) / 2.0, pxmin, pxmax, step_size, even=False)
            ys = spaced_coords((pymin + pymax) / 2.0, pymin, pymax, step_size, even=False)
            for x in xs:
                for y in ys:
                    fov = shapely_box(x - half, y - half, x + half, y + half)
                    if fov.intersects(piece):
                        new_pts.append((float(x), float(y)))

        if not new_pts:
            # Nothing constructed actually reaches the remaining gap (shouldn't
            # happen -- fov_size_um is normally far bigger than one gap piece's
            # own extent) -- stop rather than loop with no progress.
            break
        added.extend(new_pts)
        current = np.concatenate([current, np.array(new_pts)], axis=0)

    if not added:
        return np.asarray(coords, dtype=float), 0
    patched = np.concatenate([np.asarray(coords, dtype=float), np.array(added)], axis=0)
    return patched, len(added)


def build_irregular_boundary_path(
    boundary_polygon: Polygon,
    hole_polygons:    List[Polygon],
    step_size:        float,
    fov_size_um:      float,
    fixed_axis:       str            = "y",
    min_width_frac:   float          = 0.1,
    fixed_offset:     float          = 0.0,
    return_side:      Optional[str]  = "auto",
    apply_overlap_fix: bool          = True,
    bad_overlap_frac:  float         = 0.3,
    min_coverage_fraction: float     = 0.0,
    subset_polygons:  Optional[List[Polygon]] = None,
    guarantee_coverage: bool         = True,
    coverage_eps_um2: float          = 1.0,
) -> np.ndarray:
    """
    Build the ordered FOV path for a single-axis-adaptive irregular grid.

    Convenience wrapper mirroring :func:`build_boundary_path`'s pipeline:
    :func:`build_irregular_bands` -> :func:`fix_overlap_clusters` (per band,
    on by default -- see its docstring for why this is required, not
    optional) -> :func:`generate_irregular_scanning_path` ->
    :func:`filter_scanning_path` -> :func:`patch_uncovered_gaps` (on by
    default -- see its docstring for why this is required, not optional),
    optionally followed by :func:`determine_return_side` +
    :func:`close_scanning_path`.

    Parameters
    ----------
    boundary_polygon, hole_polygons, step_size, fov_size_um : as in
        :func:`build_irregular_bands`
    fixed_axis       : ``"y"`` or ``"x"`` -- see :func:`build_irregular_bands`
    min_width_frac, fixed_offset : forwarded to :func:`build_irregular_bands`
    return_side      : ``"auto"`` (default) picks the correct natural side
                       via :func:`determine_return_side` -- deliberately the
                       default rather than requiring an explicit guess,
                       since an earlier hardcoded guess in this method's own
                       validation notebook was measurably the WORSE side
                       (see that function's docstring). An explicit
                       ``"left"``/``"right"`` (``fixed_axis="y"``) or
                       ``"top"``/``"bottom"`` (``fixed_axis="x"``) closes on
                       that side directly; ``None`` skips closure entirely
                       (raw snake order), matching :func:`build_boundary_path`.
    apply_overlap_fix : apply :func:`fix_overlap_clusters` to every band
                       before path ordering (default ``True``) -- see that
                       function's docstring for why this matters.
    bad_overlap_frac : forwarded to :func:`fix_overlap_clusters`
    min_coverage_fraction, subset_polygons : forwarded to
                       :func:`filter_scanning_path`, same contract as
                       :func:`build_boundary_path`
    guarantee_coverage : apply :func:`patch_uncovered_gaps` against the
                       effective tissue (``boundary_polygon`` minus
                       ``hole_polygons``, intersected with
                       ``subset_polygons`` when given) after filtering
                       (default ``True``) -- see that function's docstring
                       for why this matters. Only applied when
                       ``min_coverage_fraction == 0.0``: a caller who set
                       ``min_coverage_fraction > 0`` is deliberately opting
                       into dropping real low-coverage tissue (see
                       :func:`filter_scanning_path`'s own docstring), and
                       patching would silently undo that choice.
    coverage_eps_um2 : forwarded to :func:`patch_uncovered_gaps` as
                       ``eps_um2``.

    Returns
    -------
    ``(M, 2)`` ordered stage coordinates.
    """
    bands = build_irregular_bands(
        boundary_polygon, hole_polygons, step_size, fov_size_um,
        fixed_axis=fixed_axis, min_width_frac=min_width_frac, fixed_offset=fixed_offset,
    )
    if apply_overlap_fix:
        bands = [
            (fixed_val, fix_overlap_clusters(cross_vals, fov_size_um, step_size, bad_overlap_frac)[0])
            for fixed_val, cross_vals in bands
        ]
    path = generate_irregular_scanning_path(bands, fixed_axis=fixed_axis)
    if len(path) == 0:
        return path
    filtered = filter_scanning_path(path, boundary_polygon, hole_polygons, fov_size_um,
                                     min_coverage_fraction=min_coverage_fraction,
                                     subset_polygons=subset_polygons)
    if guarantee_coverage and min_coverage_fraction == 0.0 and len(filtered) > 0:
        effective_tissue = (boundary_polygon.difference(unary_union(hole_polygons))
                             if hole_polygons else boundary_polygon)
        if subset_polygons:
            effective_tissue = effective_tissue.intersection(unary_union(subset_polygons))
        filtered, _ = patch_uncovered_gaps(filtered, effective_tissue, step_size, fov_size_um,
                                            eps_um2=coverage_eps_um2)
    if return_side is not None and len(filtered) > 1:
        if return_side == "auto":
            return_side, _ = determine_return_side(filtered, fixed_axis, step_size)
        filtered, _ = close_scanning_path(filtered, step_size, return_side=return_side)
    return filtered


@dataclass
class IrregularGridOffsetResult:
    """Winner (by fewest FOVs) plus every fixed-axis offset evaluated."""
    coords:            np.ndarray
    best_offset:       float
    n_fovs_per_offset: List[Tuple[float, int]]


def optimize_irregular_grid(
    boundary_polygon: Polygon,
    hole_polygons:    List[Polygon],
    step_size:        float,
    fov_size_um:      float,
    fixed_axis:       str            = "y",
    min_width_frac:   float          = 0.1,
    n_samples:        int            = 9,
    return_side:      Optional[str]  = "auto",
    apply_overlap_fix: bool          = True,
    bad_overlap_frac:  float         = 0.3,
    min_coverage_fraction: float     = 0.0,
    subset_polygons:  Optional[List[Polygon]] = None,
    guarantee_coverage: bool         = True,
    coverage_eps_um2: float          = 1.0,
) -> IrregularGridOffsetResult:
    """
    Search the fixed axis's lattice phase (offset within one *step_size*
    period) for the one needing the fewest FOVs -- the irregular-grid
    analogue of :func:`optimize_grid_offset`'s offset search, but for the
    ONE axis this grid still shares a single phase across (the cross axis
    is already independently re-centred per band, so it has nothing to
    search).

    Only *fixed_offset* varies across candidates; every other
    :func:`build_irregular_boundary_path` parameter stays fixed. Offsets a
    full *step_size* apart reproduce the same lattice, so
    ``[-step_size/2, step_size/2)`` covers every distinct phase.

    Parameters
    ----------
    n_samples : candidate offsets evaluated (:func:`optimize_grid_offset`'s
               own default, 9, reused here)
    (all other parameters forwarded to :func:`build_irregular_boundary_path`)

    Returns
    -------
    :class:`IrregularGridOffsetResult` -- ``coords`` is the winning
    candidate's ``(M, 2)`` path (already closed if *return_side* was
    given); ``n_fovs_per_offset`` holds every evaluated offset's FOV count.
    """
    offsets = np.linspace(-step_size / 2.0, step_size / 2.0, n_samples, endpoint=False)
    best_coords, best_offset, best_n = None, 0.0, None
    n_fovs_per_offset: List[Tuple[float, int]] = []
    for off in offsets:
        coords = build_irregular_boundary_path(
            boundary_polygon, hole_polygons, step_size, fov_size_um,
            fixed_axis=fixed_axis, min_width_frac=min_width_frac, fixed_offset=float(off),
            return_side=return_side, apply_overlap_fix=apply_overlap_fix,
            bad_overlap_frac=bad_overlap_frac, min_coverage_fraction=min_coverage_fraction,
            subset_polygons=subset_polygons,
            guarantee_coverage=guarantee_coverage, coverage_eps_um2=coverage_eps_um2,
        )
        n_fovs_per_offset.append((float(off), len(coords)))
        if best_n is None or len(coords) < best_n:
            best_coords, best_offset, best_n = coords, float(off), len(coords)
    return IrregularGridOffsetResult(coords=best_coords, best_offset=best_offset,
                                      n_fovs_per_offset=n_fovs_per_offset)


# ── Redundant-FOV removal ──────────────────────────────────────────────────────

@dataclass
class RedundantFOVResult:
    """Result of :func:`find_fully_redundant_fovs`."""
    detected:           List[int]    # indices flagged at least once as fully redundant
    removed:            Set[int]     # indices verified safe to drop together
    kept_ids:           List[int]    # indices of *coords* NOT removed, in original order
    kept_coords:        np.ndarray   # (M, 2) -- coords[kept_ids]
    n_passes:           int          # iterative fixed-point passes taken to converge
    uncovered_area_um2: float        # tissue area left uncovered after removing `removed`


def find_fully_redundant_fovs(
    coords:         np.ndarray,
    tissue_polygon: Polygon,
    fov_size_um:    float,
    eps_um2:        float = 1.0,
) -> RedundantFOVResult:
    """
    Find FOVs in *coords* whose entire tissue overlap is already duplicated
    by some other FOV, and verify they can all be dropped together.

    A FOV is "fully redundant" when subtracting every OTHER FOV's own square
    from its own tissue-intersection region leaves nothing real (area below
    *eps_um2*, a noise floor for floating-point-residual polygon slivers):
    everywhere it touches tissue, some neighbor already reaches the same
    tissue, so dropping it alone loses no coverage. This shows up in
    practice at narrow tissue tips/corners, where a diagonally-adjacent
    FOV's own square already reaches the small sliver of tissue the flagged
    FOV clips.

    Two FOVs can be redundant only *with each other* though (removing
    either alone is safe; removing both isn't, since each was only
    "covered" by the other) -- a single pass over the initially-detected
    set can't see that. This re-checks each still-flagged FOV against the
    *currently remaining* grid repeatedly until no more can be dropped (a
    fixed point), then verifies the final removal leaves no real tissue
    uncovered (`uncovered_area_um2`, checked against the same *eps_um2*
    floor).

    Uses an `STRtree` so each FOV only checks the handful of geometric
    neighbors whose square could possibly overlap it (axis-aligned squares
    are their own bounding box, so an ``"intersects"`` query against the
    tree is already exact -- no buffering needed) instead of unioning
    against every other FOV.

    Parameters
    ----------
    coords         : ``(N, 2)`` FOV center coordinates (µm)
    tissue_polygon : the tissue region FOVs are meant to cover (a boundary
                     polygon with holes already subtracted, i.e. the same
                     "effective tissue" *not* the raw boundary)
    fov_size_um    : camera FOV side length (µm) -- each FOV is modelled as
                     a square of this size centred at its coordinate
    eps_um2        : area (µm²) below which a residual is treated as
                     floating-point noise rather than real tissue

    Returns
    -------
    :class:`RedundantFOVResult`
    """
    coords = np.asarray(coords, dtype=float)
    half   = fov_size_um / 2.0
    boxes  = [shapely_box(x - half, y - half, x + half, y + half) for x, y in coords]
    tree   = STRtree(boxes)
    n      = len(boxes)

    # Each FOV's real geometric neighbors (fixed -- doesn't depend on which
    # FOVs end up removed later), found once up front.
    neighbor_idx: List[List[int]] = []
    for i, b in enumerate(boxes):
        idx = [int(j) for j in tree.query(b, predicate="intersects") if j != i]
        neighbor_idx.append(idx)

    def exclusive_area(i: int, excluded: frozenset = frozenset()) -> float:
        own_overlap = boxes[i].intersection(tissue_polygon)
        if own_overlap.is_empty:
            return 0.0
        others = [boxes[j] for j in neighbor_idx[i] if j not in excluded]
        if not others:
            return own_overlap.area
        return own_overlap.difference(unary_union(others)).area

    detected = [i for i in range(n) if exclusive_area(i) < eps_um2]

    removed: Set[int] = set()
    n_passes = 0
    changed  = True
    while changed:
        changed = False
        n_passes += 1
        for i in detected:
            if i in removed:
                continue
            if exclusive_area(i, excluded=removed) < eps_um2:
                removed.add(i)
                changed = True

    kept_ids    = [i for i in range(n) if i not in removed]
    kept_coords = coords[kept_ids] if kept_ids else coords[:0]
    kept_union  = unary_union([boxes[i] for i in kept_ids]) if kept_ids else None
    uncovered   = tissue_polygon.difference(kept_union) if kept_union is not None else tissue_polygon
    uncovered_area_um2 = 0.0 if uncovered.is_empty else uncovered.area

    return RedundantFOVResult(
        detected=detected, removed=removed, kept_ids=kept_ids,
        kept_coords=kept_coords, n_passes=n_passes,
        uncovered_area_um2=uncovered_area_um2,
    )


# ── Single-call FOV path builder ────────────────────────────────────────────────

@dataclass
class ReducedFOVPathResult:
    """Result of :func:`build_reduced_fov_path`."""
    coords:                  np.ndarray          # (M, 2) -- final path; redundant FOVs already dropped
                                                  # unless remove_redundant_fovs=False, then == the raw grid
    n_fovs_before_redundant: int                 # grid size before redundant-FOV removal
    redundant:               RedundantFOVResult  # full find_fully_redundant_fovs() detail -- always
                                                  # computed, even when remove_redundant_fovs=False


def build_reduced_fov_path(
    boundary_polygon: Polygon,
    hole_polygons:    List[Polygon],
    step_size:        float,
    fov_size_um:      float,
    irregular_grid:   bool             = False,
    optimize_offset:  bool             = False,
    direction:        str              = "vertical",
    fixed_axis:       str              = "y",
    return_side:      Optional[str]    = None,
    n_samples:        int              = 9,
    min_coverage_fraction: float       = 0.0,
    subset_polygons:  Optional[List[Polygon]] = None,
    eps_um2:          float            = 1.0,
    remove_redundant_fovs: bool        = True,
) -> ReducedFOVPathResult:
    """
    Build one boundary's final FOV path in a single call: a grid builder
    (picked by *irregular_grid* x *optimize_offset* from
    :func:`build_boundary_path`, :func:`build_boundary_path_optimized`,
    :func:`build_irregular_boundary_path`, :func:`optimize_irregular_grid`),
    then :func:`find_fully_redundant_fovs` (drop a FOV only if all the tissue
    it covers is also covered by other FOVs).

    The redundancy detection always runs (``result.redundant`` reports what
    could be dropped and the uncovered area). *remove_redundant_fovs* decides
    whether the drop is applied. Turn it off to compare *optimize_offset*
    settings fairly: both methods remove the same wasted tip/corner FOVs, so
    with removal on they can look no different.

    Parameters
    ----------
    boundary_polygon, hole_polygons, step_size, fov_size_um : as in
        :func:`build_boundary_path`
    irregular_grid   : False (default) = one regular lattice; True = a
                       single-axis-adaptive grid (see the "Irregular grid" section
                       comment for the tradeoff)
    optimize_offset  : True also searches the grid phase for the fewest FOVs
    direction        : boustrophedon direction (regular grid only)
    fixed_axis       : ``"y"`` or ``"x"`` (irregular grid only; see
                       :func:`build_irregular_bands`)
    return_side      : forwarded to the path builder; only reorders the path
                       (:func:`close_scanning_path`), never changes the count
    n_samples        : candidate offsets (only with ``optimize_offset=True``)
    min_coverage_fraction, subset_polygons : forwarded to the path builder, as
                       in :func:`build_boundary_path`
    eps_um2          : forwarded to :func:`find_fully_redundant_fovs`
    remove_redundant_fovs : True (default) drops the redundant FOVs; False
                       returns the full grid as ``coords``

    Returns
    -------
    :class:`ReducedFOVPathResult`.
    """
    if irregular_grid:
        if optimize_offset:
            opt  = optimize_irregular_grid(
                boundary_polygon, hole_polygons, step_size, fov_size_um,
                fixed_axis=fixed_axis, n_samples=n_samples,
                return_side=return_side, min_coverage_fraction=min_coverage_fraction,
                subset_polygons=subset_polygons,
            )
            path = opt.coords
        else:
            path = build_irregular_boundary_path(
                boundary_polygon, hole_polygons, step_size, fov_size_um,
                fixed_axis=fixed_axis, return_side=return_side,
                min_coverage_fraction=min_coverage_fraction, subset_polygons=subset_polygons,
            )
    else:
        if optimize_offset:
            path = build_boundary_path_optimized(
                boundary_polygon, hole_polygons, step_size, fov_size_um,
                direction=direction, return_side=return_side, n_samples=n_samples,
                min_coverage_fraction=min_coverage_fraction, subset_polygons=subset_polygons,
            )
        else:
            path = build_boundary_path(
                boundary_polygon, hole_polygons, step_size, fov_size_um,
                direction=direction, return_side=return_side,
                min_coverage_fraction=min_coverage_fraction, subset_polygons=subset_polygons,
            )

    effective_tissue = boundary_polygon.difference(unary_union(hole_polygons)) if hole_polygons else boundary_polygon
    if subset_polygons:
        effective_tissue = effective_tissue.intersection(unary_union(subset_polygons))

    redundant = find_fully_redundant_fovs(path, effective_tissue, fov_size_um, eps_um2=eps_um2)
    coords = redundant.kept_coords if remove_redundant_fovs else np.asarray(path, dtype=float)
    return ReducedFOVPathResult(
        coords=coords, n_fovs_before_redundant=len(path), redundant=redundant,
    )