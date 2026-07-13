# MERci/acquisition/positions.py
"""
Generate FOV grids and optimised scanning paths for stage-based MERFISH imaging.

Typical workflow
----------------
1. Define the tissue boundary and any excluded regions (holes).
2. ``create_grid_positions`` – build a regular ``(H, W, 2)`` grid with an odd
   number of rows and columns, centred on the midpoint of the boundary bounding box.
3. ``generate_scanning_path`` – order grid points in a boustrophedon pattern.
4. ``load_hole_polygons`` – load polygon masks for excluded areas.
5. ``filter_scanning_path`` – keep FOVs whose camera frame overlaps the boundary;
   remove FOVs whose camera frame overlaps any hole.
6. ``close_scanning_path`` – move the "return" points to the end of the path.
7. ``get_path_stats`` – inspect total travel distance and largest single step.
8. ``MERci.common.io.save_positions_array`` – write positions.txt.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, box as shapely_box

log = logging.getLogger(__name__)


# ── Grid construction ─────────────────────────────────────────────────────────

def _odd_1d_coords(center: float, d_min: float, d_max: float, step: float) -> np.ndarray:
    """
    Return evenly-spaced 1-D coordinates centred exactly on *center*.

    The count is the smallest odd number such that the coordinates span at
    least [d_min, d_max].
    """
    span   = d_max - d_min
    n      = int(np.ceil(span / step))
    if n % 2 == 0:
        n += 1                          # ensure odd
    n_half = (n - 1) // 2
    return center + np.arange(-n_half, n_half + 1) * step


def create_grid_positions(
    boundary_polygon: Polygon,
    step_size:        float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a regular 2-D grid centred on the midpoint of *boundary_polygon*'s
    bounding box.

    The grid has an odd number of rows (H) and columns (W), so there is always
    one cell exactly at the centre.  The grid is large enough to cover the full
    bounding box of the boundary.

    Parameters
    ----------
    boundary_polygon : Shapely Polygon of the tissue boundary
    step_size        : distance between adjacent grid points (µm)

    Returns
    -------
    grid : ``(H, W, 2)`` array of ``(x, y)`` coordinates
    xs   : 1-D x-coordinates (length W, odd)
    ys   : 1-D y-coordinates (length H, odd)
    """
    xmin, ymin, xmax, ymax = boundary_polygon.bounds
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    xs = _odd_1d_coords(cx, xmin, xmax, step_size)
    ys = _odd_1d_coords(cy, ymin, ymax, step_size)

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
    with path.open() as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) >= 2:
                try:
                    coords.append((float(row[0]), float(row[1])))
                except ValueError:
                    pass   # skip header-like lines
    return coords


def load_hole_polygons(
    hole_dir: Path,
    pattern:  str = "hole*.txt",
) -> List[Polygon]:
    """
    Load exclusion-region polygons from a directory.

    Each ``hole{n}.txt`` file matching *pattern* must be a comma-separated
    ``x,y`` file (one vertex per line); files with fewer than three vertices
    are skipped. A hole can optionally have one or more companion
    ``hole{n}_island{m}.txt`` files (same format) -- these become interior
    rings of the hole polygon, for a hole that is really a donut/annulus
    around genuine tissue (e.g. auto-derived by
    ``MERci.acquisition.mosaic.segment_mosaic_tissue``/``save_boundary_from
    _mosaic``): the island area is then correctly excluded *from* the hole
    (i.e. still imaged), rather than swallowed whole into a solid disk.
    Island companion files are never treated as holes in their own right.

    Returns
    -------
    List of valid Shapely :class:`~shapely.geometry.Polygon` objects (with
    interior rings where island companion files were found).
    """
    hole_dir = Path(hole_dir)
    hole_re  = re.compile(r"^hole(\d+)\.txt$", re.IGNORECASE)
    polygons: List[Polygon] = []

    for path in sorted(hole_dir.glob(pattern)):
        m = hole_re.match(path.name)
        if not m:
            continue   # e.g. a hole{n}_island{m}.txt companion file -- not its own hole
        hole_id = m.group(1)

        coords = _read_xy_file(path)
        if len(coords) < 3:
            log.warning(
                "%s has fewer than 3 valid points – skipping.", path.name
            )
            continue

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

        poly = Polygon(coords, islands) if islands else Polygon(coords)
        if poly.is_empty or not poly.is_valid:
            log.warning(
                "%s produced an invalid Shapely polygon – skipping.", path.name
            )
            continue

        polygons.append(poly)

    return polygons


# ── Path filtering ─────────────────────────────────────────────────────────────

def filter_scanning_path(
    coords:           np.ndarray,
    boundary_polygon: Polygon,
    hole_polygons:    List[Polygon],
    fov_size_um:      float,
) -> np.ndarray:
    """
    Keep FOVs whose camera frame overlaps the tissue boundary; exclude any
    that are fully contained within a hole region.

    Each FOV is modelled as a square of side *fov_size_um* centred at its
    stage coordinate (``pixel_size_um × image_size_px``).  A FOV is kept when:

    * its square has **any** overlap with *boundary_polygon*, **and**
    * no hole polygon **fully contains** its square.

    A FOV that only partially overlaps a hole is kept — it still captures
    tissue outside the hole.

    Parameters
    ----------
    coords           : ``(N, 2)`` candidate stage coordinates
    boundary_polygon : outer tissue boundary
    hole_polygons    : list of Shapely Polygons to exclude
    fov_size_um      : camera FOV side length in stage units
                       (``pixel_size_um × image_size_px``)

    Returns
    -------
    ``(M, 2)`` array of accepted coordinates in their original order.
    """
    coords = np.asarray(coords, dtype=float)
    half   = fov_size_um / 2.0

    kept = []
    for x, y in coords:
        fov_poly = shapely_box(x - half, y - half, x + half, y + half)
        if not fov_poly.intersects(boundary_polygon):
            continue
        if any(hole.contains(fov_poly) for hole in hole_polygons):
            continue
        kept.append((x, y))

    return np.array(kept) if kept else np.empty((0, 2))


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

    multi_re  = re.compile(r"^tissue_(\d+)_boundary_positions_(\d+)\.txt$", re.IGNORECASE)
    single_re = re.compile(r"^boundary_positions_(\d+)\.txt$", re.IGNORECASE)

    multi:  List[BoundarySpec] = []
    single: List[BoundarySpec] = []
    for p in sorted(positions_dir.glob("*.txt")):
        m = multi_re.match(p.name)
        if m:
            t, b = int(m.group(1)), int(m.group(2))
            multi.append(BoundarySpec(t, b, p, f"T{t}B{b}"))
            continue
        s = single_re.match(p.name)
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


def has_boundary_files(positions_dir: Path) -> bool:
    """Return ``True`` if *positions_dir* holds boundary files of any layout.

    Checks for the multi (``tissue_<t>_boundary_positions_<b>.txt``), single
    (``boundary_positions_<b>.txt``) or legacy (``boundary_positions.txt``)
    naming — i.e. whether :func:`discover_boundary_files` would succeed.
    """
    positions_dir = Path(positions_dir)
    if not positions_dir.is_dir():
        return False
    multi_re  = re.compile(r"^tissue_\d+_boundary_positions_\d+\.txt$", re.IGNORECASE)
    single_re = re.compile(r"^boundary_positions_\d+\.txt$", re.IGNORECASE)
    for p in positions_dir.glob("*.txt"):
        if multi_re.match(p.name) or single_re.match(p.name):
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
    coords: List[Tuple[float, float]] = []
    with path.open() as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                try:
                    coords.append((float(row[0]), float(row[1])))
                except ValueError:
                    pass  # skip header/comment lines
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
) -> np.ndarray:
    """
    Build the ordered FOV path for a single boundary.

    Convenience wrapper that runs the standard per-boundary pipeline:
    :func:`create_grid_positions` → :func:`generate_scanning_path` →
    :func:`filter_scanning_path`, optionally followed by
    :func:`close_scanning_path`.

    Parameters
    ----------
    boundary_polygon : the tissue boundary for this segment
    hole_polygons    : exclusion polygons (applied to this boundary)
    step_size        : grid spacing (µm)
    fov_size_um      : camera FOV side length (µm)
    direction        : boustrophedon direction for
                       :func:`generate_scanning_path`
    return_side      : if given, :func:`close_scanning_path` moves that side's
                       points to the end; if ``None`` (default) the raw snake
                       order is kept — preferred in the multi-boundary layout,
                       where the transit segments handle travel between boundaries

    Returns
    -------
    ``(M, 2)`` ordered stage coordinates for this boundary.
    """
    grid, _, _ = create_grid_positions(boundary_polygon, step_size)
    path       = generate_scanning_path(grid, direction=direction)
    filtered   = filter_scanning_path(path, boundary_polygon, hole_polygons, fov_size_um)
    if return_side is not None and len(filtered) > 1:
        filtered, _ = close_scanning_path(filtered, step_size, return_side=return_side)
    return filtered