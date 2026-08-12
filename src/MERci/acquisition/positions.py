# MERci/acquisition/positions.py
"""
Generate FOV grids and optimised scanning paths for stage-based MERFISH imaging.

Typical workflow
----------------
1. Define the tissue boundary and any excluded regions (holes).
2. ``create_grid_positions`` – build a regular ``(H, W, 2)`` grid centred on the
   midpoint of the boundary bounding box; the traversal axis (columns for
   direction="vertical", rows for "horizontal") is forced even for a short
   scan-path return leg, the other axis odd for a centred cell (see its docstring).
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
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union

log = logging.getLogger(__name__)


# ── Grid construction ─────────────────────────────────────────────────────────

def _spaced_coords(
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

    if direction == "vertical":
        xs = _spaced_coords(cx, xmin, xmax, step_size, even=True)
        ys = _spaced_coords(cy, ymin, ymax, step_size, even=False)
    elif direction == "horizontal":
        xs = _spaced_coords(cx, xmin, xmax, step_size, even=False)
        ys = _spaced_coords(cy, ymin, ymax, step_size, even=True)
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


# ── Exterior-FOV detection ────────────────────────────────────────────────────

def find_exterior_fovs(
    positions:          Dict[int, Tuple[float, float]],
    step_size:           float,
    connectivity:        str   = "8",
    tolerance_fraction:  float = 0.25,
) -> Set[int]:
    """
    Find FOVs on the exterior of an imaged FOV grid -- the true outer
    perimeter of the imaged footprint, AND the inner boundary of any holes --
    i.e. any FOV with at least one grid-adjacent neighbour position that is
    NOT actually imaged.

    Queries real stage coordinates directly via a KD-tree rather than
    snapping every position onto one shared integer ``(row, col)`` grid
    index (as the private :func:`_grid_indices` above does): each tissue
    boundary/piece in a multi-boundary layout gets its own FOV grid centred
    on that piece's own bounding-box midpoint (see
    :func:`create_grid_positions`), so different pieces' grids are not in
    general phase-aligned with each other. A single shared grid-index snap
    would risk misjudging adjacency exactly at a tissue-piece boundary;
    testing "is there a real FOV near this exact candidate neighbour
    position" is correct regardless of any other piece's grid phase, hole
    geometry, or nearby transit-point irregularity -- so multi-tissue/hole
    layouts are handled for free, with no per-tissue-piece logic needed.

    Parameters
    ----------
    positions          : {fov_id: (x, y)} stage coordinates (µm) of the FOVs
                         to test -- scope this to one round's own real
                         imaged FOVs (not the raw experiment-wide
                         positions.txt), so transit-only FOVs (blank frames)
                         never enter the result.
    step_size          : grid step size (µm), e.g. ``ExperimentConfig.step_size_um``
    connectivity       : "4" (N/S/E/W neighbours only) or "8" (+ diagonals).
                         "8" (default) also catches FOVs at diagonal-only
                         tissue notches / concave corners / hole-island
                         corners that "4" would miss.
    tolerance_fraction : match tolerance for "is a neighbour actually
                         present", as a fraction of step_size -- absorbs
                         small positioning jitter without over-matching to a
                         FOV that is really one grid cell further away.

    Returns
    -------
    Set of FOV ids that are exterior (their FFC-estimation candidates).
    """
    from scipy.spatial import KDTree

    if connectivity not in ("4", "8"):
        raise ValueError(f"connectivity must be '4' or '8', got {connectivity!r}")

    fov_ids = list(positions.keys())
    coords  = np.array([positions[f] for f in fov_ids], dtype=float)
    if len(fov_ids) == 0:
        return set()

    tree = KDTree(coords)
    tol  = tolerance_fraction * step_size

    offsets = [(step_size, 0.0), (-step_size, 0.0), (0.0, step_size), (0.0, -step_size)]
    if connectivity == "8":
        offsets += [
            (step_size, step_size), (step_size, -step_size),
            (-step_size, step_size), (-step_size, -step_size),
        ]

    exterior: Set[int] = set()
    for fov_id, (x, y) in zip(fov_ids, coords):
        for dx, dy in offsets:
            dist, _ = tree.query([x + dx, y + dy])
            if dist > tol:
                exterior.add(fov_id)
                break

    return exterior


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
    from scipy.spatial import KDTree

    offsets = {
        "right": (step_size, 0.0), "left": (-step_size, 0.0),
        "up":    (0.0, step_size), "down": (0.0, -step_size),
    }
    if direction not in offsets:
        raise ValueError(f"direction must be one of {list(offsets)}, got {direction!r}")
    if fov_id not in positions:
        raise KeyError(f"fov_id {fov_id} not in positions.")

    fov_ids = list(positions.keys())
    coords  = np.array([positions[f] for f in fov_ids], dtype=float)
    tree    = KDTree(coords)

    x, y   = positions[fov_id]
    dx, dy = offsets[direction]
    dist, idx = tree.query([x + dx, y + dy])
    if dist > tolerance_fraction * step_size:
        return None
    return fov_ids[int(idx)]


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
    Group consecutive same-tissue boundaries into acquisition-order
    "boundary" segments, honouring each tissue's own path mode.

    A tissue's own boundaries (a contiguous run in *boundaries*, since it is
    sorted by tissue then boundary -- see :func:`discover_boundary_files`)
    are merged into ONE segment when ``tissue_path_mode(tissue) == "legacy"``
    and it has more than one boundary; otherwise (``"transit"``, or a tissue
    with only one boundary) each boundary keeps its own segment.

    This is the single source of truth for that grouping decision, shared by
    ``02_create_positions_from_boundaries.ipynb`` (which attaches the actual
    FOV coordinates per segment) and
    :func:`MERci.acquisition.dave.create_round_info_multitissue` (which only
    needs the resulting segment/label structure to build ``round_info.csv``
    rows matching whatever notebook 02 actually wrote to ``positions/``) --
    so the two can never disagree about which positions files exist.

    This function does NOT add transit segments between the groups it
    returns -- callers insert those uniformly (bridging every consecutive
    pair of the returned groups, wrapping the last back to the first,
    whenever more than one group is returned), since that part doesn't
    depend on the per-tissue path mode.

    Parameters
    ----------
    boundaries       : from :func:`discover_boundary_files`
    mode             : ``"multi"``, ``"single"`` or ``"legacy"`` (from the
                       same discovery call) -- selects the merged-segment
                       label (``"T{t}"`` for multi, ``""`` otherwise)
    tissue_path_mode : tissue index -> ``"legacy"`` or ``"transit"``

    Returns
    -------
    List of :class:`BoundaryGroup`, in acquisition order.
    """
    n = len(boundaries)
    groups: List[BoundaryGroup] = []
    i = 0
    while i < n:
        spec, t = boundaries[i], boundaries[i].tissue
        j = i
        while j < n and boundaries[j].tissue == t:
            j += 1
        run_len = j - i

        if tissue_path_mode(t) == "legacy" and run_len > 1:
            label = f"T{t}" if mode == "multi" else ""
            groups.append(BoundaryGroup(t, label, tuple(range(i, j))))
        elif run_len > 1:   # tissue_path_mode(t) == "transit"
            for k in range(i, j):
                groups.append(BoundaryGroup(t, boundaries[k].label, (k,)))
        else:                # run_len == 1: nothing of this tissue's own to merge
            groups.append(BoundaryGroup(t, spec.label, (i,)))
        i = j
    return groups


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
    grid, _, _ = create_grid_positions(boundary_polygon, step_size, direction=direction)
    path       = generate_scanning_path(grid, direction=direction)
    filtered   = filter_scanning_path(path, boundary_polygon, hole_polygons, fov_size_um)
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
) -> GridOffsetResult:
    """
    Search the grid's phase (offset within one *step_size* period) for the
    one that best minimises the count of near-empty FOVs, wasted (non-tissue)
    imaged area, scan travel length, and FOV count -- without touching the
    hard parity constraint :func:`create_grid_positions` already enforces for
    a short return leg.

    Only the grid's *offset* varies across candidates; *step_size* and
    *direction* stay fixed (per the parity rule they'd otherwise break), so
    this is a phase search, not a redesign of the grid itself. Offsets a
    full *step_size* apart reproduce the same lattice against a polygon
    fixed in space, so ``[-step_size/2, step_size/2)`` in each axis covers
    every distinct phase.

    ``n_low_coverage_fovs`` (a FOV whose camera square overlaps
    *effective_tissue* by less than *low_coverage_fraction* of its own area
    -- :func:`filter_scanning_path`'s own keep rule is coverage-blind, so
    these near-empty FOVs already survive filtering) is a genuinely distinct
    objective from ``waste_area_um2``: per-FOV acquisition time is roughly
    fixed regardless of how much tissue a FOV actually contains, so a
    count-based objective targets wasted imaging *time* directly, while
    ``waste_area_um2`` (a continuous area sum, dominated by whichever FOVs
    happen to be biggest/most wasteful) targets wasted *area* -- related but
    not the same, and a phase that minimises one need not minimise the
    other.

    Default *priority* leads with ``n_fovs`` itself -- the simplest, most
    direct proxy for total imaging time (every FOV costs roughly the same
    fixed acquisition time regardless of content), chosen deliberately over
    leading with ``n_low_coverage_fovs`` after a full sweep of every
    priority ordering on a real benchmark
    (``notebooks/tests/compare_fov_coverage_constraint.ipynb``) showed the
    two only trade off by a handful of FOVs either way (e.g. 1152 FOVs/214
    low-coverage vs. 1161 FOVs/209 low-coverage on that benchmark) -- no
    ordering dominates the other, so the simpler, more directly-motivated
    objective was preferred.

    Parameters
    ----------
    boundary_polygon : the tissue boundary for this segment
    hole_polygons    : exclusion polygons (applied to this boundary) -- also
                       subtracted from the boundary when computing wasted
                       area/coverage below, since a hole-covered pixel is
                       just as much non-tissue as one outside the boundary
                       entirely (unlike :func:`filter_scanning_path`, which
                       only drops FOVs a hole *fully* contains -- a
                       partially hole-overlapping FOV survives filtering,
                       and its hole-covered area still counts as waste/
                       low-coverage here)
    step_size        : grid spacing (µm)
    fov_size_um      : camera FOV side length (µm)
    direction        : boustrophedon direction, forwarded to
                       :func:`create_grid_positions`/:func:`generate_scanning_path`
    return_side      : forwarded to :func:`close_scanning_path`, applied
                       before each candidate's travel length is measured, so
                       the reported length matches what the caller will
                       actually use
    n_samples        : candidate offsets per axis (``n_samples**2`` total
                       evaluated) -- odd values include the un-shifted
                       ``(0, 0)`` centred grid as one candidate
    priority         : ``GridOffsetCandidate`` field names, most important
                       first, used to lexicographically rank candidates
                       (each minimised). Default: FOV count first (the
                       simplest direct proxy for imaging time), then wasted
                       area, then travel length, then low-coverage FOV count.
    low_coverage_fraction : a FOV counts as "low coverage" when its own
                       tissue-overlap fraction (tissue-overlap area / FOV
                       area) is strictly below this threshold. Default 0.5
                       (less than half the FOV is real tissue).

    Returns
    -------
    :class:`GridOffsetResult` -- ``coords`` is the winning candidate's
    ``(M, 2)`` path (already closed if *return_side* was given, matching
    :func:`build_boundary_path`'s contract); ``candidates`` holds every
    evaluated offset's metrics, so a caller can inspect the actual spread
    (e.g. to see whether the objectives move together or trade off against
    each other on this particular boundary).
    """
    if hole_polygons:
        effective_tissue = boundary_polygon.difference(unary_union(hole_polygons))
    else:
        effective_tissue = boundary_polygon

    half   = fov_size_um / 2.0
    fov_area = fov_size_um * fov_size_um
    offsets = np.linspace(-step_size / 2.0, step_size / 2.0, n_samples, endpoint=False)

    candidates: List[GridOffsetCandidate] = []
    for dx in offsets:
        for dy in offsets:
            grid, _, _ = create_grid_positions(
                boundary_polygon, step_size, direction=direction, offset=(dx, dy),
            )
            path     = generate_scanning_path(grid, direction=direction)
            filtered = filter_scanning_path(path, boundary_polygon, hole_polygons, fov_size_um)
            if return_side is not None and len(filtered) > 1:
                filtered, _ = close_scanning_path(filtered, step_size, return_side=return_side)

            waste_area_um2      = 0.0
            n_low_coverage_fovs = 0
            for x, y in filtered:
                fov_poly        = shapely_box(x - half, y - half, x + half, y + half)
                tissue_overlap  = fov_poly.intersection(effective_tissue).area
                waste_area_um2 += fov_poly.area - tissue_overlap
                if tissue_overlap / fov_area < low_coverage_fraction:
                    n_low_coverage_fovs += 1

            total_length_um, max_step_um = get_path_stats(filtered)
            candidates.append(GridOffsetCandidate(
                offset              = (float(dx), float(dy)),
                n_fovs              = len(filtered),
                waste_area_um2       = waste_area_um2,
                total_length_um      = total_length_um,
                max_step_um          = max_step_um,
                n_low_coverage_fovs  = n_low_coverage_fovs,
            ))

    def _sort_key(c: GridOffsetCandidate) -> Tuple[float, ...]:
        return tuple(getattr(c, field) for field in priority)

    candidates.sort(key=_sort_key)
    best = candidates[0]

    grid, _, _ = create_grid_positions(
        boundary_polygon, step_size, direction=direction, offset=best.offset,
    )
    path     = generate_scanning_path(grid, direction=direction)
    filtered = filter_scanning_path(path, boundary_polygon, hole_polygons, fov_size_um)
    if return_side is not None and len(filtered) > 1:
        filtered, _ = close_scanning_path(filtered, step_size, return_side=return_side)

    return GridOffsetResult(coords=filtered, best=best, candidates=candidates)


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
    )
    return result.coords