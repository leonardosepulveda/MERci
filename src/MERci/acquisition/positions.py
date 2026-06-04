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
from pathlib import Path
from typing import List, Tuple

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

def load_hole_polygons(
    hole_dir: Path,
    pattern:  str = "hole*.txt",
) -> List[Polygon]:
    """
    Load exclusion-region polygons from a directory.

    Each file matching *pattern* must be a comma-separated ``x,y`` file
    (one vertex per line).  Files with fewer than three vertices are skipped.

    Returns
    -------
    List of valid Shapely :class:`~shapely.geometry.Polygon` objects.
    """
    hole_dir  = Path(hole_dir)
    polygons: List[Polygon] = []

    for path in sorted(hole_dir.glob(pattern)):
        with path.open() as fh:
            reader = csv.reader(fh)
            coords = []
            for row in reader:
                if len(row) >= 2:
                    try:
                        coords.append((float(row[0]), float(row[1])))
                    except ValueError:
                        pass   # skip header-like lines

        if len(coords) < 3:
            log.warning(
                "%s has fewer than 3 valid points – skipping.", path.name
            )
            continue

        poly = Polygon(coords)
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