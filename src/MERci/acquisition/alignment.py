# MERci/acquisition/alignment.py
"""
Cross-microscope FOV alignment.

Map FOV stage positions imaged on a *source* microscope onto the coordinate
system of a *target* microscope, assuming the physical sample (stage insert)
was moved between scopes with the **same X/Y axis directions** and at most a
small change.  Because it is the same rigid tissue, the two stage coordinate
systems differ only by an **isotropic similarity transform** — a single uniform
scale plus a translation, with **no rotation** but optionally a **per-axis flip**
(a mirror on x and/or y, which can differ between microscopes):

    q_x = (±scale) * p_x + tx
    q_y = (±scale) * p_y + ty

The transform is found by overlapping the two hand-drawn tissue-boundary
polygons (one per microscope).  When flips are allowed, the four axis-flip
combinations are each fitted and the highest-overlap one is kept.  A closed-form moment match (centroid +
area-ratio scale) gives an excellent starting guess, which is then refined by
maximising the polygon intersection-over-union (IoU) with a derivative-free
optimiser — robust even though IoU is non-smooth.

Typical workflow
----------------
1. ``src = load_boundary_polygon(src_txt)`` / ``tgt = load_boundary_polygon(tgt_txt)``
2. ``fit = fit_isotropic_alignment(src, tgt)``
3. ``tgt_positions = fit.transform_points(src_positions)``
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize
from shapely.affinity import scale as _shp_scale, translate as _shp_translate
from shapely.geometry import Polygon

log = logging.getLogger(__name__)


# ── Boundary I/O ────────────────────────────────────────────────────────────

def _read_xy(path: Path) -> List[Tuple[float, float]]:
    """Read a comma-separated ``x,y`` vertex file (one vertex per line).

    Lines that do not parse as two floats (headers, blanks, ``#`` comments)
    are skipped — matching :func:`MERci.acquisition.positions.load_hole_polygons`.
    """
    coords: List[Tuple[float, float]] = []
    with Path(path).open() as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                try:
                    coords.append((float(row[0]), float(row[1])))
                except ValueError:
                    pass
    return coords


def load_boundary_polygon(path: Path) -> Polygon:
    """
    Load a tissue-boundary polygon from a comma-separated ``x,y`` file.

    Parameters
    ----------
    path : boundary file (e.g. ``boundary_positions_mf4.txt``); same format as
           the operator-supplied boundaries used by ``prepare_imaging/02``.

    Returns
    -------
    A valid Shapely :class:`~shapely.geometry.Polygon`.  Self-intersecting
    rings are repaired with ``buffer(0)``.

    Raises
    ------
    ValueError if the file has fewer than three vertices or cannot be repaired
    into a non-empty polygon.
    """
    coords = _read_xy(path)
    if len(coords) < 3:
        raise ValueError(
            f"{Path(path).name} has only {len(coords)} valid vertices; "
            "a boundary polygon needs at least 3."
        )
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area == 0.0:
        raise ValueError(f"{Path(path).name} did not produce a non-empty polygon.")
    return poly


# ── Similarity transform result ─────────────────────────────────────────────

@dataclass
class AlignmentResult:
    """
    An isotropic similarity transform mapping source-microscope coordinates
    onto target-microscope coordinates, optionally with per-axis flips
    (reflections) — but never a rotation:

        q_x = (±scale) * p_x + tx
        q_y = (±scale) * p_y + ty

    where the sign of each axis is negative when that axis is flipped between
    the two microscopes.

    Attributes
    ----------
    scale     : uniform scale **magnitude** (≈1 when both scopes report µm units)
    tx, ty    : translation applied *after* scaling, in target stage units
    iou       : intersection-over-union of the fitted overlap (0–1; 1 = perfect)
    iou_init  : IoU of the closed-form initialisation, before refinement
    n_iter    : optimiser iterations used (0 if refinement was skipped)
    refined   : whether IoU refinement ran and improved on the initial guess
    flip_x    : True if the x-axis is mirrored between source and target
    flip_y    : True if the y-axis is mirrored between source and target
    """

    scale:    float
    tx:       float
    ty:       float
    iou:      float
    iou_init: float
    n_iter:   int
    refined:  bool
    flip_x:   bool = False
    flip_y:   bool = False

    @property
    def translation(self) -> Tuple[float, float]:
        return (self.tx, self.ty)

    @property
    def signed_scale(self) -> Tuple[float, float]:
        """``(sx, sy)`` — per-axis signed scale (negative = that axis flipped)."""
        return (-self.scale if self.flip_x else self.scale,
                -self.scale if self.flip_y else self.scale)

    def transform_points(self, coords: np.ndarray) -> np.ndarray:
        """Apply the transform to an ``(N, 2)`` array of ``(x, y)`` points."""
        coords = np.asarray(coords, dtype=float)
        sx, sy = self.signed_scale
        return coords * np.array([sx, sy]) + np.array([self.tx, self.ty])

    def transform_polygon(self, poly: Polygon) -> Polygon:
        """Apply the transform to a Shapely polygon."""
        sx, sy = self.signed_scale
        return _apply_to_polygon(poly, sx, sy, self.tx, self.ty)


# ── Polygon helpers ─────────────────────────────────────────────────────────

def _apply_to_polygon(poly: Polygon, sx: float, sy: float, tx: float, ty: float) -> Polygon:
    """Scale per-axis about the global origin (0, 0) — allowing negative
    factors for flips — then translate, matching the point transform
    ``q = (sx * p_x + tx, sy * p_y + ty)``."""
    scaled = _shp_scale(poly, xfact=sx, yfact=sy, origin=(0.0, 0.0))
    return _shp_translate(scaled, xoff=tx, yoff=ty)


def polygon_iou(a: Polygon, b: Polygon) -> float:
    """Intersection-over-union of two polygons (0 when they do not overlap)."""
    if not a.is_valid:
        a = a.buffer(0)
    if not b.is_valid:
        b = b.buffer(0)
    inter = a.intersection(b).area
    if inter == 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


# ── Fitting ─────────────────────────────────────────────────────────────────

def _initial_guess(
    src: Polygon,
    tgt: Polygon,
    fx:  float = 1.0,
    fy:  float = 1.0,
) -> Tuple[float, float, float]:
    """
    Closed-form moment match for a given pair of axis-flip signs ``(fx, fy)``.

    A flip (reflection) leaves the area unchanged, so the scale magnitude is
    still ``scale = sqrt(area_tgt / area_src)``.  The translation aligns the
    (flipped) source centroid onto the target centroid:
    ``t = centroid_tgt - scale * (fx, fy) * centroid_src``.
    """
    scale = float(np.sqrt(tgt.area / src.area))
    cs = np.array(src.centroid.coords[0], dtype=float)
    ct = np.array(tgt.centroid.coords[0], dtype=float)
    tx, ty = ct - scale * np.array([fx, fy]) * cs
    return scale, float(tx), float(ty)


def _fit_one_flip(
    src: Polygon,
    tgt: Polygon,
    fx:  float,
    fy:  float,
    refine:  bool,
    maxiter: int,
) -> AlignmentResult:
    """Fit scale + translation for a fixed pair of axis-flip signs."""
    s0, tx0, ty0 = _initial_guess(src, tgt, fx, fy)
    iou_init = polygon_iou(_apply_to_polygon(src, s0 * fx, s0 * fy, tx0, ty0), tgt)
    flip_x, flip_y = (fx < 0), (fy < 0)

    if not refine:
        return AlignmentResult(
            scale=s0, tx=tx0, ty=ty0, iou=iou_init, iou_init=iou_init,
            n_iter=0, refined=False, flip_x=flip_x, flip_y=flip_y,
        )

    def neg_iou(params: np.ndarray) -> float:
        s, tx, ty = params
        s = abs(s)                      # keep scale magnitude positive
        if s == 0.0:
            return 1.0
        moved = _apply_to_polygon(src, s * fx, s * fy, tx, ty)
        return 1.0 - polygon_iou(moved, tgt)

    res = minimize(
        neg_iou,
        x0=np.array([s0, tx0, ty0], dtype=float),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-9},
    )
    s_r, tx_r, ty_r = float(abs(res.x[0])), float(res.x[1]), float(res.x[2])
    iou_r = 1.0 - float(res.fun)

    if iou_r >= iou_init:
        return AlignmentResult(
            scale=s_r, tx=tx_r, ty=ty_r, iou=iou_r, iou_init=iou_init,
            n_iter=int(res.nit), refined=True, flip_x=flip_x, flip_y=flip_y,
        )
    # Refinement made things worse (rare) — keep the closed-form guess.
    return AlignmentResult(
        scale=s0, tx=tx0, ty=ty0, iou=iou_init, iou_init=iou_init,
        n_iter=int(res.nit), refined=False, flip_x=flip_x, flip_y=flip_y,
    )


def fit_isotropic_alignment(
    src:        Polygon,
    tgt:        Polygon,
    refine:     bool = True,
    allow_flip: bool = True,
    maxiter:    int  = 2000,
) -> AlignmentResult:
    """
    Fit an isotropic similarity transform (uniform scale + translation, no
    rotation, optional per-axis flips) mapping *src* onto *tgt* by maximising
    boundary overlap.

    For each candidate flip combination the closed-form centroid/area match
    (:func:`_initial_guess`) seeds a Nelder–Mead refinement that minimises
    ``1 - IoU`` (derivative-free, tolerant of IoU's non-smoothness); the
    refined estimate is kept only if it does not decrease IoU.  When
    *allow_flip* is True the four axis-flip combinations — none, flip-x,
    flip-y, flip-both — are each fitted and the **highest-IoU** result is
    returned, so a microscope that mirrors an axis is handled automatically.

    Note that flip-both is a 180° point reflection, *not* a free rotation; only
    axis-aligned reflections are considered, never an arbitrary angle.

    Parameters
    ----------
    src, tgt   : source and target boundary polygons
    refine     : run the IoU-maximising refinement (default True)
    allow_flip : also try x/y axis flips and keep the best (default True)
    maxiter    : maximum optimiser iterations per flip combination

    Returns
    -------
    AlignmentResult (its ``flip_x`` / ``flip_y`` record which axes were mirrored)
    """
    flips = (
        [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
        if allow_flip else [(1.0, 1.0)]
    )

    best: AlignmentResult | None = None
    for fx, fy in flips:
        cand = _fit_one_flip(src, tgt, fx, fy, refine, maxiter)
        if best is None or cand.iou > best.iou:
            best = cand

    if best.flip_x or best.flip_y:
        log.info(
            "Best alignment flips axes (flip_x=%s, flip_y=%s) with IoU=%.4f.",
            best.flip_x, best.flip_y, best.iou,
        )
    return best
