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
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
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
           the operator-supplied boundaries used by ``before_imaging/02``.

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


# ── Per-FOV bead drift refinement ───────────────────────────────────────────
#
# The boundary fit above is a *coarse* whole-sample transform. After moving the
# stage insert and re-imaging fiducial beads on the target microscope at the
# coarse-predicted positions, a small residual drift remains per FOV. We measure
# it the way fishtank's `align_experiments` does its coarse stage — image phase
# cross-correlation between the reference (source) and moving (target) bead frame
# — applied independently to each FOV, giving one (dx, dy) correction per FOV.
# (Ref: jweissmanlab/fishtank, src/fishtank/scripts/align_experiments_script.py,
#  which calls skimage.registration.phase_cross_correlation for its coarse shift.)


def bead_frame_indices(
    frame_table: pd.DataFrame,
    bead_color:  Optional[float] = None,
) -> List[int]:
    """
    Return **all** fiducial-bead frame indices within a frame table, in order.

    The frame table has columns ``color``, ``channel``, ``z`` (one row per camera
    frame; see ``acquisition.configs.get_frame_table``).  Fiducial beads are
    imaged in a dedicated colour confined to a single z-plane (e.g. 488 nm at
    z=0, bracketing the multi-z data stack), so when *bead_color* is ``None`` the
    bead colour is auto-detected as the non-blank colour whose frames all sit at
    one z value (preferring the one with the fewest frames when several qualify).

    Parameters
    ----------
    frame_table : DataFrame with ``color`` and ``z`` columns
    bead_color  : force a specific bead colour (nm) instead of auto-detecting

    Returns
    -------
    list of int frame indices (the bead frames, in table order).
    """
    ft = frame_table
    if "color" not in ft.columns or "z" not in ft.columns:
        raise ValueError("frame_table must have 'color' and 'z' columns.")

    if bead_color is None:
        single_plane = []  # (n_frames, color)
        for color, g in ft.dropna(subset=["color"]).groupby("color"):
            if g["z"].nunique() == 1:
                single_plane.append((len(g), color))
        if not single_plane:
            raise ValueError(
                "Could not auto-detect a single-z-plane fiducial colour; "
                "pass bead_color explicitly."
            )
        single_plane.sort()                      # fewest frames first
        bead_color = single_plane[0][1]

    frames = [int(i) for i, c in zip(ft.index, ft["color"]) if c == bead_color]
    if not frames:
        raise ValueError(
            f"No frames with bead colour {bead_color} in frame table "
            f"(colours present: {sorted(set(ft['color'].dropna()))})."
        )
    return frames


def select_bead_frame(
    frame_table: pd.DataFrame,
    bead_color:  Optional[float] = None,
    which:       str             = "first",
) -> int:
    """
    Return a single fiducial-bead frame index from a frame table.

    Thin wrapper over :func:`bead_frame_indices` that picks one of the bead frames
    by *which* — ``"first"`` (default), ``"last"``, or ``"middle"``. See
    :func:`bead_frame_indices` for the (auto-detectable) *bead_color*.
    """
    frames = bead_frame_indices(frame_table, bead_color)
    if which == "first":
        return frames[0]
    if which == "last":
        return frames[-1]
    if which == "middle":
        return frames[len(frames) // 2]
    raise ValueError("which must be 'first', 'last', or 'middle'.")


def extract_bead_frames(
    image_path:   "Path",
    out_path:     "Path",
    frame_indices: Sequence[int],
    frame_width:  Optional[int] = None,
    frame_height: Optional[int] = None,
) -> List[int]:
    """
    Read *image_path*, keep only *frame_indices* (the full-resolution frames), and
    write them to *out_path* as a multi-page ``.tiff``.

    Used by the source-side bead-extraction notebook to shrink each ~30-frame
    acquisition down to just its fiducial frames (full frames — phase
    cross-correlation needs the image, not fitted positions), so only the small
    file is moved to the NAS for the target-side drift step.

    Returns
    -------
    the list of frame indices written (so callers can build a matching compact
    frame table).
    """
    import tifffile
    from MERci.common.io import read_image

    stack = read_image(image_path, frame_width=frame_width, frame_height=frame_height)
    try:
        idx = [int(i) for i in frame_indices]
        sel = np.stack([np.asarray(stack[i]) for i in idx]).astype(stack.dtype)
    finally:
        del stack
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), sel)
    return idx


_ORIENTATIONS = ("none", "fliplr", "flipud", "transpose", "rot90", "rot180", "rot270")


def apply_orientation(img: np.ndarray, orient: str = "none") -> np.ndarray:
    """
    Re-orient a 2-D image to reconcile a fixed pixel-axis convention difference
    between two microscopes (camera handedness / 90° mounting), which ``HAL``'s
    ``flip_horizontal``/``flip_vertical``/``transpose`` flags do **not** capture
    when they are identical on both scopes but the optical paths still differ.

    *orient* is one of: ``"none"`` (default), ``"fliplr"``, ``"flipud"``,
    ``"transpose"``, ``"rot90"``, ``"rot180"``, ``"rot270"``. Apply it to the
    *moving* (target) image before cross-correlation/overlay so it shares the
    reference (source) image's orientation.
    """
    a = np.asarray(img)
    if orient in (None, "none"):
        return a
    if orient == "fliplr":
        return a[:, ::-1]
    if orient == "flipud":
        return a[::-1, :]
    if orient == "transpose":
        return a.T
    if orient == "rot90":
        return np.rot90(a, 1)
    if orient == "rot180":
        return np.rot90(a, 2)
    if orient == "rot270":
        return np.rot90(a, 3)
    raise ValueError(f"orient must be one of {_ORIENTATIONS}, got {orient!r}")


def remove_hot_pixels(
    img:         np.ndarray,
    size:        int   = 3,
    ratio:       float = 5.0,
    sigma_floor: float = 5.0,
) -> np.ndarray:
    """
    Replace isolated hot/dead camera pixels with their local median, before
    registration (:func:`phase_drift`).

    Hot/defective pixels sit at a FIXED detector position, so they are
    identical in every frame and dominate ``phase_cross_correlation`` --
    pinning the estimated drift to exactly ``[0, 0]`` whenever the real
    fiducial (bead) signal is dim. They are isolated single-pixel spikes far
    above the *local* median; a real bead spans several pixels (the PSF) and
    so sits on a bright neighbourhood with a value close to its own local
    median. That difference is what lets this drop hot pixels while leaving
    beads untouched. Ported from a validated fix to the same bug in fishtank
    (``jweissmanlab/fishtank``'s ``detect_spots_script.py``, which this
    module's ``phase_drift`` already mirrors for its registration primitive) --
    confirmed on real data there: the detector flags ~10 isolated hot pixels
    per frame and zero real bead pixels, and drift recovered after removal
    matches whether only the saturated hot pixel or all of them are removed.

    Why not just clip at a fixed maximum (e.g. the dtype max)? A fixed cutoff
    is detector-specific and only catches FULLY SATURATED pixels; a hot pixel
    reading e.g. 12000 on a ~100 background is just as disruptive but slips
    through. The local-median ratio test below is intensity-agnostic and
    catches saturated and sub-saturation hot pixels alike, without flagging a
    real (bright but spatially extended) bead.

    Parameters
    ----------
    img         : 2-D registration image (a single bead/fiducial frame)
    size        : side length (pixels) of the square window for the local
                  median; 3 isolates single-pixel spikes -- a larger window
                  also catches small clusters but risks eroding faint beads
    ratio       : a pixel is flagged when its value exceeds ``ratio`` times its
                  local median. Hot pixels sit on background, so this ratio is
                  large (~10-600 observed); real bead cores sit on a bright
                  neighbourhood, so theirs is ~1-2. 5.0 separates them.
    sigma_floor : also require the excess over the local median to exceed
                  ``sigma_floor`` times the background-noise standard
                  deviation, so faint noise blips on near-zero background
                  (where the ratio alone can be large) are not flagged. Units:
                  noise sigma.

    Returns
    -------
    Copy of *img* with hot pixels replaced by their local median (returned
    unchanged, no copy, if none are flagged).
    """
    local_median = median_filter(img, size=size, mode="nearest").astype(np.float64)
    excess = img.astype(np.float64) - local_median   # height of each pixel above its neighbours

    # Robust background-noise scale from the spread of `excess`. MAD = median
    # absolute deviation; unlike the plain std it is not inflated by the few
    # hot pixels / bead edges we are hunting. 1.4826 = 1 / norm.ppf(0.75)
    # converts a MAD to a Gaussian standard deviation.
    mad = float(np.median(np.abs(excess - np.median(excess))))
    noise_sigma = 1.4826 * mad

    # Hot pixel = isolated spike (value >> local median) that also clears the noise floor.
    hot = (img > ratio * np.maximum(local_median, 1.0)) & (excess > sigma_floor * noise_sigma)
    if hot.any():
        img = img.copy()
        img[hot] = local_median[hot].astype(img.dtype)
    return img


def phase_drift(
    ref2d:           np.ndarray,
    mov2d:           np.ndarray,
    upsample_factor: int = 10,
) -> Tuple[np.ndarray, float]:
    """
    Subpixel registration shift between two 2-D bead images, via
    ``skimage.registration.phase_cross_correlation`` (the same primitive
    fishtank uses for its coarse alignment).

    Parameters
    ----------
    ref2d, mov2d    : 2-D images (reference = source scope, moving = target scope)
    upsample_factor : subpixel precision (1/upsample_factor px); 10 → 0.1 px

    Returns
    -------
    (shift, error) where ``shift`` is ``np.array([dy, dx])`` in pixels — the
    offset that, applied to *mov2d*, registers it onto *ref2d* — and ``error`` is
    the normalised RMS registration error returned by scikit-image.
    """
    from skimage.registration import phase_cross_correlation

    ref = np.asarray(ref2d, dtype=float)
    mov = np.asarray(mov2d, dtype=float)
    shift, error, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample_factor)
    return np.asarray(shift, dtype=float), float(error)


def _drift_one_pair(args: tuple) -> dict:
    """
    Picklable per-pair worker for :func:`compute_fov_drifts`'s
    ``ProcessPoolExecutor`` -- takes a single positional tuple (rather than
    the function's own keyword args) since ``Executor.map`` passes one
    argument per call.
    """
    (fov_id, ref_path, mov_path, ref_frame, mov_frame, mov_orient,
     upsample_factor, pixel_size_um, sign_x, sign_y, frame_width, frame_height) = args
    from MERci.common.io import read_image

    ref_stack = read_image(ref_path, frame_width=frame_width, frame_height=frame_height)
    mov_stack = read_image(mov_path, frame_width=frame_width, frame_height=frame_height)
    try:
        # remove_hot_pixels first: a fixed hot/dead detector pixel is identical
        # in both frames and dominates phase_cross_correlation when the real
        # bead signal is dim, pinning the estimate to [0, 0] (see its
        # docstring) -- so both images get it before registration, not just
        # the moving one, in case the reference frame carries its own.
        shift, error = phase_drift(
            remove_hot_pixels(ref_stack[ref_frame]),
            remove_hot_pixels(apply_orientation(mov_stack[mov_frame], mov_orient)),
            upsample_factor,
        )
    finally:
        del ref_stack, mov_stack
    dy, dx = float(shift[0]), float(shift[1])
    return {
        "fov":        int(fov_id),
        "dy_px":      dy,
        "dx_px":      dx,
        "error":      error,
        "drift_x_um": dx * pixel_size_um * sign_x,
        "drift_y_um": dy * pixel_size_um * sign_y,
    }


def compute_fov_drifts(
    pairs:           Sequence[Tuple[int, "Path", "Path"]],
    ref_frame:       int,
    mov_frame:       int,
    pixel_size_um:   float,
    *,
    upsample_factor: int   = 10,
    sign_x:          float = 1.0,
    sign_y:          float = 1.0,
    mov_orient:      str   = "none",
    frame_width:     Optional[int] = None,
    frame_height:    Optional[int] = None,
    n_workers:       Optional[int] = None,
) -> pd.DataFrame:
    """
    Measure per-FOV bead drift between paired source and target image files.

    For each ``(fov_id, ref_path, mov_path)`` it reads both stacks, takes the
    chosen single z-slice (``ref_frame`` / ``mov_frame``), re-orients the target
    slice by *mov_orient* (see :func:`apply_orientation`), and registers it onto
    the source slice with :func:`phase_drift`.  The pixel shift is converted to
    target stage micrometres via *pixel_size_um* and the axis-sign factors.

    *mov_orient* corrects a fixed pixel-axis convention difference between the two
    microscopes (e.g. one image is flipped/transposed/rotated relative to the
    other) that is NOT reflected in HAL's flip flags. Phase cross-correlation only
    recovers a translation, so if the two scopes' images differ by an orientation
    (or pixel scale), they will not colocalise until that is corrected first — set
    *mov_orient* to whatever the bead overlay shows is needed (default ``"none"``).

    The ``sign_x`` / ``sign_y`` factors (±1) map image (col, row) pixel axes onto
    the target stage (x, y) axes; the correct signs depend on the microscope's
    camera↔stage convention and **must be confirmed against the data** (the
    notebook's quiver plot is for exactly this check). Defaults assume image +x→
    stage +x and image +y→stage +y.

    Each pair reads two full image stacks and runs phase-cross-correlation --
    both I/O- and CPU-bound -- so pairs are processed in parallel across a
    ``ProcessPoolExecutor`` (*n_workers*; ``None`` uses its own default,
    normally ``os.cpu_count()``).

    Returns
    -------
    DataFrame with columns ``fov, dy_px, dx_px, error, drift_x_um, drift_y_um``,
    in the same order as *pairs*.
    """
    columns = ["fov", "dy_px", "dx_px", "error", "drift_x_um", "drift_y_um"]
    pairs = list(pairs)
    if not pairs:
        return pd.DataFrame(columns=columns)

    args_list = [
        (fov_id, ref_path, mov_path, ref_frame, mov_frame, mov_orient,
         upsample_factor, pixel_size_um, sign_x, sign_y, frame_width, frame_height)
        for fov_id, ref_path, mov_path in pairs
    ]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        rows = list(pool.map(_drift_one_pair, args_list))
    return pd.DataFrame(rows, columns=columns)


# ── Bead-detection + point-set registration (modality-robust) ───────────────
#
# Phase cross-correlation needs the two images to look alike. Across modalities
# (epi vs spinning-disk confocal) the raw bead frames differ, so we instead
# DETECT bead centroids in each image and register the two point sets by the
# shift on which the most bead pairs agree (pairwise-displacement voting — a
# Hough/consensus approach robust to spurious or non-matching beads). The
# returned ``score`` (inlier fraction) flags FOVs where the bead sets don't
# actually correspond.


def detect_beads(
    img2d:        np.ndarray,
    min_dist_px:  float = 5.0,
    thresh_sigma: float = 5.0,
    max_beads:    Optional[int] = None,
) -> np.ndarray:
    """
    Detect bead centroids in a 2-D image, returning an ``(N, 2)`` array of
    ``(row, col)`` positions (wraps
    :func:`MERci.analysis.spot_localization.detect_beads_2d`).

    When *max_beads* is set and more are found, the brightest *max_beads* are
    kept (ranked by intensity at the detected pixel).
    """
    from MERci.analysis.spot_localization import detect_beads_2d

    im = np.asarray(img2d)
    pts = detect_beads_2d(im, min_dist_px, thresh_sigma).astype(float)
    if max_beads and len(pts) > max_beads:
        inten = im[pts[:, 0].astype(int), pts[:, 1].astype(int)]
        pts = pts[np.argsort(inten)[::-1][:max_beads]]
    return pts


def register_point_translation(
    src_xy:       np.ndarray,
    tgt_xy:       np.ndarray,
    max_shift_px: float = 1024.0,
    bin_px:       float = 4.0,
) -> dict:
    """
    Find the pure translation that best maps the *tgt* point set onto *src*, by
    voting over all pairwise displacements ``src_i - tgt_j`` and taking the
    histogram peak (then refining with the inliers near it).

    Returns ``{dy, dx, n_inliers, score}`` where ``(dy, dx)`` is the shift in
    pixels (apply to *tgt* to register onto *src*, matching :func:`phase_drift`'s
    convention) and ``score`` is ``n_inliers / min(n_src, n_tgt)`` — a 0–1
    confidence that the two bead sets share a common translation.
    """
    src = np.asarray(src_xy, float)
    tgt = np.asarray(tgt_xy, float)
    if len(src) == 0 or len(tgt) == 0:
        return {"dy": np.nan, "dx": np.nan, "n_inliers": 0, "score": 0.0}

    D = (src[:, None, :] - tgt[None, :, :]).reshape(-1, 2)        # displacements
    m = (np.abs(D[:, 0]) <= max_shift_px) & (np.abs(D[:, 1]) <= max_shift_px)
    D = D[m]
    if len(D) == 0:
        return {"dy": np.nan, "dx": np.nan, "n_inliers": 0, "score": 0.0}

    nb = max(1, int(2 * max_shift_px / bin_px))
    rng = [[-max_shift_px, max_shift_px], [-max_shift_px, max_shift_px]]
    H, ye, xe = np.histogram2d(D[:, 0], D[:, 1], bins=nb, range=rng)
    iy, ix = np.unravel_index(int(H.argmax()), H.shape)
    dy0 = 0.5 * (ye[iy] + ye[iy + 1])
    dx0 = 0.5 * (xe[ix] + xe[ix + 1])

    r = 2 * bin_px
    inl = D[(np.abs(D[:, 0] - dy0) < r) & (np.abs(D[:, 1] - dx0) < r)]
    dy, dx = (inl.mean(0) if len(inl) else (dy0, dx0))
    score = len(inl) / max(1, min(len(src), len(tgt)))
    return {"dy": float(dy), "dx": float(dx), "n_inliers": int(len(inl)), "score": float(score)}


def _drift_one_pair_beads(args: tuple) -> dict:
    """
    Picklable per-pair worker for :func:`compute_fov_drifts_beads`'s
    ``ProcessPoolExecutor`` -- same one-tuple-argument convention as
    :func:`_drift_one_pair`.
    """
    (fov, ref_path, mov_path, mov_orient, min_dist_px, thresh_sigma, max_beads,
     max_shift_px, bin_px, pixel_size_um, sign_x, sign_y, frame_width, frame_height) = args
    from MERci.common.io import read_image

    def _proj(stack):
        a = np.asarray(stack)
        return a.max(0) if a.ndim == 3 else a

    ref = read_image(ref_path, frame_width=frame_width, frame_height=frame_height)
    mov = read_image(mov_path, frame_width=frame_width, frame_height=frame_height)
    try:
        ref2d = _proj(ref)
        mov2d = apply_orientation(_proj(mov), mov_orient)
    finally:
        del ref, mov
    sp = detect_beads(ref2d, min_dist_px, thresh_sigma, max_beads)
    tp = detect_beads(mov2d, min_dist_px, thresh_sigma, max_beads)
    reg = register_point_translation(sp, tp, max_shift_px, bin_px)
    dy, dx = reg["dy"], reg["dx"]
    return {
        "fov":        int(fov),
        "dy_px":      dy,
        "dx_px":      dx,
        "n_src":      int(len(sp)),
        "n_tgt":      int(len(tp)),
        "n_inliers":  reg["n_inliers"],
        "score":      reg["score"],
        "drift_x_um": dx * pixel_size_um * sign_x if np.isfinite(dx) else np.nan,
        "drift_y_um": dy * pixel_size_um * sign_y if np.isfinite(dy) else np.nan,
    }


def compute_fov_drifts_beads(
    pairs:         Sequence[Tuple[int, "Path", "Path"]],
    pixel_size_um: float,
    *,
    min_dist_px:   float = 5.0,
    thresh_sigma:  float = 5.0,
    max_beads:     int   = 800,
    max_shift_px:  float = 1024.0,
    bin_px:        float = 4.0,
    mov_orient:    str   = "none",
    sign_x:        float = 1.0,
    sign_y:        float = 1.0,
    frame_width:   Optional[int] = None,
    frame_height:  Optional[int] = None,
    progress_every: int  = 50,
    n_workers:      Optional[int] = None,
) -> pd.DataFrame:
    """
    Per-FOV bead drift by detection + point-set registration (modality-robust
    alternative to :func:`compute_fov_drifts`).

    For each ``(fov, ref_path, mov_path)`` it max-projects both stacks, detects
    beads in each (the target is first re-oriented by *mov_orient*), registers
    the point sets with :func:`register_point_translation`, and converts the
    pixel shift to target stage µm.

    Each pair reads two full stacks and runs bead detection + registration --
    both I/O- and CPU-bound -- so pairs are processed in parallel across a
    ``ProcessPoolExecutor`` (*n_workers*; ``None`` uses its own default,
    normally ``os.cpu_count()``); *progress_every* still logs every N
    completed pairs.

    Returns a DataFrame (rows in the same order as *pairs*) with columns
    ``fov, dy_px, dx_px, n_src, n_tgt, n_inliers, score, drift_x_um,
    drift_y_um``. A low ``score`` means the bead sets did not find a common
    shift (e.g. the two scopes imaged different beads).
    """
    columns = ["fov", "dy_px", "dx_px", "n_src", "n_tgt",
               "n_inliers", "score", "drift_x_um", "drift_y_um"]
    pairs = list(pairs)
    if not pairs:
        return pd.DataFrame(columns=columns)

    args_list = [
        (fov, ref_path, mov_path, mov_orient, min_dist_px, thresh_sigma, max_beads,
         max_shift_px, bin_px, pixel_size_um, sign_x, sign_y, frame_width, frame_height)
        for fov, ref_path, mov_path in pairs
    ]
    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for k, row in enumerate(pool.map(_drift_one_pair_beads, args_list)):
            rows.append(row)
            if progress_every and (k + 1) % progress_every == 0:
                log.info("bead drift: %d/%d FOVs done", k + 1, len(pairs))
    return pd.DataFrame(rows, columns=columns)
