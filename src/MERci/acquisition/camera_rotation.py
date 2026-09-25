# MERci/acquisition/camera_rotation.py
"""
Camera-vs-stage rotation correction.

Every microscope's camera sensor is mounted at some small, fixed angle
relative to the stage's true X/Y travel axes. A FOV grid built assuming
perfect alignment (see ``acquisition.positions``) is therefore always
slightly wrong -- barcodes and cells near a FOV border can be lost or
double-counted during MERlin segmentation/decoding, because two nominally
4-connected-adjacent FOVs' real image content doesn't actually overlap
where the grid assumes it does.

Since the rotation is a fixed property of the optical path, one single
global affine transform corrects every FOV in the experiment identically --
there is no need to re-image at different stage positions to fix it. This
module estimates that one transform by directly measuring, for a handful of
sampled "anchor" FOVs, the real pixel shift needed to align each anchor with
its 4-connected neighbours in their real overlapping border region (via
phase cross-correlation on a DAPI/cells-round frame -- reusing
:func:`MERci.acquisition.alignment.phase_drift`), then fitting an affine
transform (via the ``affine6p`` package) from every anchor+neighbour pair's
(nominal, measured) position correspondence, pooled together into one fit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .alignment import phase_drift, remove_hot_pixels
from .configs import apply_microscope_orientation
from .positions import find_grid_neighbor

log = logging.getLogger(__name__)

_DIRECTIONS = ("right", "left", "up", "down")

# Every (transpose, flip_horizontal, flip_vertical) combination, the search
# space of detect_image_orientation (an audit/fallback only: each scope's
# MERlin microscope JSON holds its verified orientation). If the audit ever
# disagrees with the JSON, suspect a bug (e.g. in crop_overlap) and inspect
# the raw overlap crops before doubting the JSON.
_ORIENTATION_COMBINATIONS = [
    (transpose, flip_horizontal, flip_vertical)
    for transpose in (False, True)
    for flip_horizontal in (False, True)
    for flip_vertical in (False, True)
]


@dataclass
class NeighborCorrespondence:
    """One anchor-neighbour pair's nominal vs. measured position.

    Attributes
    ----------
    anchor_fov, neighbor_fov : FOV ids
    direction    : one of ``"right"``/``"left"``/``"up"``/``"down"``
                   (anchor -> neighbour)
    nominal_xy   : the neighbour's recorded grid position (µm)
    measured_xy  : the neighbour's true position, i.e. the anchor's own
                   (assumed-correct) recorded position plus the real
                   relative shift measured from image registration (µm)
    error        : phase_cross_correlation's normalised RMS registration
                   error for this pair (lower = more confident)
    """
    anchor_fov:   int
    neighbor_fov: int
    direction:    str
    nominal_xy:   Tuple[float, float]
    measured_xy:  Tuple[float, float]
    error:        float


def crop_overlap(
    anchor_img:       np.ndarray,
    neighbor_img:     np.ndarray,
    direction:        str,
    overlap_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop the expected overlapping strip from a pair of 4-connected-neighbour
    frames, ready for :func:`MERci.acquisition.alignment.phase_drift`.

    *direction* is anchor -> neighbour (e.g. ``"right"`` means the neighbour
    sits on the anchor's +x side, so the anchor's own right edge should
    match the neighbour's left edge). *overlap_fraction* is the expected
    overlap as a fraction of the frame's full width/height (e.g.
    ``1 - ExperimentConfig.non_overlap_fraction``).

    Row convention for "up"/"down": for a correctly-oriented frame (per its
    MERlin microscope-parameters JSON), row index 0 is the physical -y
    (down) edge, not +y (up) -- so "up" crops the anchor's LAST n rows
    against the neighbour's FIRST n rows. Getting this backwards can cancel
    out with a missing ``flip_vertical`` and look correct by coincidence, so
    don't assume it's right just because the crops look plausible.

    Returns
    -------
    (anchor_crop, neighbor_crop) -- two same-shape 2-D arrays that should
    align (up to the real camera-rotation-induced residual) if the nominal
    grid positions were exactly right.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")

    h, w = anchor_img.shape
    if direction in ("right", "left"):
        n = max(1, int(round(w * overlap_fraction)))
        if direction == "right":
            return anchor_img[:, w - n:], neighbor_img[:, :n]
        return anchor_img[:, :n], neighbor_img[:, w - n:]
    else:  # "up" / "down"
        n = max(1, int(round(h * overlap_fraction)))
        if direction == "up":
            return anchor_img[h - n:, :], neighbor_img[:n, :]
        return anchor_img[:n, :], neighbor_img[h - n:, :]


def register_neighbor_pair(
    anchor_img:       np.ndarray,
    neighbor_img:     np.ndarray,
    anchor_xy:        Tuple[float, float],
    neighbor_xy:      Tuple[float, float],
    direction:        str,
    overlap_fraction: float,
    pixel_size_um:    float,
    upsample_factor:    int = 10,
    orient_transpose:       bool = False,
    orient_flip_horizontal: bool = False,
    orient_flip_vertical:   bool = False,
) -> Tuple[Tuple[float, float], float]:
    """
    Measure the neighbour's TRUE position relative to the anchor, from the
    real pixel shift needed to align their overlapping border crop.

    Parameters
    ----------
    orient_transpose, orient_flip_horizontal, orient_flip_vertical : applied
                  to BOTH images (via :func:`apply_microscope_orientation`)
                  before cropping/registering -- corrects for this camera's
                  raw-frame row/column axes not lining up with physical
                  stage x/y the way ``crop_overlap`` assumes (row=y, col=x,
                  no flip). Camera/mounting-specific -- read the correct
                  values from this microscope's own MERlin microscope-
                  parameters JSON rather than assuming all-``False``; see
                  :func:`detect_image_orientation` for a fallback/audit
                  search when that file is unavailable or suspect.

    Returns
    -------
    (measured_neighbor_xy, error) -- the neighbour's measured true (x, y)
    stage position (µm), and the registration's normalised RMS error.
    """
    if orient_transpose or orient_flip_horizontal or orient_flip_vertical:
        orientation = dict(transpose=orient_transpose, flip_horizontal=orient_flip_horizontal,
                           flip_vertical=orient_flip_vertical)
        anchor_img   = apply_microscope_orientation(anchor_img, **orientation)
        neighbor_img = apply_microscope_orientation(neighbor_img, **orientation)
    a_crop, n_crop = crop_overlap(anchor_img, neighbor_img, direction, overlap_fraction)
    shift, error = phase_drift(
        remove_hot_pixels(a_crop), remove_hot_pixels(n_crop), upsample_factor
    )
    dy_px, dx_px = float(shift[0]), float(shift[1])

    nom_dx = neighbor_xy[0] - anchor_xy[0]
    nom_dy = neighbor_xy[1] - anchor_xy[1]
    meas_dx = nom_dx + dx_px * pixel_size_um
    meas_dy = nom_dy + dy_px * pixel_size_um
    return (anchor_xy[0] + meas_dx, anchor_xy[1] + meas_dy), error


def sample_neighbor_correspondences(
    fov_ids:            List[int],
    positions:          Dict[int, Tuple[float, float]],
    load_frame:         Callable[[int], np.ndarray],
    step_size_um:       float,
    pixel_size_um:      float,
    overlap_fraction:   float,
    n_anchors:          int = 10,
    directions:         Tuple[str, ...] = _DIRECTIONS,
    tolerance_fraction: float = 0.25,
    upsample_factor:    int = 10,
    orient_transpose:       bool = False,
    orient_flip_horizontal: bool = False,
    orient_flip_vertical:   bool = False,
    seed:               Optional[int] = 0,
    progress_callback:  Optional[Callable[[int, int], None]] = None,
) -> List[NeighborCorrespondence]:
    """
    Sample *n_anchors* FOVs spread across *fov_ids* and register each one
    against its present 4-connected neighbours.

    Parameters
    ----------
    fov_ids     : candidate anchor FOV ids (typically one round's real
                  imaged FOVs -- excludes transit-only positions)
    positions   : {fov_id: (x, y)} nominal grid positions (µm), covering
                  every id in *fov_ids* and its neighbours
    load_frame  : ``load_frame(fov_id) -> np.ndarray``, returning the 2-D
                  registration image (e.g. a DAPI frame at a fixed z) for
                  one FOV -- kept generic so this function doesn't need to
                  know about image file formats/paths; results are cached
                  per FOV id since a neighbour can also be sampled as
                  another anchor's neighbour
    step_size_um, pixel_size_um, overlap_fraction : grid/camera geometry
                  (e.g. from :class:`MERci.common.config.ExperimentConfig`)
    n_anchors   : how many anchor FOVs to sample (default 10)
    directions  : which 4-connected directions to test per anchor (default
                  all four)
    orient_transpose, orient_flip_horizontal, orient_flip_vertical : passed
                  through to :func:`register_neighbor_pair` -- read these
                  from this microscope's own MERlin microscope-parameters
                  JSON (``data/configs/merlin/microscope/*.json``) before
                  trusting any correspondence this function returns; see
                  :func:`detect_image_orientation` for a fallback/audit.
    seed        : RNG seed for anchor sampling (deterministic by default;
                  ``None`` for a fresh random sample each call)
    progress_callback : optional ``callback(done, total)`` for a live
                  progress display (see ``NOTEBOOK_GUIDELINES.md`` #4)

    Returns
    -------
    List of :class:`NeighborCorrespondence`, one per successfully-registered
    anchor+neighbour pair (a direction is skipped when that neighbour wasn't
    imaged, e.g. the anchor sits on the grid's exterior on that side).
    """
    rng = np.random.default_rng(seed)
    candidates = list(fov_ids)
    rng.shuffle(candidates)
    anchors = candidates[:n_anchors]

    frame_cache: Dict[int, np.ndarray] = {}

    def _get_frame(fov_id: int) -> np.ndarray:
        if fov_id not in frame_cache:
            frame_cache[fov_id] = load_frame(fov_id)
        return frame_cache[fov_id]

    correspondences: List[NeighborCorrespondence] = []
    total = len(anchors) * len(directions)
    done = 0
    for anchor_fov in anchors:
        anchor_img = _get_frame(anchor_fov)
        for direction in directions:
            neighbor_fov = find_grid_neighbor(
                anchor_fov, positions, direction, step_size_um, tolerance_fraction
            )
            done += 1
            if progress_callback:
                progress_callback(done, total)
            if neighbor_fov is None:
                continue
            neighbor_img = _get_frame(neighbor_fov)
            measured_xy, error = register_neighbor_pair(
                anchor_img, neighbor_img,
                positions[anchor_fov], positions[neighbor_fov],
                direction, overlap_fraction, pixel_size_um, upsample_factor,
                orient_transpose, orient_flip_horizontal, orient_flip_vertical,
            )
            correspondences.append(NeighborCorrespondence(
                anchor_fov=anchor_fov, neighbor_fov=neighbor_fov, direction=direction,
                nominal_xy=positions[neighbor_fov], measured_xy=measured_xy, error=error,
            ))
    return correspondences


def detect_image_orientation(
    fov_ids:            List[int],
    positions:          Dict[int, Tuple[float, float]],
    load_frame:         Callable[[int], np.ndarray],
    step_size_um:       float,
    pixel_size_um:      float,
    overlap_fraction:   float,
    n_trial_anchors:    int = 3,
    tolerance_fraction: float = 0.25,
    upsample_factor:    int = 10,
    seed:               Optional[int] = 0,
) -> Tuple[Tuple[bool, bool, bool], pd.DataFrame]:
    """
    Audit/fallback: try all 8 (transpose, flip_horizontal, flip_vertical)
    combinations on a few trial anchors and return the one with the SMALLEST
    median registered-shift magnitude.

    The microscope's MERlin JSON (``data/configs/merlin/microscope/*.json``) is
    the primary source; use this to cross-check it or when it is missing. It
    searches all 8 combinations because a best-of-a-subset can still leave a
    visible mis-stitch. The criterion works because real camera rotation is
    well under a degree, so true neighbours need only a few pixels of shift,
    and a wrong orientation won't give uniformly small shifts.

    Parameters
    ----------
    fov_ids, positions, load_frame, step_size_um, pixel_size_um,
    overlap_fraction, tolerance_fraction, upsample_factor : as in
                  :func:`sample_neighbor_correspondences`
    n_trial_anchors : anchors sampled per combination (default 3)
    seed        : shared by all combinations, so every one is scored on the
                  same anchors and neighbours

    Returns
    -------
    ((transpose, flip_horizontal, flip_vertical), results_df) -- the winner, and
    one row per combination (transpose, flip_horizontal, flip_vertical,
    n_correspondences, median_shift_um).
    """
    import pandas as pd

    rows = []
    for transpose, flip_h, flip_v in _ORIENTATION_COMBINATIONS:
        trial = sample_neighbor_correspondences(
            fov_ids=fov_ids, positions=positions, load_frame=load_frame,
            step_size_um=step_size_um, pixel_size_um=pixel_size_um,
            overlap_fraction=overlap_fraction, n_anchors=n_trial_anchors,
            tolerance_fraction=tolerance_fraction, upsample_factor=upsample_factor,
            orient_transpose=transpose, orient_flip_horizontal=flip_h, orient_flip_vertical=flip_v,
            seed=seed,
        )
        row = {"transpose": transpose, "flip_horizontal": flip_h, "flip_vertical": flip_v}
        if not trial:
            rows.append({**row, "n_correspondences": 0, "median_shift_um": np.inf})
            continue
        shifts_um = [
            float(np.hypot(c.measured_xy[0] - c.nominal_xy[0], c.measured_xy[1] - c.nominal_xy[1]))
            for c in trial
        ]
        rows.append({**row, "n_correspondences": len(trial), "median_shift_um": float(np.median(shifts_um))})

    results_df = pd.DataFrame(rows).sort_values("median_shift_um").reset_index(drop=True)
    best = results_df.iloc[0]
    best_combination = (bool(best["transpose"]), bool(best["flip_horizontal"]), bool(best["flip_vertical"]))
    return best_combination, results_df


@dataclass
class CameraRotationCorrection:
    """
    A fitted camera-vs-stage rotation correction (2-D affine transform).

    Attributes
    ----------
    matrix : ``(3, 3)`` affine matrix (``affine6p`` convention: last row
             ``[0, 0, 1]``); use :meth:`transform_points` to apply it
    n_correspondences : how many neighbour-pair measurements fed the fit
    zero_translation  : whether the fitted translation was dropped before
                         being stored (see :func:`fit_camera_rotation`)
    """
    matrix:            np.ndarray
    n_correspondences: int
    zero_translation:  bool

    def transform_points(self, coords: np.ndarray) -> np.ndarray:
        """Apply the affine transform to an ``(N, 2)`` array of (x, y) points."""
        coords = np.asarray(coords, dtype=float)
        ones = np.ones((coords.shape[0], 1))
        homog = np.hstack([coords, ones])
        return (self.matrix @ homog.T).T[:, :2]

    def save(self, path: Path) -> None:
        """Save the affine matrix as a ``.npy`` file."""
        np.save(str(path), self.matrix, allow_pickle=False)

    @classmethod
    def load(
        cls, path: Path, n_correspondences: int = -1, zero_translation: bool = True,
    ) -> "CameraRotationCorrection":
        """Load a previously-saved affine matrix.

        *n_correspondences*/*zero_translation* are metadata this method
        cannot recover from the bare ``.npy`` matrix -- pass them through if
        known (e.g. from a sibling metadata file), otherwise they default to
        placeholders (``-1`` / ``True``) that don't affect
        :meth:`transform_points`.
        """
        matrix = np.load(str(path), allow_pickle=False)
        return cls(matrix=matrix, n_correspondences=n_correspondences,
                    zero_translation=zero_translation)


def fit_camera_rotation(
    correspondences:  List[NeighborCorrespondence],
    zero_translation: bool = True,
) -> CameraRotationCorrection:
    """
    Fit ONE global affine transform from every sampled neighbour-pair
    correspondence, pooled together -- not one fit per anchor then averaged,
    since every correspondence reflects the same single physical rotation
    (this module's whole premise), so pooling all of them into one
    least-squares estimate is more robust than any one anchor's own ~4
    points could give alone.

    Uses ``affine6p`` (``pip install affine6p``) to fit a full 2-D affine
    (rotation + scale + shear + translation) from >= 3 point
    correspondences by least squares.

    Parameters
    ----------
    correspondences  : from :func:`sample_neighbor_correspondences`; each
                       contributes one (nominal, measured) point pair
    zero_translation : if True (default), the fitted translation
                       (``matrix[0, 2]``/``matrix[1, 2]``) is zeroed before
                       returning. Every correspondence measures a LOCAL,
                       anchor-relative displacement -- this method has no
                       way to observe a real absolute/global position
                       offset, only the rotation+scale relating any two
                       neighbouring FOVs -- so a non-zero fitted translation
                       reflects finite-sample noise across the pooled
                       anchors rather than a real effect; keeping it would
                       risk shifting the whole corrected grid without cause.

    Returns
    -------
    CameraRotationCorrection mapping a *nominal* (recorded) position to its
    corrected (true) position -- apply it to the full experiment's
    positions.txt array via :meth:`CameraRotationCorrection.transform_points`.
    """
    import affine6p

    if len(correspondences) < 3:
        raise ValueError(
            f"Need at least 3 correspondences to fit an affine transform, "
            f"got {len(correspondences)}."
        )

    nominal  = [list(c.nominal_xy) for c in correspondences]
    measured = [list(c.measured_xy) for c in correspondences]
    trans = affine6p.estimate(nominal, measured)
    matrix = np.array(trans.get_matrix(), dtype=float)

    if zero_translation:
        matrix[0, 2] = 0.0
        matrix[1, 2] = 0.0

    return CameraRotationCorrection(
        matrix=matrix, n_correspondences=len(correspondences),
        zero_translation=zero_translation,
    )


def filter_correspondence_outliers(
    correspondences: List[NeighborCorrespondence],
    mad_threshold:   float = 5.0,
) -> Tuple[List[NeighborCorrespondence], List[NeighborCorrespondence]]:
    """
    Split correspondences into (kept, rejected) by a robust outlier test on
    each one's ``|measured - nominal|`` shift magnitude.

    Even with the right :func:`detect_image_orientation` in hand, a handful
    of individual registrations can still fail outright -- weak/sparse DAPI
    signal in that particular FOV, an occasional bad phase-correlation peak.
    The bulk of correspondences cluster tightly (a few um), with a handful
    of clear outliers an order of magnitude or more larger -- a real gap in
    the distribution, not a continuum, so a robust threshold cleanly
    separates them without needing manual review.

    Parameters
    ----------
    correspondences : from :func:`sample_neighbor_correspondences`
    mad_threshold   : reject a correspondence if its shift magnitude exceeds
                      ``median + mad_threshold * robust_sigma``, where
                      ``robust_sigma = 1.4826 * median_absolute_deviation``
                      (1.4826 = 1/norm.ppf(0.75), the standard conversion
                      from a MAD to a Gaussian-equivalent standard
                      deviation). Default 5.0 is generous -- it only drops
                      genuinely discrepant measurements, not real spread in
                      an otherwise well-behaved set.

    Returns
    -------
    (kept, rejected) -- both lists of :class:`NeighborCorrespondence`, in
    the same order as *correspondences*.
    """
    if len(correspondences) < 3:
        return list(correspondences), []

    shifts_um = np.array([
        np.hypot(c.measured_xy[0] - c.nominal_xy[0], c.measured_xy[1] - c.nominal_xy[1])
        for c in correspondences
    ])
    median = float(np.median(shifts_um))
    mad = float(np.median(np.abs(shifts_um - median)))
    robust_sigma = 1.4826 * mad
    threshold = median + mad_threshold * robust_sigma if robust_sigma > 0 else median

    kept     = [c for c, s in zip(correspondences, shifts_um) if s <= threshold]
    rejected = [c for c, s in zip(correspondences, shifts_um) if s > threshold]
    return kept, rejected


@dataclass
class GlobalPositionCorrection:
    """
    Per-FOV positions from jointly solving every measured FOV's own position
    against all its pairwise neighbour constraints at once (see
    :func:`fit_global_positions`), instead of fitting one whole-grid affine.

    Attributes
    ----------
    positions         : ``{fov_id: (x, y)}`` (µm) -- only FOVs that appeared
                        in at least one kept correspondence; merge over a
                        full nominal (or affine-corrected) positions dict as
                        a fallback for every other FOV.
    anchor_fovs        : ``{component_id: fov_id}`` -- the one FOV in each
                        connected correspondence-graph component held fixed
                        at its own nominal position, to remove that
                        component's translational null space (a uniform
                        shift of every position in an isolated component
                        satisfies its own constraints equally well, so one
                        reference point per component is required).
    n_fovs_solved      : ``len(positions)``
    n_correspondences  : how many correspondences fed the solve
    n_components       : how many disconnected correspondence-graph
                        components were solved independently (this
                        module's sparse anchor-sampling strategy typically
                        produces one component per sampled anchor, rarely
                        overlapping -- see :func:`fit_global_positions`)
    residual_rms_um    : RMS of ``(p[B] - p[A]) - measured_relative_offset``
                        across every correspondence, evaluated at the
                        solved positions -- 0.0 whenever every component is
                        a simple star (exactly-determined, no redundant
                        measurement to disagree with itself); only becomes
                        informative once some FOV is constrained by more
                        than one independent correspondence.
    """
    positions:         Dict[int, Tuple[float, float]]
    anchor_fovs:       Dict[int, int]
    n_fovs_solved:     int
    n_correspondences: int
    n_components:      int
    residual_rms_um:   float


def _connected_components(correspondences: List[NeighborCorrespondence]) -> List[List[int]]:
    """Plain BFS connected components of the anchor<->neighbour graph --
    these graphs are tiny (tens to low hundreds of nodes), no graph library
    needed."""
    adjacency: Dict[int, set] = {}
    for c in correspondences:
        adjacency.setdefault(c.anchor_fov, set()).add(c.neighbor_fov)
        adjacency.setdefault(c.neighbor_fov, set()).add(c.anchor_fov)

    visited: set = set()
    components = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack, component = [start], []
        visited.add(start)
        while stack:
            fov = stack.pop()
            component.append(fov)
            for nb in adjacency[fov]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(sorted(component))
    return components


def fit_global_positions(
    correspondences:  List[NeighborCorrespondence],
    nominal_positions: Dict[int, Tuple[float, float]],
    lsqr_atol:         float = 1.0e-12,
    lsqr_btol:         float = 1.0e-12,
) -> GlobalPositionCorrection:
    """
    Solve for each measured FOV's own position from all kept neighbour
    correspondences, instead of one global affine (:func:`fit_camera_rotation`)
    for the whole grid. A global affine only corrects rotation/scale/shear
    shared by the whole grid; it averages away real per-FOV stage jitter.

    Each correspondence (anchor A, neighbour B) gives
    ``r_AB = measured_xy(B) - nominal_positions[A]``. Solve::

        minimize over positions p:  sum_AB || (p[B] - p[A]) - r_AB ||^2

    as two sparse linear least-squares problems (x and y,
    ``scipy.sparse.linalg.lsqr``). Each connected component of the
    correspondence graph can shift freely, so one FOV per component (the one
    that is ``anchor_fov`` in the most correspondences) is pinned to its
    nominal position.

    Limitation: with sparse sampling (:func:`sample_neighbor_correspondences`
    gives mostly separate anchor + ~4-neighbour stars) each leaf has one
    constraint, so its solved position equals its ``measured_xy`` and
    ``residual_rms_um`` is 0. Real error averaging needs overlapping samples.

    Keep *lsqr_atol*/*lsqr_btol* at 1e-12: on a dense correspondence set,
    scipy's default tolerances stop far short of the ~3 µm signal being
    resolved. Check ``istop`` is 1 or 2 (not 7 = iteration limit, or 3/4 =
    ill-conditioned) before loosening them. Raising ``PIN_WEIGHT`` does not fix
    convergence.

    Parameters
    ----------
    correspondences   : from :func:`sample_neighbor_correspondences`, after
                        :func:`filter_correspondence_outliers`
    nominal_positions : ``{fov_id: (x, y)}`` for the whole grid (anchors'
                        nominal positions are not stored on correspondences)
    lsqr_atol, lsqr_btol : ``lsqr`` tolerances for both solves (see above)

    Returns
    -------
    GlobalPositionCorrection. Merge ``.positions`` over a full nominal (or
    affine-corrected) positions dict for FOVs that were not measured.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import lsqr

    if not correspondences:
        return GlobalPositionCorrection(
            positions={}, anchor_fovs={}, n_fovs_solved=0,
            n_correspondences=0, n_components=0, residual_rms_um=0.0,
        )

    components = _connected_components(correspondences)
    all_fovs = sorted({fov for comp in components for fov in comp})
    fov_to_idx = {fov: i for i, fov in enumerate(all_fovs)}
    n = len(all_fovs)

    # Pin each component's most-sampled real anchor to its own nominal position.
    anchor_counts: Dict[int, int] = {}
    for c in correspondences:
        anchor_counts[c.anchor_fov] = anchor_counts.get(c.anchor_fov, 0) + 1
    anchor_fovs = {
        comp_id: max(comp, key=lambda fov: anchor_counts.get(fov, 0))
        for comp_id, comp in enumerate(components)
    }

    n_corr = len(correspondences)
    n_pins = len(anchor_fovs)
    # Heavily weighted relative to unit-weighted correspondence rows -- pins
    # the component's reference FOV to within numerical noise of its real
    # nominal position without needing a true equality-constrained solver.
    PIN_WEIGHT = 1.0e4

    def _solve_axis(axis: int) -> np.ndarray:
        A = lil_matrix((n_corr + n_pins, n), dtype=float)
        b = np.zeros(n_corr + n_pins, dtype=float)

        for row, c in enumerate(correspondences):
            i_a, i_b = fov_to_idx[c.anchor_fov], fov_to_idx[c.neighbor_fov]
            A[row, i_b] += 1.0
            A[row, i_a] += -1.0
            b[row] = c.measured_xy[axis] - nominal_positions[c.anchor_fov][axis]

        for offset, (comp_id, pin_fov) in enumerate(anchor_fovs.items()):
            row = n_corr + offset
            A[row, fov_to_idx[pin_fov]] = PIN_WEIGHT
            b[row] = PIN_WEIGHT * nominal_positions[pin_fov][axis]

        solution = lsqr(A.tocsr(), b, atol=lsqr_atol, btol=lsqr_btol)[0]
        return solution

    x_solution = _solve_axis(0)
    y_solution = _solve_axis(1)
    positions = {
        fov: (float(x_solution[i]), float(y_solution[i])) for fov, i in fov_to_idx.items()
    }

    residuals_um = []
    for c in correspondences:
        p_a = positions[c.anchor_fov]
        p_b = positions[c.neighbor_fov]
        r_ab = (
            c.measured_xy[0] - nominal_positions[c.anchor_fov][0],
            c.measured_xy[1] - nominal_positions[c.anchor_fov][1],
        )
        residuals_um.append(np.hypot(p_b[0] - p_a[0] - r_ab[0], p_b[1] - p_a[1] - r_ab[1]))
    residual_rms_um = float(np.sqrt(np.mean(np.square(residuals_um)))) if residuals_um else 0.0

    return GlobalPositionCorrection(
        positions=positions, anchor_fovs=anchor_fovs, n_fovs_solved=len(positions),
        n_correspondences=n_corr, n_components=len(components), residual_rms_um=residual_rms_um,
    )


@dataclass
class LocalPositionCorrection:
    """
    Per-FOV positions from a GREEDY, most-reliable-direction-first
    spanning-tree walk outward from a fixed root FOV (see
    :func:`greedy_local_positions`) -- an alternative to
    :func:`fit_global_positions`'s joint least-squares solve, for a densely
    (not sparsely) sampled correspondence set where every FOV has several
    real 4-connected measurements and a genuine choice of which one to trust.

    Attributes
    ----------
    positions        : ``{fov_id: (x, y)}`` (µm) -- every id appearing in
                       *nominal_positions* or the correspondence graph;
                       unreached FOVs fall back to their own nominal position
    root_fov          : the FOV held fixed at its own nominal position
    n_fovs_placed     : FOVs actually reached via the spanning-tree walk
                       (excludes root_fov and any fallback-to-nominal FOV)
    n_fovs_unreached  : FOVs with no path to *root_fov* through the kept
                       correspondence graph -- fell back to nominal position
    n_correspondences : how many correspondences fed the walk
    """
    positions:         Dict[int, Tuple[float, float]]
    root_fov:          int
    n_fovs_placed:     int
    n_fovs_unreached:  int
    n_correspondences: int


def greedy_local_positions(
    correspondences:       List[NeighborCorrespondence],
    nominal_positions:     Dict[int, Tuple[float, float]],
    direction_reliability: Optional[Dict[str, float]] = None,
    root_fov:              int = 0,
) -> LocalPositionCorrection:
    """
    Place every FOV by walking outward from *root_fov*, each step taking the
    correspondence from the most reliable direction that reaches an unplaced
    FOV from a placed one (Prim's algorithm; the weight is the whole
    direction's reliability, not one measurement's noise).

    Unlike :func:`fit_global_positions`, which averages all correspondences,
    each FOV here is placed by exactly one correspondence and any other edge to
    it is dropped. That uses "this direction is noisier" information the
    least-squares fit can't see. Best on a dense correspondence set (most FOVs
    measured against most neighbours). With *direction_reliability* ``None``
    it is plain BFS (first reached wins).

    Parameters
    ----------
    correspondences       : from :func:`sample_neighbor_correspondences`,
                            ideally after :func:`filter_correspondence_outliers`
    nominal_positions     : ``{fov_id: (x, y)}`` for the whole grid: gives each
                            anchor's nominal position, and the fallback for FOVs
                            the walk never reaches
    direction_reliability : ``{direction: score}``, LOWER = more reliable (e.g.
                            that direction's std of ``measured - nominal``);
                            ``None`` treats all directions equally
    root_fov              : FOV held at its nominal position (default 0)

    Returns
    -------
    LocalPositionCorrection
    """
    import heapq

    # Bidirectional adjacency: correspondence anchor->neighbor with measured
    # relative offset r = measured_xy(neighbor) - nominal_xy(anchor) implies
    # neighbor's position = anchor's position + r (forward), or equally
    # anchor's position = neighbor's position - r (reverse) -- same
    # correspondence, usable to place either endpoint from the other.
    adjacency: Dict[int, List[Tuple[int, Tuple[float, float], str]]] = {}
    for c in correspondences:
        r = (
            c.measured_xy[0] - nominal_positions[c.anchor_fov][0],
            c.measured_xy[1] - nominal_positions[c.anchor_fov][1],
        )
        adjacency.setdefault(c.anchor_fov, []).append((c.neighbor_fov, r, c.direction))
        adjacency.setdefault(c.neighbor_fov, []).append((c.anchor_fov, (-r[0], -r[1]), c.direction))

    def _priority(direction: str) -> float:
        if direction_reliability is None:
            return 0.0
        return direction_reliability.get(direction, float("inf"))

    positions: Dict[int, Tuple[float, float]] = {root_fov: nominal_positions[root_fov]}
    heap: List[Tuple[float, int, int, int, Tuple[float, float]]] = []
    counter = 0   # stable tie-break within equal priority, insertion order

    def _push_frontier(fov: int) -> None:
        nonlocal counter
        for nb, r, direction in adjacency.get(fov, []):
            if nb in positions:
                continue
            counter += 1
            heapq.heappush(heap, (_priority(direction), counter, fov, nb, r))

    _push_frontier(root_fov)
    while heap:
        _, _, frm, to, r = heapq.heappop(heap)
        if to in positions:
            continue   # already placed via a higher-priority path since this was queued
        frm_pos = positions[frm]
        positions[to] = (frm_pos[0] + r[0], frm_pos[1] + r[1])
        _push_frontier(to)

    n_unreached = 0
    for fov in set(nominal_positions) | set(adjacency):
        if fov not in positions:
            positions[fov] = nominal_positions[fov]
            n_unreached += 1

    return LocalPositionCorrection(
        positions=positions, root_fov=root_fov,
        n_fovs_placed=len(positions) - n_unreached - 1,   # exclude root_fov itself
        n_fovs_unreached=n_unreached,
        n_correspondences=len(correspondences),
    )


def overlap_correlation(
    anchor_img:       np.ndarray,
    neighbor_img:     np.ndarray,
    direction:        str,
    overlap_fraction: float,
    extra_shift_um:   Tuple[float, float] = (0.0, 0.0),
    pixel_size_um:    float = 1.0,
) -> float:
    """
    Pearson correlation between an anchor/neighbour pair's overlap-band
    crops, optionally shifting the neighbour's crop by *extra_shift_um*
    (an ``(dx, dy)`` offset BEYOND the nominal-grid alignment
    :func:`crop_overlap` already assumes) before correlating.

    Unlike :func:`register_neighbor_pair`, this does not itself measure a
    shift via phase correlation -- it evaluates agreement AT a shift already
    decided elsewhere (zero, for the raw nominal grid; a fitted affine
    transform's own implied residual; a position-solve's own implied
    residual; ...), so several candidate position sets can be compared
    against each other on equal footing using the same real image content.

    Returns ``0.0`` (not ``NaN``) for a degenerate (constant) crop -- a
    zero-variance crop has no real correlation to report, and ``NaN`` would
    silently corrupt any downstream mean.
    """
    from scipy.ndimage import shift as ndi_shift

    a_crop, n_crop = crop_overlap(anchor_img, neighbor_img, direction, overlap_fraction)
    a_crop = remove_hot_pixels(a_crop).astype(np.float64)
    n_crop = remove_hot_pixels(n_crop).astype(np.float64)

    dx_px = extra_shift_um[0] / pixel_size_um
    dy_px = extra_shift_um[1] / pixel_size_um
    if dx_px != 0.0 or dy_px != 0.0:
        n_crop = ndi_shift(n_crop, shift=(dy_px, dx_px), order=1, mode="nearest")

    a_flat, n_flat = a_crop.ravel(), n_crop.ravel()
    if a_flat.std() == 0.0 or n_flat.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a_flat, n_flat)[0, 1])
