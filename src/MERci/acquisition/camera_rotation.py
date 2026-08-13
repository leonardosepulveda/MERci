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

Historical context: this replaces an earlier BigStitcher-based manual
workflow (see ``notebooks/misc/correct_camera_rotation.ipynb``'s intro and
this project's ``prompt_history``) -- BigStitcher gave good results on a
small (~5x5) contiguous tile block via full pairwise-shift + global
optimization, but did not scale to a full experiment (>1000 tiles) and
required manual bad-link curation. Sampling several small, independent
4-connected-neighbour groups scattered across the whole grid and pooling
their correspondences into one fit achieves the same "one global rotation"
estimate without either limitation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .alignment import phase_drift, remove_hot_pixels
from .positions import find_grid_neighbor

log = logging.getLogger(__name__)

_DIRECTIONS = ("right", "left", "up", "down")

# Every (transpose, flip_horizontal, flip_vertical) combination --
# detect_image_orientation()'s audit/fallback search space. Whether a
# camera's raw frame rows/columns line up with physical stage x/y (and which
# way) is NOT standardised across microscopes/mountings. MERlin's own
# microscope-parameters JSON (data/configs/merlin/microscope/*.json) already
# records the correct, verified combination per microscope as
# transpose/flip_horizontal/flip_vertical booleans -- read and apply THAT
# directly (see notebooks/misc/correct_camera_rotation.ipynb section 4)
# rather than guessing. This 8-combination search exists only as a fallback/
# audit.
#
# History worth knowing before trusting this search's output: on a real MF3
# dataset, this audit twice ranked "transpose alone" above MERFISH3.json's
# real combination (transpose=True, flip_horizontal=False,
# flip_vertical=True). Both times the cause was a SEPARATE bug in
# crop_overlap below, not a wrong microscope-parameters file: crop_overlap's
# "up" direction had its anchor/neighbour row selection backwards, which
# happened to cancel out with the missing flip_vertical and looked like a
# clean signal. Confirmed directly (see prompt_history) by cropping the same
# real overlap region under both candidates and inspecting it visually --
# the JSON's combination with the ORIGINAL (buggy) crop_overlap produced
# visibly mismatched crops and a large, inconsistent measured shift, while
# the same JSON combination with crop_overlap's row selection swapped gave
# small, visually-matching crops identical to what "transpose alone" had
# been giving. crop_overlap now has the corrected row convention, and this
# audit agrees with MERFISH3.json (ranks it #1/8) as a result -- but if this
# audit ever again ranks a DIFFERENT combination above the microscope-
# parameters JSON's declared one, do not assume the JSON is wrong: inspect
# the raw overlap crops directly first, the same way this bug was actually
# found, since a compensating bug elsewhere is at least as likely as a wrong
# JSON file.
_ORIENTATION_COMBINATIONS = [
    (transpose, flip_horizontal, flip_vertical)
    for transpose in (False, True)
    for flip_horizontal in (False, True)
    for flip_vertical in (False, True)
]


def apply_microscope_orientation(
    img:              np.ndarray,
    transpose:        bool = False,
    flip_horizontal:  bool = False,
    flip_vertical:    bool = False,
) -> np.ndarray:
    """
    Re-orient a raw camera frame to match MERlin's own microscope-parameters
    convention (``data/configs/merlin/microscope/*.json``'s ``transpose``/
    ``flip_horizontal``/``flip_vertical`` fields).

    Order matters and is fixed: transpose first, then flip_horizontal
    (``np.flip(..., axis=1)``, i.e. mirror columns), then flip_vertical
    (``axis=0``, mirror rows) -- exactly the order used by this project's own
    historical BC341 reference implementation (``transform_image``, see this
    module's docstring), which these same microscope-parameters JSON files
    were written for.
    """
    if transpose:
        img = img.T
    if flip_horizontal:
        img = np.flip(img, axis=1)
    if flip_vertical:
        img = np.flip(img, axis=0)
    return img


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

    Row convention for "up"/"down" (confirmed directly against a real MF3
    dataset, once the image was correctly oriented per its MERlin
    microscope-parameters JSON -- see ``notebooks/misc/
    correct_camera_rotation.ipynb`` section 4): for a correctly-oriented
    frame, row index 0 is the physical -y (down) edge, not +y (up) -- so
    "up" crops the anchor's LAST n rows against the neighbour's FIRST n
    rows. An earlier version of this function assumed the opposite, which
    happened to cancel out with a missing ``flip_vertical`` and looked
    correct by coincidence until the image orientation was fixed to match
    the microscope-parameters JSON.

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
        anchor_img   = apply_microscope_orientation(
            anchor_img, orient_transpose, orient_flip_horizontal, orient_flip_vertical)
        neighbor_img = apply_microscope_orientation(
            neighbor_img, orient_transpose, orient_flip_horizontal, orient_flip_vertical)
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
) -> Tuple[Tuple[bool, bool, bool], "pd.DataFrame"]:
    """
    Audit/fallback search: try all 8 (transpose, flip_horizontal,
    flip_vertical) combinations on a small trial set and report whichever
    gives the SMALLEST median registered-shift magnitude.

    Prefer reading the real transpose/flip_horizontal/flip_vertical values
    from this microscope's own MERlin microscope-parameters JSON
    (``data/configs/merlin/microscope/*.json``) instead of trusting this
    function's output as the primary source -- it exists to CROSS-CHECK that
    file (or substitute for it when unavailable), not replace it. Confirmed
    directly: on a real MF3 dataset, "transpose alone" (missing MERFISH3.
    json's flip_vertical=True) scored best among a smaller 7-candidate
    single-transform search yet still left a visible residual mis-stitch --
    an empirically-best-among-limited-options answer is not the same as the
    actual correct one, which is exactly why the full 8-combination search
    (matching the JSON file's 3 independent booleans) exists here now.

    Why the "smallest median shift" criterion is still reasonable for this
    exhaustive search: real camera-vs-stage rotation is small (well under a
    degree), so two genuinely 4-connected-adjacent FOVs should need only a
    few pixels of correction. A wrong combination has no reason to produce
    uniformly small shifts across many independent, unrelated
    correspondences, so picking the minimum-median combination is a real
    discriminating test, not just noise.

    Parameters
    ----------
    fov_ids, positions, load_frame, step_size_um, pixel_size_um,
    overlap_fraction, tolerance_fraction, upsample_factor : same as
                  :func:`sample_neighbor_correspondences`
    n_trial_anchors : how many anchors to sample PER candidate combination
                  (default 3 -- enough for a clear signal without paying for
                  a full :func:`sample_neighbor_correspondences` run 8 times
                  over)
    seed        : shared across every candidate combination, so all of them
                  are tested on the SAME trial anchors/neighbours -- an
                  apples-to-apples comparison, not different FOVs per
                  candidate

    Returns
    -------
    ((transpose, flip_horizontal, flip_vertical), results_df) -- the winning
    combination, and a DataFrame with one row per candidate (columns:
    transpose, flip_horizontal, flip_vertical, n_correspondences,
    median_shift_um) for a full audit trail.
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
    correspondences by least squares -- the same package this correction's
    own historical precedent (BC341, see this module's docstring) validated.

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
    signal in that particular FOV, an occasional bad phase-correlation peak
    -- the same "bad link" problem BigStitcher's manual workflow required
    curating by hand (see this module's docstring). Confirmed directly on
    real data: the bulk of correspondences cluster tightly (a few um), with
    a handful of clear outliers an order of magnitude or more larger -- a
    real gap in the distribution, not a continuum, so a robust threshold
    cleanly separates them without needing manual review.

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
    Jointly solve for every measured FOV's own real position from all kept
    pairwise neighbour correspondences, instead of fitting one global affine
    transform (:func:`fit_camera_rotation`) applied uniformly to the whole
    nominal grid.

    Why this exists: a single global affine can only correct a rotation/
    scale/shear that's coherent across the WHOLE fov grid -- it cannot
    correct real, independent per-FOV stage-positioning jitter, even given
    perfect measurements. Confirmed directly on real data
    (BC553_sample_02/MF3): one correspondence (anchor 1087, neighbour 1082,
    direction "left") measured a real, clear 5.17 um y-shift, but the pooled
    global-affine fit came out near-identity (no other correspondence
    echoed the same pattern) -- that real measurement never reached the
    "corrected" position at all, it was averaged away rather than applied.
    This is the same problem BigStitcher's own tile-position optimization
    solves for a small contiguous tile block (see this module's docstring
    for why a full dense version doesn't scale to a whole multi-thousand-FOV
    experiment): treat every sampled FOV's position as its own free
    variable, and jointly minimize its disagreement with every
    correspondence that constrains it, instead of reducing every
    measurement to one shared rotation/scale.

    Method
    ------
    For each kept correspondence (anchor A, neighbour B), the measured
    relative offset ``r_AB = measured_xy(B) - nominal_positions[A]`` is a
    direct, independent estimate of B's true position relative to A's own
    nominal position. Solving::

        minimize over every FOV's unknown position p[F]:
            sum_AB || (p[B] - p[A]) - r_AB ||^2

    is a sparse linear least-squares problem that separates cleanly into
    two independent solves (x and y), via ``scipy.sparse.linalg.lsqr``.

    The correspondence graph is typically NOT one connected mesh -- this
    module's own sparse anchor-sampling strategy (:func:`sample_neighbor_
    correspondences`) produces ``N_ANCHORS`` separate ~4-neighbour stars
    that rarely overlap. Each connected component has its own 1-D-per-axis
    translational null space (uniformly shifting every position in it
    satisfies every constraint in that component equally), removed by
    pinning ONE FOV per component -- whichever appears as an ``anchor_fov``
    in the most correspondences, i.e. a real sampled anchor with several
    real measurements attached, not an arbitrary leaf -- to its own nominal
    position.

    Honest limitation: with this sparse, non-overlapping sampling, most
    components are simple stars (one anchor + up to 4 leaves, each leaf
    constrained by exactly one correspondence) -- the solved leaf position
    is then numerically identical to that correspondence's own
    ``measured_xy``, and ``residual_rms_um`` is 0.0 (nothing to disagree
    with). The real benefit here is using each measured FOV's own direct
    measurement instead of discarding it into a diluted global-affine
    average -- not yet genuine cross-measurement error averaging, which
    would need denser/overlapping sampling to provide redundant constraints
    per FOV.

    ``lsqr``'s own default convergence tolerances are too loose once this IS
    run on a dense, overlapping correspondence set (e.g. every FOV of a
    several-hundred-FOV grid measured against most/all of its real
    neighbours, as `notebooks/tests/compare_stitching_correction_methods.
    ipynb` does) -- confirmed directly on real data (BC555_sample_05/epi,
    476 FOVs, 1662 kept correspondences, one connected component): calling
    ``lsqr`` with no explicit ``atol``/``btol`` (i.e. scipy's own defaults)
    declared convergence with ``residual_rms_um`` = 81.6 -- roughly 25x the
    real ~3um signal this whole method exists to resolve -- while
    ``atol=btol=1e-12`` on the exact same input converges properly
    (``istop`` 1 or 2, a genuine "good enough" stop, not an iteration-limit
    cutoff) to 0.025um, a two-thousand-fold tighter, far more physically
    plausible residual. The small, sparse-star components this function was
    originally written for never exposed this: a handful of unknowns
    converges to any reasonable tolerance in a few iterations regardless, so
    the default tolerance being loose never mattered until a large, densely
    connected system was actually solved. Do not loosen these below their
    own defaults without re-confirming convergence the same way (``istop``
    close to 1/2, not 7 (iteration limit) or 3/4 (ill-conditioned) --
    raising ``PIN_WEIGHT`` far past its current value chases the same
    convergence problem: at ``1e8`` on this same dataset, ``lsqr`` hit
    ``istop=3`` (excessive condition number) after only 5 iterations and the
    residual got WORSE (206.8), not better).

    Parameters
    ----------
    correspondences   : from :func:`sample_neighbor_correspondences`,
                        already passed through
                        :func:`filter_correspondence_outliers`
    nominal_positions : ``{fov_id: (x, y)}`` -- the full experiment's
                        nominal grid positions (needed to look up each
                        correspondence's ANCHOR's own nominal position,
                        which isn't stored on the correspondence itself --
                        only the neighbour's is)
    lsqr_atol, lsqr_btol : passed straight through to
                        ``scipy.sparse.linalg.lsqr`` for both the x and y
                        solves -- see the convergence note above before
                        loosening these.

    Returns
    -------
    GlobalPositionCorrection -- see its own docstring. Merge ``.positions``
    over a full nominal (or affine-corrected) positions dict as a fallback
    for every FOV not directly measured.
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
    Place every FOV by walking outward from *root_fov*, at each step always
    taking the highest-priority (most reliable direction) correspondence
    that reaches a not-yet-placed FOV from an already-placed one -- Prim's
    algorithm for a maximum-priority spanning tree, where "weight" is a
    whole DIRECTION's own reliability (e.g. that direction's std of
    ``measured - nominal`` deviation across many independent
    correspondences), not any single measurement's own noise.

    Contrast with :func:`fit_global_positions`: that function uses every
    kept correspondence AT ONCE and lets disagreements average out via least
    squares -- principled, but blind to "this whole direction is generally
    noisier" (it only sees per-measurement disagreement, and with one
    measurement per FOV -- the common sparse-sampling case -- there is
    nothing to average against at all). This function instead uses exactly
    ONE correspondence to place any given FOV -- whichever available one
    belongs to the currently-most-reliable direction -- and simply never
    uses any other correspondence that also reaches that FOV (a "cut" edge).
    Well-suited to a DENSE, exhaustive correspondence set (every FOV
    measured against most/all of its real neighbours) where a genuine,
    informative choice between directions exists at almost every step;
    degrades to plain BFS (first-reached wins) if *direction_reliability* is
    ``None``.

    Parameters
    ----------
    correspondences        : from :func:`sample_neighbor_correspondences`,
                              ideally already passed through
                              :func:`filter_correspondence_outliers`
    nominal_positions       : ``{fov_id: (x, y)}`` -- the full grid's nominal
                              positions; used to look up each
                              correspondence's ANCHOR's own nominal position
                              (needed to recover the real relative offset,
                              same as :func:`fit_global_positions`), and as
                              the fallback position for any FOV this walk
                              never reaches
        direction_reliability : ``{direction: score}``, LOWER = more reliable
                              (e.g. that direction's own std of
                              ``measured - nominal`` deviation, from a
                              reliability scatter plot). ``None`` (default)
                              treats every direction equally.
    root_fov                : the FOV held fixed at its own nominal position
                              (default 0)

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
