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

from .alignment import apply_orientation, phase_drift, remove_hot_pixels
from .positions import find_grid_neighbor

log = logging.getLogger(__name__)

_DIRECTIONS = ("right", "left", "up", "down")

# Every orientation apply_orientation() accepts -- the candidate pool
# detect_image_orientation() searches over. Whether a camera's raw frame rows/
# columns line up with physical stage x/y (and which way) is NOT standardised
# across microscopes/mounting -- confirmed directly on a real MF3 dataset that
# the naive assumption (row=y, col=x, no flip) produces wildly wrong, physically
# implausible registrations (tens to hundreds of pixels, when true camera
# rotation should shift an adjacent FOV by at most a few pixels), while
# "transpose" produced small, consistent shifts across multiple independent
# correspondence pairs and all four directions. Never assume "none" is correct
# for a new microscope -- run detect_image_orientation() first.
_IMAGE_ORIENTATIONS = ("none", "fliplr", "flipud", "transpose", "rot90", "rot180", "rot270")


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
            return anchor_img[:n, :], neighbor_img[h - n:, :]
        return anchor_img[h - n:, :], neighbor_img[:n, :]


def register_neighbor_pair(
    anchor_img:       np.ndarray,
    neighbor_img:     np.ndarray,
    anchor_xy:        Tuple[float, float],
    neighbor_xy:      Tuple[float, float],
    direction:        str,
    overlap_fraction: float,
    pixel_size_um:    float,
    upsample_factor:    int = 10,
    image_orientation:  str = "none",
) -> Tuple[Tuple[float, float], float]:
    """
    Measure the neighbour's TRUE position relative to the anchor, from the
    real pixel shift needed to align their overlapping border crop.

    Parameters
    ----------
    image_orientation : one of :data:`_IMAGE_ORIENTATIONS`, applied to BOTH
                  images (via :func:`MERci.acquisition.alignment.
                  apply_orientation`) before cropping/registering -- corrects
                  for this camera's raw-frame row/column axes not lining up
                  with physical stage x/y the way ``crop_overlap`` assumes
                  (row=y, col=x, no flip). Camera/mounting-specific and NOT
                  safe to assume "none" -- see :func:`detect_image_orientation`.

    Returns
    -------
    (measured_neighbor_xy, error) -- the neighbour's measured true (x, y)
    stage position (µm), and the registration's normalised RMS error.
    """
    if image_orientation != "none":
        anchor_img   = apply_orientation(anchor_img, image_orientation)
        neighbor_img = apply_orientation(neighbor_img, image_orientation)
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
    image_orientation:  str = "none",
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
    image_orientation : passed through to :func:`register_neighbor_pair` --
                  see :func:`detect_image_orientation` to determine the
                  right value for a given microscope before trusting any
                  correspondence this function returns.
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
                image_orientation,
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
) -> Tuple[str, "pd.DataFrame"]:
    """
    Determine which :data:`_IMAGE_ORIENTATIONS` value actually matches this
    camera's real row/column-vs-stage-x/y convention, by trying each one on a
    small trial set and keeping whichever gives the SMALLEST median
    registered-shift magnitude.

    Why this is a reasonable criterion: real camera-vs-stage rotation is
    small (typically well under a degree), so two genuinely 4-connected-
    adjacent FOVs should need only a few pixels of correction at most. Tried
    directly on a real MF3 dataset: the (wrong) default assumption
    ("none") produced correspondences ranging from ~6 to ~110 um -- a smooth,
    unbroken spread with no small-and-good cluster to separate out by
    thresholding -- while "transpose" gave small (1-5 um), consistent shifts
    across every direction and several independent anchor pairs. An
    orientation that's actually wrong has no reason to produce uniformly
    small shifts across many independent, unrelated correspondences, so
    picking the minimum-median orientation is a real discriminating test,
    not just noise.

    Parameters
    ----------
    fov_ids, positions, load_frame, step_size_um, pixel_size_um,
    overlap_fraction, tolerance_fraction, upsample_factor : same as
                  :func:`sample_neighbor_correspondences`
    n_trial_anchors : how many anchors to sample PER candidate orientation
                  (default 3 -- enough for a clear signal without paying for
                  a full :func:`sample_neighbor_correspondences` run 7 times
                  over)
    seed        : shared across every candidate orientation, so all of them
                  are tested on the SAME trial anchors/neighbours -- an
                  apples-to-apples comparison, not different FOVs per
                  candidate

    Returns
    -------
    (best_orientation, results_df) -- the winning orientation string, and a
    DataFrame with one row per candidate orientation (columns: orientation,
    n_correspondences, median_shift_um) for a full audit trail.
    """
    import pandas as pd

    rows = []
    for orientation in _IMAGE_ORIENTATIONS:
        trial = sample_neighbor_correspondences(
            fov_ids=fov_ids, positions=positions, load_frame=load_frame,
            step_size_um=step_size_um, pixel_size_um=pixel_size_um,
            overlap_fraction=overlap_fraction, n_anchors=n_trial_anchors,
            tolerance_fraction=tolerance_fraction, upsample_factor=upsample_factor,
            image_orientation=orientation, seed=seed,
        )
        if not trial:
            rows.append({"orientation": orientation, "n_correspondences": 0, "median_shift_um": np.inf})
            continue
        shifts_um = [
            float(np.hypot(c.measured_xy[0] - c.nominal_xy[0], c.measured_xy[1] - c.nominal_xy[1]))
            for c in trial
        ]
        rows.append({
            "orientation": orientation, "n_correspondences": len(trial),
            "median_shift_um": float(np.median(shifts_um)),
        })

    results_df = pd.DataFrame(rows).sort_values("median_shift_um").reset_index(drop=True)
    best_orientation = str(results_df.iloc[0]["orientation"])
    return best_orientation, results_df


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
