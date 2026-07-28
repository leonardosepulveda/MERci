# MERci/analysis/ffc.py
"""
Flat-field correction (FFC) for round mosaics.

A per-channel, per-pixel illumination/vignette profile, estimated once per
experiment (not per round -- vignetting is a fixed property of the
microscope/channel) from a small subset of FOVs, then divided out of every
FOV before it's placed into a round mosaic (see :mod:`MERci.analysis.round`'s
``create_mosaic_ffc``/``load_raw_frames_for_round`` and
:func:`MERci.scheduler.build_round_mosaics`).

Does NOT touch the per-FOV ``analyze_file``/``create_thumbnail`` pipeline
used for live stats/histograms during acquisition -- FFC only affects the
round-mosaic path.

Three interchangeable FOV/frame selection strategies feed the same
:func:`compute_ffc_field_for_color`, which just accumulates over a flat list
of ``(path, frame_idx)`` samples:

- ``"exterior_grid"``    : FOVs on the exterior of the imaged FOV grid
                           (:func:`MERci.acquisition.positions.find_exterior_fovs`),
                           one mid-z frame each.
- ``"emptiest_stats"``   : the N FOVs with the lowest mean+variance, read
                           from already-computed per-FOV stats CSVs (no new
                           raw reads needed just to select candidates).
- ``"single_fov_all_frames"`` : every z-frame of one near-empty FOV, treated
                           as independent samples of the same illumination
                           profile (vignetting doesn't depend on z).

Which strategy is fastest/accurate-enough in practice is an open empirical
question -- see ``notebooks/misc/investigate_ffc_sample_size.ipynb``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── FOV/frame selection strategies ────────────────────────────────────────────

def select_ffc_exterior_fovs(
    round_id: int,
    config,                             # ExperimentConfig
    metadata,                           # ExperimentMetadata
    frame_idx: int,
    series_idx: int = 0,
) -> List[Tuple[Path, int]]:
    """
    Exterior FOVs of *round_id*'s own imaged FOV grid (outer perimeter + any
    hole boundaries), each paired with *frame_idx*.

    Uses :func:`MERci.acquisition.positions.find_exterior_fovs` scoped to
    this round's own real imaged-FOV positions (not the raw experiment-wide
    positions.txt), so transit-only FOVs (blank frames) never enter the set.
    """
    from MERci.acquisition.positions import find_exterior_fovs

    round_info = metadata.rounds.get(round_id)
    if round_info is None:
        raise KeyError(f"Round {round_id} not found in metadata")

    positions = {
        fov_id: metadata.fovs[fov_id].position
        for fov_id in round_info.fov_files
        if round_info.fov_files[fov_id]
    }
    exterior_ids = find_exterior_fovs(
        positions, config.step_size_um,
        connectivity=config.ffc_connectivity,
        tolerance_fraction=config.ffc_neighbor_tolerance,
    )

    samples = []
    for fov_id in exterior_ids:
        file_list = round_info.fov_files[fov_id]
        idx = min(series_idx, len(file_list) - 1)
        samples.append((file_list[idx], frame_idx))
    return samples


def select_emptiest_fovs(
    round_id: int,
    config,                             # ExperimentConfig
    metadata,                           # ExperimentMetadata
    tracker,                            # ProgressTracker
    frame_idx: int,
    n_fovs: int,
    series_idx: int = 0,
) -> List[Tuple[Path, int]]:
    """
    The *n_fovs* FOVs in *round_id* with the lowest (mean, std) at
    *frame_idx*, read from each FOV's already-computed stats CSV
    (:func:`MERci.analysis.fov.measure_stats`'s output, one row per frame) --
    no new raw image reads needed just to rank candidates.

    FOVs missing a stats file (analysis not complete yet) are skipped with a
    warning rather than raising, since this is only a candidate search.
    """
    round_info = metadata.rounds.get(round_id)
    if round_info is None:
        raise KeyError(f"Round {round_id} not found in metadata")

    ranked = []
    for fov_id, file_list in round_info.fov_files.items():
        if not file_list:
            continue
        idx = min(series_idx, len(file_list) - 1)
        fpath = file_list[idx]
        stats_path = tracker.stats_path(fpath)
        if not stats_path.exists():
            log.warning("Stats file missing, skipping for FFC candidate ranking: %s", stats_path)
            continue
        stats_df = pd.read_csv(stats_path)
        row = stats_df[stats_df["frame"] == frame_idx]
        if row.empty:
            continue
        mean = float(row["mean"].iloc[0])
        std  = float(row["std"].iloc[0])
        ranked.append((mean, std, fpath))

    ranked.sort(key=lambda t: (t[0], t[1]))
    return [(fpath, frame_idx) for _, _, fpath in ranked[:n_fovs]]


def select_all_frames_of_fov(
    fov_path: Path,
    frame_table: pd.DataFrame,
    color: float,
    bead_z: float = 0.0,
) -> List[Tuple[Path, int]]:
    """
    Every z-frame of *color* within one FOV file, as independent FFC samples
    (vignetting is a property of the optics, not of z).
    """
    from MERci.acquisition.configs import get_all_color_frame_indices

    frame_indices = get_all_color_frame_indices(frame_table, color, bead_z)
    return [(fov_path, fi) for fi in frame_indices]


# ── Field computation ──────────────────────────────────────────────────────────

def compute_ffc_field_for_color(
    samples: List[Tuple[Path, int]],
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    smooth_sigma_px: float = 50.0,
    normalize_percentile: float = 99.99,
    ffc_min_value: float = 0.10,
) -> Tuple[np.ndarray, dict]:
    """
    Estimate a flat-field-correction field from *samples* -- a flat list of
    ``(path, frame_idx)`` pairs, each read and accumulated one at a time (so
    memory use never exceeds one raw frame, regardless of how many samples
    are given).

    Steps: accumulate a running per-pixel mean over every sample -> Gaussian-
    smooth (suppresses residual tissue-signal noise, more useful the smaller
    the sample) -> normalize by the smoothed field's *normalize_percentile*-th
    percentile (so a well-illuminated pixel ends up near 1.0) -> floor-clip to
    *ffc_min_value* (guards near-zero divisors at the field's dim edges).

    Parameters
    ----------
    samples              : [(image_path, frame_idx), ...]
    frame_width/height    : passed to the raw-frame reader
    smooth_sigma_px       : Gaussian smoothing sigma, in pixels
    normalize_percentile  : percentile of the smoothed field used to
                            normalize it to ~1.0 peak
    ffc_min_value         : floor clip applied after normalization

    Returns
    -------
    field : float32 (H, W) flat-field-correction map
    meta  : {"n_samples": int, "smooth_sigma_px": float,
             "normalize_percentile": float, "ffc_min_value": float}
    """
    from scipy.ndimage import gaussian_filter
    from MERci.common.io import read_image_frames

    if not samples:
        raise ValueError("samples list is empty")

    total = None
    for path, frame_idx in samples:
        frame = read_image_frames(path, [frame_idx], frame_width, frame_height)[0]
        frame = frame.astype(np.float64)
        if total is None:
            total = frame
        else:
            total += frame

    field = (total / len(samples)).astype(np.float32)
    field = gaussian_filter(field, sigma=smooth_sigma_px)

    norm_value = np.percentile(field, normalize_percentile)
    if norm_value > 0:
        field = field / norm_value
    field = np.clip(field, ffc_min_value, None).astype(np.float32)

    meta = {
        "n_samples": len(samples),
        "smooth_sigma_px": smooth_sigma_px,
        "normalize_percentile": normalize_percentile,
        "ffc_min_value": ffc_min_value,
    }
    return field, meta


def apply_ffc(frame: np.ndarray, ffc_field: np.ndarray) -> np.ndarray:
    """Divide *frame* by *ffc_field*, clipping negative results (from any
    upstream dark-offset subtraction) to zero."""
    return np.clip(frame.astype(np.float32) / ffc_field, 0, None)


def compute_mosaic_crop_px(config) -> int:
    """
    Pixels to crop from every edge of a raw FOV frame before placing it in a
    mosaic, to drop the overlap border shared with neighbouring FOVs.

    Generic replacement for a fixed hardcoded crop: the overlap width in
    pixels is ``image_size_px * (1 - non_overlap_fraction)``, shared
    symmetrically between two adjacent tiles, so half of that is cropped
    from each side.
    """
    overlap_px = config.image_size_px * (1 - config.non_overlap_fraction)
    return int(round(overlap_px / 2))


# ── Caching ────────────────────────────────────────────────────────────────────

def save_ffc_field(path: Path, field: np.ndarray, meta: dict) -> None:
    """Save *field* + *meta* to an ``.npz`` file (mirrors
    :func:`MERci.analysis.fov.save_channel_counters`'s convention)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, field=field.astype(np.float32), **meta)


def load_ffc_field(path: Path) -> Tuple[np.ndarray, dict]:
    """Load a field saved by :func:`save_ffc_field`."""
    npz = np.load(Path(path), allow_pickle=True)
    field = npz["field"]
    meta = {k: npz[k].item() if npz[k].ndim == 0 else npz[k] for k in npz.files if k != "field"}
    return field, meta


# ── Top-level entry point ─────────────────────────────────────────────────────

def resolve_ffc_reference_round(
    config,                             # ExperimentConfig
    metadata,                           # ExperimentMetadata
    tracker,                            # ProgressTracker
    color: float,
) -> Optional[int]:
    """
    First round_id (ascending) in which *color* both appears (per
    :func:`MERci.scheduler.resolve_round_color_frame_indices`) and is fully
    imaged (:func:`ProgressTracker.all_fovs_done_for_round`). Resolved
    per-color, not once globally, since different rounds can use different
    colors (e.g. a cells round's 405 vs. a bits round's 750/650/560).

    Returns ``None`` if no such round exists yet.
    """
    from MERci.scheduler import resolve_round_color_frame_indices

    for round_id in metadata.valid_round_ids():
        if not tracker.all_fovs_done_for_round(round_id, metadata, config.fov_subset):
            continue
        color_indices = resolve_round_color_frame_indices(round_id, config, metadata)
        if color in color_indices:
            return round_id
    return None


def compute_and_cache_ffc(
    config,                             # ExperimentConfig
    metadata,                           # ExperimentMetadata
    tracker,                            # ProgressTracker
    color: float,
) -> Optional[Path]:
    """
    Ensure the FFC field for *color* is computed and cached, computing it
    exactly once per experiment (idempotent -- safe to call from every
    ``build_round_mosaics`` call for every round).

    Returns the cache path, or ``None`` (not an error) if no round is ready
    yet, or the selected sample count is below ``config.ffc_min_samples`` --
    in either case mosaics for that color are built without FFC that round
    and this is retried on the next round.
    """
    from MERci.scheduler import resolve_round_color_frame_indices

    cache_path = tracker.ffc_field_path(color)
    if tracker.is_ffc_done(color):
        return cache_path

    ref_round = resolve_ffc_reference_round(config, metadata, tracker, color)
    if ref_round is None:
        log.warning("No fully-imaged round yet has color %snm -- FFC not computed.", color)
        return None

    color_indices = resolve_round_color_frame_indices(ref_round, config, metadata)
    frame_idx = color_indices[color]

    strategy = config.ffc_fov_selection_strategy
    if strategy == "exterior_grid":
        samples = select_ffc_exterior_fovs(ref_round, config, metadata, frame_idx)
    elif strategy == "emptiest_stats":
        samples = select_emptiest_fovs(
            ref_round, config, metadata, tracker, frame_idx, config.ffc_emptiest_n_fovs,
        )
    elif strategy == "single_fov_all_frames":
        candidates = select_emptiest_fovs(ref_round, config, metadata, tracker, frame_idx, 1)
        if not candidates:
            samples = []
        else:
            from MERci.acquisition.configs import find_frame_table_for_hal_config
            fov_path = candidates[0][0]
            series = next(
                (s for s in metadata.series_for_round(ref_round) if s.hal_config), None,
            )
            frame_table = None
            if series is not None and config.settings_dir is not None:
                ft_path = find_frame_table_for_hal_config(
                    config.settings_dir / series.hal_config, config.metadata_dir,
                )
                if ft_path is not None:
                    frame_table = pd.read_csv(ft_path, index_col=0)
            samples = (
                select_all_frames_of_fov(fov_path, frame_table, color)
                if frame_table is not None else [(fov_path, frame_idx)]
            )
    else:
        raise ValueError(f"Unknown ffc_fov_selection_strategy: {strategy!r}")

    if len(samples) < config.ffc_min_samples:
        log.warning(
            "Only %d FFC sample(s) for color %snm (< ffc_min_samples=%d) -- "
            "FFC not computed this round; will retry on a later round.",
            len(samples), color, config.ffc_min_samples,
        )
        return None

    field, meta = compute_ffc_field_for_color(
        samples, config.frame_width, config.frame_height,
        config.ffc_smooth_sigma_px, config.ffc_normalize_percentile, config.ffc_min_value,
    )
    save_ffc_field(cache_path, field, {**meta, "round_id": ref_round, "color": color})
    tracker.mark_ffc_done(color)
    log.info("FFC field computed for %snm from %d sample(s) (round %d): %s",
              color, len(samples), ref_round, cache_path)
    return cache_path
