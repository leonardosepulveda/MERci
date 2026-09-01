# MERci/analysis/fast_spot_quantification.py
"""
Logic behind ``notebooks/during_imaging/fast_spot_quantification.ipynb`` --
per-bit hybridization-reagent QC. A compromised bit's foci can look visibly
dimmer/sparser than its neighbours, and catching that early (while the round
could still be re-imaged/re-hybridized) is much cheaper than finding out
after the whole experiment is decoded.

For a small set of representative FOVs, measures each detected focus's
background-subtracted peak intensity per (round, color): sample
``n_z_samples`` z-planes evenly spaced across that round's real z-sweep for
the color, crop to the FOV's own center (away from edge vignetting),
max-project, then detect foci via
:func:`MERci.analysis.spot_localization.detect_beads_2d` (the same
background-subtraction + local-maxima primitive this repo already uses for
bead/fiducial detection, reused here for real hybridization foci instead).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ..common.config import ExperimentConfig
from ..common.metadata import ExperimentMetadata, SeriesInfo
from ..common.io import read_image_frames
from ..acquisition.configs import find_frame_table_for_hal_config, get_all_color_frame_indices
from .spot_localization import detect_beads_2d


def sample_z_frame_indices(all_indices: Sequence[int], n_samples: int) -> List[int]:
    """Evenly-spaced picks from *all_indices* (already ascending z order) --
    if there are fewer real z-planes than requested, just use all of them."""
    if len(all_indices) <= n_samples:
        return list(all_indices)
    positions = np.linspace(0, len(all_indices) - 1, n_samples)
    return sorted({all_indices[int(round(p))] for p in positions})


def resolve_round_color_frame_indices(
    round_id: int, config: ExperimentConfig, metadata: ExperimentMetadata,
    excluded_colors: Sequence[float], n_z_samples: int,
) -> Dict[float, List[int]]:
    """
    Return ``{color_nm: [frame_idx, ...]}`` for *round_id*'s own frame table
    -- every real color it has (except *excluded_colors* and blanks, which
    have no color at all), each with *n_z_samples* sampled frame indices
    (:func:`sample_z_frame_indices`) evenly spaced across
    :func:`MERci.acquisition.configs.get_all_color_frame_indices`'s full
    ascending-z list for that color.
    """
    color_frames: Dict[float, List[int]] = {}
    for s in metadata.series_for_round(round_id):
        if not s.hal_config:
            continue
        frame_table_path = find_frame_table_for_hal_config(
            config.settings_dir / s.hal_config, config.metadata_dir)
        if frame_table_path is None:
            continue
        frame_table = pd.read_csv(frame_table_path)
        for color in sorted(frame_table["color"].dropna().unique()):
            if any(round(color) == round(excluded) for excluded in excluded_colors):
                continue
            all_indices = get_all_color_frame_indices(frame_table, float(color))
            if not all_indices:
                continue
            color_frames[float(color)] = sample_z_frame_indices(all_indices, n_z_samples)
    return color_frames


def crop_center(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Square crop of side *crop_size*, centered on *frame*."""
    h, w = frame.shape
    y0 = max(0, h // 2 - crop_size // 2)
    x0 = max(0, w // 2 - crop_size // 2)
    y1 = min(h, y0 + crop_size)
    x1 = min(w, x0 + crop_size)
    return frame[y0:y1, x0:x1]


def compute_background_median(image: np.ndarray, bg_percentile: float = 80) -> float:
    """
    Gaussian-smoothed, bottom-*bg_percentile*-percentile background median --
    the same convention :func:`MERci.analysis.spot_localization.detect_beads_2d`
    itself uses internally for its detection threshold, kept consistent here
    so the *subtracted* background matches the *detection* background.
    """
    blurred = gaussian_filter(image.astype(float), sigma=1.5)
    bg_mask = blurred < np.percentile(blurred, bg_percentile)
    return float(np.median(blurred[bg_mask])) if bg_mask.any() else float(np.median(blurred))


def detect_foci_in_crop(frames: np.ndarray, crop_size: int, min_dist_px: float, thresh_sigma: float):
    """
    *frames*: ``(n_z, H, W)`` raw array for one (fov, round, color). Crops
    every z-plane to *crop_size*, max-projects, then detects candidate foci.

    Returns
    -------
    max_proj   : the cropped max-projection (float32)
    bg_med     : background median (:func:`compute_background_median`)
    candidates : ``(N, 2)`` ``[row, col]`` array, from :func:`detect_beads_2d`
    """
    cropped = np.stack([crop_center(f, crop_size) for f in frames], axis=0)
    max_proj = cropped.max(axis=0).astype(np.float32)
    bg_med = compute_background_median(max_proj)
    candidates = detect_beads_2d(max_proj, min_dist_px, thresh_sigma)
    return max_proj, bg_med, candidates


def spot_cache_path(output_dir: Path, round_id: int, color_nm: float, fov_id: int) -> Path:
    return output_dir / f"round{round_id:03d}_{color_nm:.0f}nm_fov{fov_id:04d}.csv"


def compute_fov_round_color_spots(
    fov_id: int, round_id: int, color_nm: float, frame_indices: List[int],
    series: List[SeriesInfo], config: ExperimentConfig,
    crop_size: int, min_dist_px: float, thresh_sigma: float,
) -> Optional[pd.DataFrame]:
    """
    Detected foci for one (fov, round, color) combination, or None if
    *fov_id* isn't imaged yet for this round -- so the caller can retry on a
    later run rather than caching an empty/wrong result.
    """
    existing = [p for p in (s.resolve_path(fov_id, config.image_suffix) for s in series) if p.exists()]
    if not existing:
        return None
    frames = read_image_frames(existing[0], frame_indices,
                                frame_width=config.frame_width, frame_height=config.frame_height)
    max_proj, bg_med, candidates = detect_foci_in_crop(frames, crop_size, min_dist_px, thresh_sigma)
    rows = [
        {"fov": fov_id, "round": round_id, "color_nm": color_nm,
         "row_px": int(r), "col_px": int(c), "intensity": float(max_proj[r, c] - bg_med)}
        for (r, c) in candidates
    ]
    return pd.DataFrame(rows, columns=["fov", "round", "color_nm", "row_px", "col_px", "intensity"])


def load_all_spots(
    output_dir: Path, round_color_frame_indices: Dict[int, Dict[float, List[int]]],
    selected_fovs: Sequence[int],
) -> pd.DataFrame:
    """Concatenate every cached spot CSV (:func:`spot_cache_path`) for the
    given rounds/colors/FOVs that has actually been computed so far."""
    frames = []
    for round_id, color_frames in round_color_frame_indices.items():
        for color_nm in color_frames:
            for fov_id in selected_fovs:
                cache_path = spot_cache_path(output_dir, round_id, color_nm, fov_id)
                if cache_path.exists():
                    frames.append(pd.read_csv(cache_path))
    if not frames:
        return pd.DataFrame(columns=["fov", "round", "color_nm", "row_px", "col_px", "intensity"])
    return pd.concat(frames, ignore_index=True)
