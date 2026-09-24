# MERci/analysis/imaged_fovs.py
"""
Logic behind ``notebooks/during_imaging/imaged_fovs.ipynb`` -- picking which
round a live acquisition-progress view should watch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from ..common.config import ExperimentConfig
from ..common.io import path_mtime as io_path_mtime
from ..common.metadata import ExperimentMetadata


def path_mtime(path: Path) -> float:
    """:func:`MERci.common.io.path_mtime`, falling back to the store's own
    mtime while a directory store is still empty."""
    try:
        return io_path_mtime(path)
    except FileNotFoundError:
        return Path(path).stat().st_mtime


def round_progress(round_id: int, config: ExperimentConfig, metadata: ExperimentMetadata) -> Tuple[int, Optional[float]]:
    """``(n_imaged, latest_mtime_or_None)`` for *round_id*, over every planned FOV."""
    series = metadata.series_for_round(round_id)
    n_imaged, latest = 0, None
    for fov_id in sorted(metadata.fovs):
        paths = [s.resolve_path(fov_id, config.image_suffix) for s in series]
        existing = [p for p in paths if p.exists()]
        if existing:
            n_imaged += 1
            mtime = max(path_mtime(p) for p in existing)
            latest = mtime if latest is None else max(latest, mtime)
    return n_imaged, latest


def detect_active_round(config: ExperimentConfig, metadata: ExperimentMetadata) -> int:
    """
    Auto-detect which round is currently being written: prefer a round with
    some but not all FOVs already imaged (i.e. one HAL/Dave is actively
    writing right now); fall back to the round right after the most
    recently active one (e.g. during the fluidics gap between one round
    finishing and the next starting, which otherwise has zero files and is
    invisible to this scan), then the first round, if nothing is imaged
    anywhere yet.
    """
    best_round, best_latest, best_in_progress = None, -1.0, False
    for round_id in sorted(metadata.rounds):
        n_imaged, latest = round_progress(round_id, config, metadata)
        if latest is None:
            continue
        in_progress = n_imaged < metadata.n_fovs
        if (in_progress, latest) > (best_in_progress, best_latest):
            best_round, best_latest, best_in_progress = round_id, latest, in_progress

    if best_round is None:
        return sorted(metadata.rounds)[0]   # nothing imaged anywhere yet -- start of the sequence

    if not best_in_progress:
        # Nothing is actively in progress -- best_round is just the most
        # recently completed one. Point at the round right after it instead,
        # unless best_round is already the last round in the experiment.
        round_ids = sorted(metadata.rounds)
        idx = round_ids.index(best_round)
        if idx + 1 < len(round_ids):
            return round_ids[idx + 1]

    return best_round
