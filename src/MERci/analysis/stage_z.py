# MERci/analysis/stage_z.py
"""
Stage-z drift QC.

Every acquired movie writes a HAL ``.off`` focus-lock log alongside its image
file (same directory, same stem — the same sidecar convention as ``.inf``,
see ``common.io.get_dax_shape``): a whitespace-delimited table, one row per
frame, columns ``frame offset power stage-z good-offset``. The focus lock is
expected to hold ``stage-z`` constant for a whole FOV's stack; this module
summarizes that column per FOV (so a notebook can check how much it
actually drifts over the course of a long acquisition) and caches the
result on disk, so a multi-thousand-FOV experiment's ``.off`` files are each
read only once, ever. It also summarizes ``good-offset`` (HAL's own
per-frame focus-lock-quality flag) via ``summarize_focus_lock``/
``focus_lock_summary_for_fov``, e.g. for reading back results from
``MERci.acquisition.dave.create_focus_test_dave_config``'s optional
per-FOV test movies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_COLUMNS = [
    "round_id", "fov_id", "series", "off_path",
    "first_stage_z", "min_stage_z", "max_stage_z", "all_same",
]


def off_path_for(image_path: Path) -> Path:
    """Return the ``.off`` sidecar path for *image_path* (same dir, same stem)."""
    return Path(image_path).with_suffix(".off")


def read_off_file(off_path: Path) -> pd.DataFrame:
    """
    Parse a HAL ``.off`` focus-lock log: whitespace-delimited, one row per
    frame, columns ``frame offset power stage-z good-offset``.
    """
    return pd.read_csv(off_path, sep=r"\s+", engine="python")


def summarize_stage_z(off_df: pd.DataFrame) -> dict:
    """
    Summarize the ``stage-z`` column of one FOV's ``.off`` table.

    Returns ``first_stage_z``/``min_stage_z``/``max_stage_z`` (µm, same units
    as the ``stage-z`` column) and ``all_same`` — whether every row shares
    the first row's exact value, true in most cases since the focus lock
    holds stage-z fixed for the whole stack.
    """
    values = off_df["stage-z"]
    first = float(values.iloc[0])
    return {
        "first_stage_z": first,
        "min_stage_z":   float(values.min()),
        "max_stage_z":   float(values.max()),
        "all_same":      bool((values == first).all()),
    }


def stage_z_summary_for_fov(image_path: Path) -> Optional[dict]:
    """
    Read *image_path*'s ``.off`` sidecar and summarize its ``stage-z``
    column, or ``None`` if the sidecar doesn't exist yet (not written, or
    this series/FOV combination doesn't apply).
    """
    off_path = off_path_for(image_path)
    if not off_path.exists():
        return None
    return summarize_stage_z(read_off_file(off_path))


def summarize_focus_lock(off_df: pd.DataFrame) -> dict:
    """
    Summarize the ``good-offset`` column of one FOV's ``.off`` table -- HAL's
    own per-frame "was the focus lock good at this instant" flag (see
    ``storm_control/hal4000/focusLock/lockControl.py``'s ``handleNewFrame``,
    which writes it as ``is_good`` alongside ``stage-z``). Only meaningful
    for a movie that actually took at least one frame -- a pure
    check-focus-only test FOV (see
    :func:`MERci.acquisition.dave.create_focus_test_dave_config`) writes no
    ``.off`` file at all, since HAL only opens it once real frames arrive.
    """
    good = off_df["good-offset"].astype(bool)
    return {
        "n_frames":     int(len(good)),
        "n_bad_frames": int((~good).sum()),
        "all_good":     bool(good.all()),
    }


def focus_lock_summary_for_fov(image_path: Path) -> Optional[dict]:
    """
    Read *image_path*'s ``.off`` sidecar and summarize its ``good-offset``
    column, or ``None`` if the sidecar doesn't exist (no movie was ever
    taken here -- e.g. a check-focus-only test FOV, or one not yet run).
    """
    off_path = off_path_for(image_path)
    if not off_path.exists():
        return None
    return summarize_focus_lock(read_off_file(off_path))


def load_stage_z_cache(cache_path: Path) -> pd.DataFrame:
    """Load the on-disk stage-z summary cache, or an empty frame if none exists yet."""
    if Path(cache_path).exists():
        return pd.read_csv(cache_path)
    return pd.DataFrame(columns=CACHE_COLUMNS)


def update_stage_z_cache(config, meta, cache_path: Path) -> pd.DataFrame:
    """
    Extend the on-disk stage-z cache with every (round, FOV, series)
    combination not already in it, reading each ``.off`` sidecar only once
    across the cache's whole lifetime. Respects ``config.fov_subset``.
    Returns the full, updated cache (existing rows + any newly read ones).
    """
    cache = load_stage_z_cache(cache_path)
    seen  = set(zip(cache["round_id"], cache["fov_id"], cache["series"]))
    fov_ids = (
        config.fov_subset if config.fov_subset is not None else sorted(meta.fovs)
    )
    new_rows = []

    for round_id in meta.valid_round_ids():
        for s in meta.series_for_round(round_id):
            for fov_id in fov_ids:
                key = (round_id, fov_id, s.name)
                if key in seen:
                    continue
                image_path = s.resolve_path(fov_id, meta.image_suffix)
                summary = stage_z_summary_for_fov(image_path)
                if summary is None:
                    continue   # .off not written yet -- pick it up on a future run
                new_rows.append({
                    "round_id": round_id, "fov_id": fov_id, "series": s.name,
                    "off_path": str(off_path_for(image_path)), **summary,
                })

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(cache_path, index=False)
    return cache


def round_label(meta, round_id: int) -> str:
    """``"cells"`` if *round_id*'s series are the cells round, else ``"hyb{round_id:02d}"``."""
    series = meta.series_for_round(round_id)
    if any((s.imaging_type or "").strip().lower() == "cells" for s in series):
        return "cells"
    return f"hyb{round_id:02d}"


def assign_x_positions(cache: pd.DataFrame) -> pd.DataFrame:
    """
    Add a global, sequential ``x`` column to *cache*: ordered by ascending
    ``round_id`` then ``fov_id`` within each round, so the whole acquisition
    lays out as one continuous axis (cells block, then hyb01, hyb02, ...).
    """
    cache = cache.sort_values(["round_id", "fov_id"]).reset_index(drop=True)
    cache["x"] = range(len(cache))
    return cache
