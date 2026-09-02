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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CACHE_COLUMNS = [
    "round_id", "fov_id", "series", "off_path",
    "first_stage_z", "min_stage_z", "max_stage_z", "all_same",
]

# Sentinel (round_id, series) tag for focus-lock-test rows in the same cache --
# the focus-lock test (MERci.acquisition.dave.create_focus_test_dave_config) is
# a standalone calibration procedure, not a real imaging round in
# round_info.csv, so it has no real round_id of its own to reuse. -1 is never
# a real imaging_round value (those start at 1).
FOCUSTEST_ROUND_ID    = -1
FOCUSTEST_SERIES_NAME = "focustest"


def off_path_for(image_path: Path) -> Path:
    """Return the ``.off`` sidecar path for *image_path* (same dir, same stem)."""
    return Path(image_path).with_suffix(".off")


def read_off_file(off_path: Path) -> pd.DataFrame:
    """
    Parse a HAL ``.off`` focus-lock log: whitespace-delimited, one row per
    frame, columns ``frame offset power stage-z good-offset``.

    May raise (``pandas.errors.EmptyDataError`` etc.) if *off_path* exists
    but hasn't been fully written yet -- see :func:`read_off_file_if_ready`
    for a version tolerant of that race.
    """
    return pd.read_csv(off_path, sep=r"\s+", engine="python")


_OFF_COLUMNS = {"frame", "offset", "power", "stage-z", "good-offset"}


def read_off_file_if_ready(off_path: Path) -> Optional[pd.DataFrame]:
    """
    Like :func:`read_off_file`, but returns ``None`` instead of raising when
    *off_path* isn't safely readable yet.

    HAL creates the ``.off`` file before it has written any rows (or a
    complete last row) -- so a reader running concurrently with acquisition
    (e.g. this module's cache update, run *during* the experiment) can
    observe the file mid-write: zero bytes, a header with no data rows yet,
    or a truncated last line. All of these are treated identically to "not
    written yet" rather than a hard failure, so a caller iterating many FOVs
    doesn't have one still-being-written file crash the whole run -- it
    picks the file up again next time it's called.
    """
    off_path = Path(off_path)
    if not off_path.exists():
        return None
    try:
        off_df = read_off_file(off_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        return None
    if off_df.empty or not _OFF_COLUMNS.issubset(off_df.columns):
        return None
    return off_df


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
    column, or ``None`` if the sidecar doesn't exist yet, or exists but
    isn't fully written yet (not written, still being written by HAL, or
    this series/FOV combination doesn't apply -- see
    :func:`read_off_file_if_ready`).
    """
    off_df = read_off_file_if_ready(off_path_for(image_path))
    if off_df is None:
        return None
    return summarize_stage_z(off_df)


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
    taken here -- e.g. a check-focus-only test FOV, or one not yet run) or
    isn't fully written yet (see :func:`read_off_file_if_ready`).
    """
    off_df = read_off_file_if_ready(off_path_for(image_path))
    if off_df is None:
        return None
    return summarize_focus_lock(off_df)


_BOOL_LIKE_MAP = {
    True: True, False: False,
    "True": True, "False": False,
    "true": True, "false": False,
    1: True, 0: False, "1": True, "0": False,
}


def _coerce_bool_column(series: pd.Series) -> pd.Series:
    """
    Coerce a boolean-like column (real ``bool``, ``"True"``/``"False"``
    text, or ``1``/``0``) back to proper ``bool`` dtype.

    A CSV round-trip through ``to_csv``/``read_csv`` doesn't reliably
    preserve ``bool`` dtype -- concatenating a freshly-computed real-``bool``
    column with one reloaded from CSV (which can come back as ``object`` or
    ``int64``) silently produces a non-bool column. ``~`` on a non-bool
    column does bitwise (not logical) negation -- e.g. ``~1 == -2`` -- so an
    unfixed column doesn't just give wrong results, it makes
    ``cache[~cache["all_same"]]`` raise ``KeyError`` (pandas reads the
    resulting ``-2``s as column labels to select, not a boolean mask).
    """
    return series.map(_BOOL_LIKE_MAP).astype(bool)


def load_stage_z_cache(cache_path: Path) -> pd.DataFrame:
    """Load the on-disk stage-z summary cache, or an empty frame if none exists yet."""
    if Path(cache_path).exists():
        cache = pd.read_csv(cache_path)
        if "all_same" in cache.columns:
            cache["all_same"] = _coerce_bool_column(cache["all_same"])
        return cache
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
        cache["all_same"] = _coerce_bool_column(cache["all_same"])
        cache.to_csv(cache_path, index=False)
    return cache


def discover_focustest_off_files(focus_test_dir: Path, movie_name: Optional[str] = None) -> dict:
    """
    ``{fov_id: off_path}`` for every focus-lock-test ``.off`` sidecar under
    *focus_test_dir* (:func:`MERci.acquisition.dave.create_focus_test_dave_config`'s
    own output directory, e.g. ``SAMPLE_DIR/data/focus_test``). ``fov_id`` is
    the file's own zero-padded index (``{movie_name}_{fov_id:0N}.off``) --
    Dave visits one position per movie in the SAME 0-based row order as the
    positions file the recipe was built from, so this is the same ``fov_id``
    space as everywhere else in this package, not a separate numbering.

    Parameters
    ----------
    movie_name : the recipe's own ``movie_name`` (e.g. ``"hal-st2-focustest"``,
        :func:`MERci.acquisition.dave.create_focus_test_dave_config`'s own
        parameter) -- auto-detected from the first ``*.off`` file present if
        not given.

    Returns
    -------
    ``{}`` if *focus_test_dir* doesn't exist or has no ``.off`` files yet.
    """
    focus_test_dir = Path(focus_test_dir)
    off_paths = sorted(focus_test_dir.glob("*.off")) if focus_test_dir.is_dir() else []
    if not off_paths:
        return {}
    if movie_name is None:
        movie_name = off_paths[0].stem.rsplit("_", 1)[0]
    result = {}
    for p in focus_test_dir.glob(f"{movie_name}_*.off"):
        idx_str = p.stem[len(movie_name) + 1:]
        if idx_str.isdigit():
            result[int(idx_str)] = p
    return result


def update_focustest_stage_z_cache(
    focus_test_dir: Path, cache_path: Path, movie_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extend the stage-z cache (same on-disk schema/file as
    :func:`update_stage_z_cache`) with the focus-lock test's own per-FOV
    ``.off`` files under *focus_test_dir*.

    The focus-lock test is a standalone calibration procedure
    (:func:`MERci.acquisition.dave.create_focus_test_dave_config`), not a
    real imaging round in ``round_info.csv`` -- :func:`update_stage_z_cache`'s
    ``ExperimentMetadata``-driven round/series loop has no way to discover
    it, so it needs this separate entry point. Rows are tagged
    ``round_id=FOCUSTEST_ROUND_ID``, ``series=FOCUSTEST_SERIES_NAME`` so they
    plot alongside real rounds via the same
    :func:`assign_x_positions`/plotting code once loaded back with
    :func:`load_stage_z_cache` (see :func:`round_label`'s focustest
    special-case for the legend label).
    """
    cache = load_stage_z_cache(cache_path)
    seen = set(zip(cache["round_id"], cache["fov_id"], cache["series"]))
    new_rows = []

    for fov_id, off_path in discover_focustest_off_files(focus_test_dir, movie_name).items():
        key = (FOCUSTEST_ROUND_ID, fov_id, FOCUSTEST_SERIES_NAME)
        if key in seen:
            continue
        off_df = read_off_file_if_ready(off_path)
        if off_df is None:
            continue   # not written yet, or still being written -- pick it up on a future run
        new_rows.append({
            "round_id": FOCUSTEST_ROUND_ID, "fov_id": fov_id, "series": FOCUSTEST_SERIES_NAME,
            "off_path": str(off_path), **summarize_stage_z(off_df),
        })

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache["all_same"] = _coerce_bool_column(cache["all_same"])
        cache.to_csv(cache_path, index=False)
    return cache


def round_label(meta, round_id: int) -> str:
    """``"cells"`` if *round_id*'s series are the cells round, else
    ``"hyb{N:02d}"`` where ``N`` is this round's 1-based rank among all
    non-cells rounds -- NOT the raw ``round_id``. ``imaging_round`` numbering
    reserves round 1 for cells (rounds 2..N_HYBS+1 are the hyb rounds), so
    ``round_id=2`` is the FIRST real hyb round and must map to ``"hyb01"``,
    matching the real ``H01``/``H02``/... data folders -- using ``round_id``
    directly mislabels every hyb round one too high (e.g. ``"hyb02"`` for the
    round that actually wrote to ``H01``).

    ``round_id == FOCUSTEST_ROUND_ID`` is not a real round at all (see
    :func:`update_focustest_stage_z_cache`) -- returned as its own label
    before touching *meta*, which has no entry for it."""
    if round_id == FOCUSTEST_ROUND_ID:
        return "focustest"

    def _is_cells(rid: int) -> bool:
        return any((s.imaging_type or "").strip().lower() == "cells"
                   for s in meta.series_for_round(rid))

    if _is_cells(round_id):
        return "cells"
    hyb_round_ids = [rid for rid in meta.valid_round_ids() if not _is_cells(rid)]
    return f"hyb{hyb_round_ids.index(round_id) + 1:02d}"


def assign_x_positions(cache: pd.DataFrame) -> pd.DataFrame:
    """
    Add an ``x`` column to *cache* equal to ``fov_id``, so every round's
    points/line share the SAME x range and overlay directly on top of each
    other -- round-to-round drift at a given FOV shows up as a vertical
    offset between lines, not a left-to-right shift along one long
    concatenated axis (the previous behaviour, which laid every round's
    FOVs out back-to-back).
    """
    cache = cache.copy()
    cache["x"] = cache["fov_id"]
    return cache


def positions_to_grid_indices(fov_ids: List[int], meta) -> Dict[int, Tuple[int, int]]:
    """
    Stage ``(x, y)`` positions -> integer ``(x_idx, y_idx)`` grid indices --
    round to the nearest integer micron, then rank each axis's unique
    values, which is robust to float imprecision on a regular grid. Same
    approach as ``after_imaging/04_view_intensity_stats.ipynb``'s /
    ``misc/measure_tissue_thickness_test.ipynb``'s own heatmaps use.
    """
    xs = np.array([round(meta.fovs[f].position[0]) for f in fov_ids])
    ys = np.array([round(meta.fovs[f].position[1]) for f in fov_ids])
    unique_xs = np.sort(np.unique(xs))
    unique_ys = np.sort(np.unique(ys))
    x_rank = {v: i for i, v in enumerate(unique_xs)}
    y_rank = {v: i for i, v in enumerate(unique_ys)}
    return {f: (x_rank[xs[i]], y_rank[ys[i]]) for i, f in enumerate(fov_ids)}
