# MERci/analysis/batch_sample_review.py
"""
Logic behind ``notebooks/after_imaging/05_batch_sample_review.ipynb`` --
reviewing a batch of finished (or in-progress) experiments together: backfill
any acquisition-time analysis a sample's FOV scheduler missed, then load
every sample's stats into one combined DataFrame for cross-sample plotting
(see :mod:`MERci.plots.batch_sample_review_plots`).

Each *sample* is a plain dict with keys ``info``/``sample_dir``/``config``/
``meta``/``tracker`` (built once per experiment by the notebook itself, since
building it needs each experiment's own ``experiment_info.yaml``).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from ..progress_display import ProgressReporter
from .fov import analyze_file, load_stats


def backfill_pending(sample: dict) -> int:
    """
    Analyse (in parallel, across ``config.resolved_n_workers`` workers) any
    of *sample*'s expected files that its acquisition-time FOV scheduler
    hasn't already processed. Returns the number of files backfilled (0 if
    everything was already done).
    """
    config, meta, tracker = sample["config"], sample["meta"], sample["tracker"]
    all_files = meta.all_expected_files()
    pending = tracker.pending_fov_files(all_files)
    if not pending:
        return 0

    kwargs_common = dict(
        thumbnails_dir=config.analysis_dir / "thumbnails",
        frame_width=config.frame_width,
        frame_height=config.frame_height,
        thumbnail_frames=config.thumbnail_frames,
        thumbnail_size=config.thumbnail_size,
        thumbnail_percentile_clip=config.thumbnail_percentile_clip,
        histogram_bins=config.histogram_bins,
        histogram_range=config.histogram_range,
    )

    reporter = ProgressReporter(total=len(pending), label=f"{sample['info'].sample_name}: backfilling")
    with ProcessPoolExecutor(max_workers=config.resolved_n_workers) as pool:
        futures = {
            pool.submit(
                analyze_file, fpath,
                stats_path=tracker.stats_path(fpath),
                histogram_path=tracker.histogram_path(fpath),
                sentinel_path=tracker.fov_sentinel(fpath),
                **kwargs_common,
            ): fpath
            for fpath in pending
        }
        for future in as_completed(futures):
            future.result()   # re-raise here if a worker errored on this FOV
            reporter.update()
    reporter.done()
    return len(pending)


def load_all_stats(sample: dict) -> pd.DataFrame:
    """Every completed stats CSV for *sample*, concatenated with a
    ``sample_name`` column added (for combining across samples)."""
    meta, tracker = sample["meta"], sample["tracker"]
    records = []
    for round_id in meta.valid_round_ids():
        round_obj = meta.rounds.get(round_id)
        if round_obj is None:
            continue
        for fov_id, file_list in round_obj.fov_files.items():
            for fpath in file_list:
                sp = tracker.stats_path(fpath)
                if not sp.exists():
                    continue
                df = load_stats(sp)
                df["round_id"] = round_id
                df["fov_id"] = fov_id
                records.append(df)
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    out["sample_name"] = sample["info"].sample_name
    return out
