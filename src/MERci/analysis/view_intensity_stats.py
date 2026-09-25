# MERci/analysis/view_intensity_stats.py
"""
Logic behind ``notebooks/after_imaging/04_view_intensity_stats.ipynb`` --
loading the per-FOV intensity stats CSVs the FOV scheduler
(``01_fov_scheduler.ipynb``) writes, annotated with round/FOV/stage-position/
z/color, into one DataFrame for plotting (see
:mod:`MERci.plots.view_intensity_stats_plots`).
"""
from __future__ import annotations


import pandas as pd

from ..common.config import ExperimentConfig
from ..common.metadata import ExperimentMetadata
from ..progress import ProgressTracker
from ..acquisition.configs import iter_round_frame_tables


def load_stats_with_annotations(
    config: ExperimentConfig, metadata: ExperimentMetadata, tracker: ProgressTracker,
) -> pd.DataFrame:
    """
    Load all completed stats CSVs and annotate with round_id, fov_id,
    stage position, z, and color.
    """
    frame_info = {}   # round_id -> frame/color/z table (or None)
    records = []
    for round_id, fov_id, sp in tracker.completed_stats_paths(metadata):
        if round_id not in frame_info:
            ft = next((ft for _, ft in iter_round_frame_tables(round_id, config, metadata)), None)
            frame_info[round_id] = None if ft is None else (
                ft[["color", "z"]]
                .reset_index()
                .rename(columns={ft.index.name or "index": "frame"})
            )
        df = pd.read_csv(sp)
        df["round_id"] = round_id
        df["fov_id"] = fov_id
        df["position_x"] = metadata.fovs[fov_id].position[0]
        df["position_y"] = metadata.fovs[fov_id].position[1]
        if frame_info[round_id] is not None:
            df = df.merge(frame_info[round_id], on="frame", how="left")
        records.append(df)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)
