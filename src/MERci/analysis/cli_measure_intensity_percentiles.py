#!/usr/bin/env python
# MERci/analysis/cli_measure_intensity_percentiles.py
"""
Standalone SLURM-array-task entry point for per-frame intensity-percentile
tables (see ``after_imaging/12_measure_intensity_percentiles.ipynb``).

Not part of the public MERci import surface -- meant to be invoked
directly as a script, one call per array task (one task per image file,
i.e. per round/hyb x FOV):

    python /path/to/SAMPLE_DIR/MERci/src/MERci/analysis/cli_measure_intensity_percentiles.py \\
        --manifest /path/to/intensity_percentiles_manifest.csv

Manifest is a CSV with NO header, 5 columns per line:
``image_path,frame_table_path,round_label,fov_id,output_path`` -- fully
resolved by the notebook (including ad hoc-round frame_table borrowing,
see ``MERci.common.metadata.discover_ad_hoc_round_dirs``), so this script
does no round/FOV discovery of its own.

Self-locates its own sibling ``src/`` root from ``__file__`` so MERci
never needs to be ``pip install``ed on the cluster.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_MERCI_SRC = Path(__file__).resolve().parents[2]   # .../MERci/src/MERci/analysis/cli_measure_intensity_percentiles.py -> .../MERci/src
sys.path.insert(0, str(_MERCI_SRC))

import pandas as pd  # noqa: E402

from MERci.analysis.fov import measure_intensity_percentiles  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path,
                    help="CSV, no header: image_path,frame_table_path,round_label,fov_id,output_path")
    p.add_argument("--array-task-id", type=int, default=None,
                    help="0-based manifest line index; defaults to $SLURM_ARRAY_TASK_ID "
                         "(useful for manual testing outside SLURM).")
    p.add_argument("--percentiles", default="25,50,75,95",
                    help="Comma-separated percentiles to compute (default: 25,50,75,95).")
    return p.parse_args(argv)


def _read_manifest_row(manifest: Path, index: int) -> list:
    with open(manifest, newline="") as fh:
        rows = [row for row in csv.reader(fh) if row]
    if not 0 <= index < len(rows):
        raise IndexError(f"Manifest {manifest} has {len(rows)} line(s); requested index {index}.")
    return rows[index]


def main(argv=None) -> None:
    args = _parse_args(argv)

    task_id = args.array_task_id
    if task_id is None:
        task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id_env is None:
            raise SystemExit(
                "No --array-task-id given and $SLURM_ARRAY_TASK_ID is not set "
                "(this script is meant to run as one task of a SLURM array job)."
            )
        task_id = int(task_id_env)

    image_path, frame_table_path, round_label, fov_id, output_path = _read_manifest_row(args.manifest, task_id)
    percentiles = tuple(int(p) for p in args.percentiles.split(","))

    frame_table = pd.read_csv(frame_table_path, index_col=0)
    df = measure_intensity_percentiles(
        Path(image_path), frame_table, Path(output_path), percentiles=percentiles,
    )
    # round_label/fov_id are manifest-level metadata (what round/FOV this file
    # belongs to) -- measure_intensity_percentiles itself is single-file and
    # doesn't know either, so they're added here before the final save.
    df.insert(0, "fov_id", int(fov_id))
    df.insert(0, "round_label", round_label)
    df.to_parquet(output_path, index=False)
    print(f"Done: {image_path} -> {output_path}  ({len(df)} frames)")


if __name__ == "__main__":
    main()
