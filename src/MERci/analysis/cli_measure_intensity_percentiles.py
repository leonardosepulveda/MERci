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
import sys
from pathlib import Path

_MERCI_SRC = Path(__file__).resolve().parents[2]   # .../MERci/src/MERci/analysis/cli_measure_intensity_percentiles.py -> .../MERci/src
sys.path.insert(0, str(_MERCI_SRC))

from MERci.analysis import _cli_common as cli  # noqa: E402
import pandas as pd  # noqa: E402

from MERci.analysis.fov import measure_intensity_percentiles  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path,
                    help="CSV, no header: image_path,frame_table_path,round_label,fov_id,output_path")
    cli.add_task_args(p)
    p.add_argument("--percentiles", default="25,50,75,95",
                    help="Comma-separated percentiles to compute (default: 25,50,75,95).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    task_id = cli.task_id(args)
    image_path, frame_table_path, round_label, fov_id, output_path = cli.manifest_row(args.manifest, task_id, header=False)
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
