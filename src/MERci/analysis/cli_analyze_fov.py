#!/usr/bin/env python
# MERci/analysis/cli_analyze_fov.py
"""
Standalone SLURM-array-task entry point for FOV analysis.

Not part of the public MERci import surface (never ``import``ed by anything
in the package) -- meant to be invoked directly as a script, one call per
array task:

    python /path/to/SAMPLE_DIR/MERci/src/MERci/analysis/cli_analyze_fov.py \\
        --sample-dir /path/to/SAMPLE_DIR \\
        --manifest /path/to/pending_fovs_round003.txt

It locates its own sibling ``src/`` root from ``__file__`` and inserts that
onto ``sys.path`` before importing anything from MERci -- the same
self-locating convention every MERci notebook uses (``sys.path.insert(0,
str(MERCI_DIR / "src"))``), so MERci never needs to be ``pip install``ed on
the cluster (see CLAUDE.md's deployment model).

Reads the manifest line at index ``$SLURM_ARRAY_TASK_ID`` (0-based; each line
is one pending image file path, written by
``07_cluster_submit_analysis.ipynb``) and runs ``analyze_file`` against it,
using the exact same ExperimentConfig-derived paths/kwargs
(``MERci.scheduler.build_fov_task_kwargs``) that ``FOVScheduler`` uses when
running locally, so cluster- and microscope-side analysis are never able to
disagree about where thumbnails/stats/histograms/sentinels land.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_MERCI_SRC = Path(__file__).resolve().parents[2]   # .../MERci/src/MERci/analysis/cli_analyze_fov.py -> .../MERci/src
sys.path.insert(0, str(_MERCI_SRC))

from MERci.common.config import ExperimentConfig    # noqa: E402
from MERci.progress      import ProgressTracker     # noqa: E402
from MERci.scheduler     import build_fov_task_kwargs  # noqa: E402
from MERci.analysis.fov  import analyze_file        # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-dir", required=True, type=Path,
                    help="Experiment root (contains data/, metadata/, analysis/, ...).")
    p.add_argument("--manifest", required=True, type=Path,
                    help="Text file, one pending image path per line.")
    p.add_argument("--array-task-id", type=int, default=None,
                    help="0-based manifest line index; defaults to $SLURM_ARRAY_TASK_ID "
                         "(useful for manual testing outside SLURM).")
    p.add_argument("--image-suffix", default=".zarr",
                    help="Image file suffix (must match how the manifest paths were built).")
    return p.parse_args(argv)


def _read_manifest_line(manifest: Path, index: int) -> Path:
    lines = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
    if not 0 <= index < len(lines):
        raise IndexError(
            f"Manifest {manifest} has {len(lines)} line(s); requested index {index}."
        )
    return Path(lines[index])


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

    fpath = _read_manifest_line(args.manifest, task_id)

    # round_info_csv/positions_txt are required ExperimentConfig fields but
    # unused by build_fov_task_kwargs/analyze_file for a single-FOV task --
    # left as plausible-but-unchecked paths (ExperimentConfig never validates
    # their existence in __post_init__).
    sample_dir = args.sample_dir
    config = ExperimentConfig(
        data_dir       = sample_dir / "data",
        metadata_dir   = sample_dir / "metadata",
        analysis_dir   = sample_dir / "analysis",
        settings_dir   = sample_dir / "settings",
        round_info_csv = sample_dir / "metadata" / "round_info.csv",
        positions_txt  = sample_dir / "positions" / "positions.txt",
        image_suffix   = args.image_suffix,
    )
    tracker = ProgressTracker(config.analysis_dir)

    kwargs = build_fov_task_kwargs(fpath, config, tracker)
    analyze_file(fpath, **kwargs)
    print(f"Done: {fpath}")


if __name__ == "__main__":
    main()
