#!/usr/bin/env python
# MERci/analysis/cli_build_round_mosaic.py
"""
Standalone SLURM-task entry point for building one round's mosaic(s).

Not part of the public MERci import surface -- invoked directly as a script,
one call per round (optionally as one task of a small SLURM array, see
``07_cluster_submit_analysis.ipynb``):

    python /path/to/SAMPLE_DIR/MERci/src/MERci/analysis/cli_build_round_mosaic.py \\
        --sample-dir /path/to/SAMPLE_DIR \\
        --round-id 3

Locates its own sibling ``src/`` root from ``__file__`` and inserts that onto
``sys.path`` before importing anything from MERci -- see
``cli_analyze_fov.py``'s docstring for why (no ``pip install`` needed on the
cluster).

Reconstructs ``ExperimentConfig``/``ExperimentMetadata``/``ProgressTracker``
from ``--sample-dir`` and calls ``MERci.scheduler.build_round_mosaics``, the
exact same function ``RoundScheduler`` calls when running locally.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_MERCI_SRC = Path(__file__).resolve().parents[2]   # .../MERci/src/MERci/analysis/cli_build_round_mosaic.py -> .../MERci/src
sys.path.insert(0, str(_MERCI_SRC))

from MERci.common.config   import ExperimentConfig    # noqa: E402
from MERci.common.metadata import ExperimentMetadata  # noqa: E402
from MERci.progress        import ProgressTracker     # noqa: E402
from MERci.scheduler       import build_round_mosaics  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-dir", required=True, type=Path,
                    help="Experiment root (contains data/, metadata/, positions/, analysis/, ...).")
    p.add_argument("--round-id", type=int, default=None,
                    help="Round id to build mosaics for. If omitted, read from "
                         "--manifest at index $SLURM_ARRAY_TASK_ID.")
    p.add_argument("--manifest", type=Path, default=None,
                    help="Text file, one round id per line -- for array-job use "
                         "when several rounds need mosaics in one submission.")
    p.add_argument("--array-task-id", type=int, default=None,
                    help="0-based manifest line index; defaults to $SLURM_ARRAY_TASK_ID.")
    p.add_argument("--image-suffix", default=".zarr")
    p.add_argument("--positions-file", type=Path, default=None,
                    help="Positions file to load (defaults to the first "
                         "positions_*.txt found under --sample-dir/positions/).")
    return p.parse_args(argv)


def _resolve_round_id(args: argparse.Namespace) -> int:
    if args.round_id is not None:
        return args.round_id
    if args.manifest is None:
        raise SystemExit("Give either --round-id or --manifest.")
    task_id = args.array_task_id
    if task_id is None:
        task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id_env is None:
            raise SystemExit(
                "No --array-task-id given and $SLURM_ARRAY_TASK_ID is not set."
            )
        task_id = int(task_id_env)
    lines = [ln.strip() for ln in args.manifest.read_text().splitlines() if ln.strip()]
    if not 0 <= task_id < len(lines):
        raise IndexError(f"Manifest {args.manifest} has {len(lines)} line(s); requested {task_id}.")
    return int(lines[task_id])


def _resolve_positions_file(sample_dir: Path, override: Path = None) -> Path:
    if override is not None:
        return override
    candidates = sorted((sample_dir / "positions").glob("positions_*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No positions_*.txt found under {sample_dir / 'positions'}.")
    return candidates[0]


def main(argv=None) -> None:
    args = _parse_args(argv)
    round_id = _resolve_round_id(args)
    sample_dir = args.sample_dir

    config = ExperimentConfig(
        data_dir       = sample_dir / "data",
        metadata_dir   = sample_dir / "metadata",
        analysis_dir   = sample_dir / "analysis",
        settings_dir   = sample_dir / "settings",
        round_info_csv = sample_dir / "metadata" / "round_info.csv",
        positions_txt  = _resolve_positions_file(sample_dir, args.positions_file),
        image_suffix   = args.image_suffix,
    )
    meta = ExperimentMetadata.load(
        config.round_info_csv, config.positions_txt, config.data_dir,
        image_suffix=config.image_suffix,
    )
    tracker = ProgressTracker(config.analysis_dir)

    built = build_round_mosaics(round_id, config, meta, tracker)
    print(f"Round {round_id}: mosaic(s) built={built}")


if __name__ == "__main__":
    main()
