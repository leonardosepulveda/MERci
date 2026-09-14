#!/usr/bin/env python
# MERci/analysis/cli_compute_fov_projections.py
"""
Standalone SLURM-array-task entry point for per-FOV, per-pixel z-stack
projections (median/max/min) -- one real frame read per z-plane, one task
per FOV, every requested statistic computed from the same in-memory stack
(no re-reading raw data per statistic). Built for
``notebooks/tests/calculate_ffc/``'s investigation into which per-FOV
projection statistic (and whether to Gaussian-smooth the resulting FFC
field at all) best characterizes real vignetting without also absorbing
real tissue signal -- min projection won that comparison (see that
notebook's own Discussion), and is now also what
``notebooks/tests/tissue_thickness/01_elevation_heatmap.ipynb`` uses this
script for (with ``--statistics min`` only).

Not part of the public MERci import surface -- meant to be invoked
directly as a script, one call per array task (one task per FOV).
Required flags: ``--manifest`` (a CSV with columns ``fov_id,image_path``),
``--frame-indices`` (comma-separated 0-based frame indices covering the
z-stack), ``--statistics`` (comma-separated subset of ``median,max,min``,
default all three), ``--flip-horizontal``/``--flip-vertical``/
``--transpose`` (this microscope's own orientation flags), and a results
directory via ``--output-dir`` -- writes one ``<fov>_<statistic>.npy`` per
requested statistic.

Self-locates its own sibling ``src/`` root from ``__file__`` (same
convention as ``cli_analyze_fov.py``/``cli_compute_fov_median.py``) so
MERci never needs to be ``pip install``ed on the cluster.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# .../MERci/src/MERci/analysis/cli_compute_fov_projections.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MERCI_SRC))

from MERci.common.io import read_image_frames                             # noqa: E402
from MERci.acquisition.merlin_config import apply_microscope_orientation  # noqa: E402

_VALID_STATISTICS = ("median", "max", "min")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path,
                    help="CSV with columns fov_id,image_path -- one row per pending FOV.")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write '<fov>_<statistic>.npy' into.")
    p.add_argument("--frame-indices", required=True,
                    help="Comma-separated 0-based frame indices covering the z-stack "
                         "(this round's own z_frame_indices for CHANNEL_NM, shared across every FOV).")
    p.add_argument("--statistics", default="median,max,min",
                    help="Comma-separated subset of median,max,min to compute (default: all three).")
    p.add_argument("--flip-horizontal", action="store_true")
    p.add_argument("--flip-vertical", action="store_true")
    p.add_argument("--transpose", action="store_true")
    p.add_argument("--array-task-id", type=int, default=None,
                    help="0-based manifest row index; defaults to $SLURM_ARRAY_TASK_ID "
                         "(useful for manual testing outside SLURM).")
    p.add_argument("--frame-width", type=int, default=None,
                    help="Only needed for .dax input; ignored for .zarr/.tiff.")
    p.add_argument("--frame-height", type=int, default=None)
    return p.parse_args(argv)


def _read_manifest_row(manifest: Path, index: int):
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not 0 <= index < len(rows):
        raise IndexError(f"Manifest {manifest} has {len(rows)} row(s); requested index {index}.")
    row = rows[index]
    return int(row["fov_id"]), Path(row["image_path"])


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

    fov_id, fpath = _read_manifest_row(args.manifest, task_id)
    frame_indices = [int(x) for x in args.frame_indices.split(",")]
    statistics = [s.strip() for s in args.statistics.split(",") if s.strip()]
    unknown = set(statistics) - set(_VALID_STATISTICS)
    if unknown:
        raise SystemExit(f"Unknown statistic(s) {sorted(unknown)} -- must be a subset of {_VALID_STATISTICS}.")
    orientation = {
        "flip_horizontal": args.flip_horizontal,
        "flip_vertical":   args.flip_vertical,
        "transpose":       args.transpose,
    }

    stack = read_image_frames(fpath, frame_indices, frame_width=args.frame_width,
                               frame_height=args.frame_height).astype(np.float32)

    # Order matters: max/min don't touch the array's element order, so
    # compute them first; median (with overwrite_input=True, cheaper than
    # letting it make its own internal copy) goes last since nothing else
    # needs `stack` intact afterward. Reorienting is a fixed per-pixel
    # relabelling applied identically to every z-plane, so it commutes with
    # each of these per-pixel-independent reductions -- reorient each
    # single 2-D result once rather than every raw frame.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if "max" in statistics:
        img = apply_microscope_orientation(np.max(stack, axis=0), **orientation)
        np.save(args.output_dir / f"fov{fov_id:04d}_max.npy", img.astype(np.float32))
    if "min" in statistics:
        img = apply_microscope_orientation(np.min(stack, axis=0), **orientation)
        np.save(args.output_dir / f"fov{fov_id:04d}_min.npy", img.astype(np.float32))
    if "median" in statistics:
        img = apply_microscope_orientation(np.median(stack, axis=0, overwrite_input=True), **orientation)
        np.save(args.output_dir / f"fov{fov_id:04d}_median.npy", img.astype(np.float32))

    print(f"Done: FOV {fov_id} ({fpath}) -> {statistics} in {args.output_dir} "
          f"({len(frame_indices)} z-plane(s) each)")


if __name__ == "__main__":
    main()
