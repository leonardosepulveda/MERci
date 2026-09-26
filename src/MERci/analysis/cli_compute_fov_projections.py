#!/usr/bin/env python
# MERci/analysis/cli_compute_fov_projections.py
"""
Standalone SLURM-array-task entry point for per-FOV, per-pixel z-stack
projections (median/max/min/mean, via
:func:`MERci.analysis.elevation.project_stack`) -- one real frame read per
z-plane, one task per FOV, every requested statistic computed from the
same in-memory stack (no re-reading raw data per statistic). Built for
a local (unshipped) FFC-method test notebook's investigation into which per-FOV
projection statistic (and whether to Gaussian-smooth the resulting FFC
field at all) best characterizes real vignetting without also absorbing
real tissue signal -- min projection won that comparison (see that
notebook's own Discussion), and is now also what
``after_imaging/08_measure_tissue_thickness.ipynb``'s production pipeline
(:func:`MERci.analysis.elevation.calculate_ffc`) use this script for (with
``--statistics min`` only). ``mean`` is also available for
``calculate_ffc``'s ``method="mean"`` option.

Not part of the public MERci import surface -- meant to be invoked
directly as a script, one call per array task (one task per FOV).
Required flags: ``--manifest`` (a CSV with columns ``fov_id,image_path``),
``--frame-indices`` (comma-separated 0-based frame indices covering the
z-stack), ``--statistics`` (comma-separated subset of
``median,max,min,mean``, default the first three), ``--flip-horizontal``/
``--flip-vertical``/``--transpose`` (this microscope's own orientation
flags), and a results directory via ``--output-dir`` -- writes one
``<fov>_<statistic>.npy`` per requested statistic.

Self-locates its own sibling ``src/`` root from ``__file__`` (same
convention as ``cli_analyze_fov.py``) so
MERci never needs to be ``pip install``ed on the cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# .../MERci/src/MERci/analysis/cli_compute_fov_projections.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MERCI_SRC))

from MERci.analysis import _cli_common as cli  # noqa: E402
from MERci.common.io import read_image_frames                             # noqa: E402
from MERci.acquisition.configs import apply_microscope_orientation  # noqa: E402
from MERci.analysis.elevation import project_stack                        # noqa: E402

_VALID_STATISTICS = ("median", "max", "min", "mean")


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
                    help="Comma-separated subset of median,max,min,mean to compute (default: the first three).")
    cli.add_orientation_args(p)
    cli.add_task_args(p, frame_size=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    task_id = cli.task_id(args)
    row = cli.manifest_row(args.manifest, task_id)
    fov_id, fpath = int(row["fov_id"]), Path(row["image_path"])
    frame_indices = [int(x) for x in args.frame_indices.split(",")]
    statistics = [s.strip() for s in args.statistics.split(",") if s.strip()]
    unknown = set(statistics) - set(_VALID_STATISTICS)
    if unknown:
        raise SystemExit(f"Unknown statistic(s) {sorted(unknown)} -- must be a subset of {_VALID_STATISTICS}.")
    orientation = cli.orientation(args)

    stack = read_image_frames(fpath, frame_indices, frame_width=args.frame_width,
                               frame_height=args.frame_height).astype(np.float32)

    # One read serves every requested statistic. Reorienting is a fixed
    # per-pixel relabelling applied identically to every z-plane, so it
    # commutes with each of these per-pixel-independent reductions --
    # reorient each single 2-D result once rather than every raw frame.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    projections = project_stack(stack, statistics)
    for stat, img in projections.items():
        oriented = apply_microscope_orientation(img, **orientation)
        np.save(args.output_dir / f"fov{fov_id:04d}_{stat}.npy", oriented.astype(np.float32))

    print(f"Done: FOV {fov_id} ({fpath}) -> {statistics} in {args.output_dir} "
          f"({len(frame_indices)} z-plane(s) each)")


if __name__ == "__main__":
    main()
