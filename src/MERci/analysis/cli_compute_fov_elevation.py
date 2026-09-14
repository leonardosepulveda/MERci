#!/usr/bin/env python
# MERci/analysis/cli_compute_fov_elevation.py
"""
Standalone SLURM-array-task entry point for per-FOV tissue-elevation
matrices + FFC-corrected, downsampled z-stacks
(:func:`MERci.analysis.elevation.compute_fov_elevation`) -- one real full
z-stack read per FOV, one task per FOV. Built for
``after_imaging/08_measure_tissue_thickness.ipynb``'s production tissue-
elevation pipeline (promoted from
``notebooks/tests/tissue_thickness/01_elevation_heatmap.ipynb``'s own
investigation), where reading every real FOV's full z-stack serially would
take hours (see that notebook's own section 6 for the same concern on the
FFC-field step).

Not part of the public MERci import surface -- meant to be invoked
directly as a script, one call per array task (one task per FOV).
Required flags: ``--manifest`` (a CSV with columns ``fov_id,image_path``),
``--ffc-field-path`` (an ``.npz`` saved by
:func:`MERci.analysis.ffc.save_ffc_field`), ``--threshold``,
``--downsample-factor``, ``--frame-indices``/``--z-um-values`` (this
round's own ``CHANNEL_NM`` z-grid, same order), ``--flip-horizontal``/
``--flip-vertical``/``--transpose``, and an output directory
(``--output-dir``) -- writes ``<fov>_elevation.npy`` (the elevation matrix
``M``) and ``<fov>_stack.npy`` (the FFC-corrected, downsampled z-stack,
saved as a plain, uncompressed ``.npy`` so
:func:`MERci.analysis.elevation.create_gif` can memory-map individual
z-planes out of it at full-grid scale instead of loading every FOV's whole
stack into memory at once).

Self-locates its own sibling ``src/`` root from ``__file__`` (same
convention as ``cli_compute_fov_projections.py``) so MERci never needs to
be ``pip install``ed on the cluster.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# .../MERci/src/MERci/analysis/cli_compute_fov_elevation.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MERCI_SRC))

from MERci.analysis.ffc import load_ffc_field            # noqa: E402
from MERci.analysis.elevation import compute_fov_elevation  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path,
                    help="CSV with columns fov_id,image_path -- one row per pending FOV.")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write '<fov>_elevation.npy'/'<fov>_stack.npy' into.")
    p.add_argument("--ffc-field-path", required=True, type=Path,
                    help="FFC field .npz (MERci.analysis.ffc.save_ffc_field).")
    p.add_argument("--threshold", required=True, type=float,
                    help="Background/foreground intensity cutoff, in the FFC-corrected + "
                         "downsampled space.")
    p.add_argument("--downsample-factor", required=True, type=int)
    p.add_argument("--frame-indices", required=True,
                    help="Comma-separated 0-based frame indices, ascending z order.")
    p.add_argument("--z-um-values", required=True,
                    help="Comma-separated z (um) for each of --frame-indices, same order.")
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
    z_um_values   = [float(x) for x in args.z_um_values.split(",")]
    orientation = {
        "flip_horizontal": args.flip_horizontal,
        "flip_vertical":   args.flip_vertical,
        "transpose":       args.transpose,
    }
    ffc_field, _ = load_ffc_field(args.ffc_field_path)

    M, ds_stack = compute_fov_elevation(
        fpath, frame_indices, z_um_values, ffc_field, args.threshold, args.downsample_factor,
        orientation=orientation, frame_width=args.frame_width, frame_height=args.frame_height,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / f"fov{fov_id:04d}_elevation.npy", M.astype(np.float32))
    np.save(args.output_dir / f"fov{fov_id:04d}_stack.npy", ds_stack.astype(np.float32))

    print(f"Done: FOV {fov_id} ({fpath}) -> elevation + {len(frame_indices)}-plane stack "
          f"in {args.output_dir}")


if __name__ == "__main__":
    main()
