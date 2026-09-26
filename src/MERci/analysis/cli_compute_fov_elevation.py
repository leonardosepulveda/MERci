#!/usr/bin/env python
# MERci/analysis/cli_compute_fov_elevation.py
"""
Standalone SLURM-array-task entry point for per-FOV tissue-elevation
matrices + FFC-corrected, downsampled z-stacks
(:func:`MERci.analysis.elevation.compute_fov_elevation`) -- one real full
z-stack read per FOV, one task per FOV. Built for
``after_imaging/08_measure_tissue_thickness.ipynb``'s production tissue-
elevation pipeline (promoted from a local, unshipped elevation-heatmap
test notebook's investigation), where reading every real FOV's full
z-stack serially would
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
import sys
from pathlib import Path

import numpy as np

# .../MERci/src/MERci/analysis/cli_compute_fov_elevation.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MERCI_SRC))

from MERci.analysis import _cli_common as cli  # noqa: E402
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
    cli.add_orientation_args(p)
    cli.add_task_args(p, frame_size=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    task_id = cli.task_id(args)
    row = cli.manifest_row(args.manifest, task_id)
    fov_id, fpath = int(row["fov_id"]), Path(row["image_path"])
    frame_indices = [int(x) for x in args.frame_indices.split(",")]
    z_um_values   = [float(x) for x in args.z_um_values.split(",")]
    orientation = cli.orientation(args)
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
