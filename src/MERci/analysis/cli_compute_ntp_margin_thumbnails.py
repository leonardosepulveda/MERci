#!/usr/bin/env python
# MERci/analysis/cli_compute_ntp_margin_thumbnails.py
"""
Standalone SLURM-array-task entry point for the NTP-based z_last + margin
sweep explored in ``notebooks/misc/measure_tissue_thickness.ipynb`` (section
23) -- for ONE FOV, reads a single bounded window of frames (covering every
candidate margin at once, same as that section's own local/sequential loop)
and writes one downsampled thumbnail per margin, using the exact
``<fov>_margin<N>.npy`` filename convention that notebook's own calculation
cell reads from.

Not part of the public MERci import surface -- meant to be invoked directly
as a script, one call per array task (one task per FOV):

    python /path/to/SAMPLE_DIR/MERci/src/MERci/analysis/cli_compute_ntp_margin_thumbnails.py \\
        --manifest /path/to/pending_ntp_margin_round003.csv \\
        --output-dir /path/to/SAMPLE_DIR/analysis/cache/measure_tissue_thickness/ntp_margin_sweep/round003 \\
        --frame-indices 12,13,14,... \\
        --z-um-values 6.5,7.0,7.5,... \\
        --margins 1,2,3,4,5,6,7,8,9,10 \\
        --thumbnail-width 200 --thumbnail-height 200 \\
        --flip-horizontal

Manifest is a CSV with columns ``fov_id,image_path,z_last_um`` -- one row
per FOV still missing at least one cached margin thumbnail (written by the
notebook's own section 23, using the same to-compute/already-cached split
that section's local path already does).

Self-locates its own sibling ``src/`` root from ``__file__`` (same
convention as ``cli_analyze_fov.py``/``cli_compute_texture_stats.py``) so
MERci never needs to be ``pip install``ed on the cluster.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# .../MERci/src/MERci/analysis/cli_compute_ntp_margin_thumbnails.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MERCI_SRC))

from MERci.common.io import iter_image_frames                             # noqa: E402
from MERci.acquisition.merlin_config import apply_microscope_orientation  # noqa: E402
from skimage.transform import resize as sk_resize                         # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path,
                    help="CSV with columns fov_id,image_path,z_last_um -- one row per pending FOV.")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write '<fov>_margin<N>.npy' into -- must match the "
                         "notebook's own margin_thumbnails_dir for this round.")
    p.add_argument("--frame-indices", required=True,
                    help="Comma-separated 0-based frame indices, in z order (this round's own "
                         "z_frame_indices for CHANNEL_NM, shared across every FOV).")
    p.add_argument("--z-um-values", required=True,
                    help="Comma-separated z (um) values, same order/length as --frame-indices.")
    p.add_argument("--margins", default="1,2,3,4,5,6,7,8,9,10",
                    help="Comma-separated integer margin values (um) to render per FOV.")
    p.add_argument("--thumbnail-width", type=int, required=True)
    p.add_argument("--thumbnail-height", type=int, required=True)
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
    return int(row["fov_id"]), Path(row["image_path"]), float(row["z_last_um"])


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

    fov_id, fpath, z_last_um_ntp = _read_manifest_row(args.manifest, task_id)
    frame_indices = [int(x) for x in args.frame_indices.split(",")]
    z_um_values   = np.array([float(x) for x in args.z_um_values.split(",")])
    margins       = [int(x) for x in args.margins.split(",")]
    tw, th        = args.thumbnail_width, args.thumbnail_height
    full_depth_um = float(z_um_values[-1])
    orientation = {
        "flip_horizontal": args.flip_horizontal,
        "flip_vertical":   args.flip_vertical,
        "transpose":       args.transpose,
    }

    # Same bounded-window idea as the notebook's own local/sequential path:
    # one read spanning every margin candidate at once, not one re-read per margin.
    pos_start   = int(np.argmin(np.abs(z_um_values - z_last_um_ntp)))
    max_margin_target = min(z_last_um_ntp + max(margins), full_depth_um)
    pos_end     = max(pos_start, int(np.argmin(np.abs(z_um_values - max_margin_target))))
    window_positions = list(range(pos_start, pos_end + 1))
    window_z_um      = z_um_values[pos_start:pos_end + 1]
    window_frame_idx = [frame_indices[p] for p in window_positions]

    frames_by_pos = {}
    for pos, (_, frame) in zip(window_positions, iter_image_frames(
        fpath, window_frame_idx, frame_width=args.frame_width, frame_height=args.frame_height,
    )):
        frame = apply_microscope_orientation(frame, **orientation)
        frames_by_pos[pos] = sk_resize(
            frame.astype(np.float64), (th, tw), anti_aliasing=True, preserve_range=True
        ).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for margin_um in margins:
        z_target  = min(z_last_um_ntp + margin_um, full_depth_um)
        local_idx = int(np.argmin(np.abs(window_z_um - z_target)))
        pos       = pos_start + local_idx
        out_path  = args.output_dir / f"fov{fov_id:04d}_margin{margin_um}.npy"
        np.save(out_path, frames_by_pos[pos])

    print(f"Done: FOV {fov_id} ({fpath}) -> {len(margins)} margin thumbnail(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
