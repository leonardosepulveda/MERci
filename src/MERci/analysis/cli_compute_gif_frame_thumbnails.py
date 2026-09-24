#!/usr/bin/env python
# MERci/analysis/cli_compute_gif_frame_thumbnails.py
"""
Standalone SLURM-array-task entry point for the z-sweep GIF explored in
``notebooks/misc/measure_tissue_thickness_test.ipynb`` (section 24) -- for ONE
FOV, reads every selected z-step's frame (the ``GIF_Z_STRIDE``-subsampled
positions) in a single batched read and writes one downsampled thumbnail
per z-step, using the exact ``z<pos>_fov<id>.npy`` filename convention that
notebook's own calculation cell reads from.

Not part of the public MERci import surface -- meant to be invoked directly
as a script, one call per array task (one task per FOV):

    python /path/to/SAMPLE_DIR/MERci/src/MERci/analysis/cli_compute_gif_frame_thumbnails.py \\
        --manifest /path/to/pending_gif_frames_round003.csv \\
        --output-dir /path/to/SAMPLE_DIR/analysis/cache/measure_tissue_thickness/gif_frames/round003 \\
        --z-positions 0,5,10,15,... \\
        --frame-indices 12,13,14,15,... \\
        --thumbnail-width 200 --thumbnail-height 200 \\
        --flip-horizontal

Manifest is a CSV with columns ``fov_id,image_path`` -- one row per FOV
still missing at least one selected z-step's cached thumbnail (written by
the notebook's own section 24, using the same to-compute/already-cached
split that section's local path already does).

``--frame-indices`` is the FULL, round-wide list of frame indices for every
z-step (0..n_z-1, in z order) -- ``--z-positions`` selects which of those
indices to actually read for each FOV (``frame_indices[z_pos]``).

Self-locates its own sibling ``src/`` root from ``__file__`` (same
convention as ``cli_analyze_fov.py``/``cli_compute_texture_stats.py``) so
MERci never needs to be ``pip install``ed on the cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# .../MERci/src/MERci/analysis/cli_compute_gif_frame_thumbnails.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MERCI_SRC))

from MERci.analysis import _cli_common as cli  # noqa: E402
from MERci.common.io import iter_image_frames                             # noqa: E402
from MERci.acquisition.merlin_config import apply_microscope_orientation  # noqa: E402
from skimage.transform import resize as sk_resize                         # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path,
                    help="CSV with columns fov_id,image_path -- one row per pending FOV.")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write 'z<pos>_fov<id>.npy' into -- must match the "
                         "notebook's own gif_frames_dir for this round.")
    p.add_argument("--z-positions", required=True,
                    help="Comma-separated 0-based z-index positions to render (the "
                         "GIF_Z_STRIDE-subsampled set from section 24).")
    p.add_argument("--frame-indices", required=True,
                    help="Comma-separated 0-based frame indices for the FULL z-grid (0..n_z-1, "
                         "in z order) -- z-positions index into this list.")
    p.add_argument("--thumbnail-width", type=int, required=True)
    p.add_argument("--thumbnail-height", type=int, required=True)
    cli.add_orientation_args(p)
    cli.add_task_args(p, frame_size=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    task_id = cli.task_id(args)
    row = cli.manifest_row(args.manifest, task_id)
    fov_id, fpath = int(row["fov_id"]), Path(row["image_path"])
    z_positions   = [int(x) for x in args.z_positions.split(",")]
    frame_indices = [int(x) for x in args.frame_indices.split(",")]
    tw, th        = args.thumbnail_width, args.thumbnail_height
    orientation = cli.orientation(args)

    read_frame_indices = [frame_indices[z_pos] for z_pos in z_positions]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for z_pos, (_, frame) in zip(z_positions, iter_image_frames(
        fpath, read_frame_indices, frame_width=args.frame_width, frame_height=args.frame_height,
    )):
        frame = apply_microscope_orientation(frame, **orientation)
        thumb = sk_resize(
            frame.astype("float64"), (th, tw), anti_aliasing=True, preserve_range=True
        ).astype("float32")
        out_path = args.output_dir / f"z{z_pos:04d}_fov{fov_id:04d}.npy"
        np.save(out_path, thumb)

    print(f"Done: FOV {fov_id} ({fpath}) -> {len(z_positions)} z-step thumbnail(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
