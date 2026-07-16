#!/usr/bin/env python
# MERci/analysis/cli_compute_texture_stats.py
"""
Standalone SLURM-array-task entry point for the texture-based BG/FG
discriminator explored in ``notebooks/misc/measure_tissue_thickness.ipynb``
(section 14) -- computes one FOV's per-z Gaussian-smoothed-Laplacian-variance
profile and writes it under ``--output-dir`` using the exact same
``<image-stem>_texture.npy`` filename convention that notebook's own
calculation cell reads from, so a cluster-submitted array job and the
notebook can never disagree about where a texture profile lives.

Not part of the public MERci import surface -- meant to be invoked directly
as a script, one call per array task:

    python /path/to/SAMPLE_DIR/MERci/src/MERci/analysis/cli_compute_texture_stats.py \\
        --manifest /path/to/pending_texture_stats_round003.txt \\
        --output-dir /path/to/SAMPLE_DIR/analysis/cache/measure_tissue_thickness/texture_stats/round003 \\
        --frame-indices 12,13,14,15 \\
        --sigma 1.0

It locates its own sibling ``src/`` root from ``__file__`` and inserts that
onto ``sys.path`` before importing anything from MERci -- the same
self-locating convention ``cli_analyze_fov.py`` uses, so MERci never needs to
be ``pip install``ed on the cluster (see CLAUDE.md's deployment model).

Reads the manifest line at index ``$SLURM_ARRAY_TASK_ID`` (0-based; each line
is one FOV image path still missing its cached texture profile -- written by
the notebook's own section 14, using the exact same already-cached?/
to-compute split that cell already does for the local/sequential path).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# .../MERci/src/MERci/analysis/cli_compute_texture_stats.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MERCI_SRC))

from MERci.common.io import iter_image_frames   # noqa: E402
from scipy.ndimage    import gaussian_filter, laplace   # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path,
                    help="Text file, one pending FOV image path per line.")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write '<image-stem>_texture.npy' into -- must "
                         "match the notebook's own texture_stats_dir for this round.")
    p.add_argument("--frame-indices", required=True,
                    help="Comma-separated 0-based frame indices, in z order (the same "
                         "z_frame_indices the notebook's section 3/14 already resolved "
                         "for CHANNEL_NM).")
    p.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian pre-smoothing sigma (pixels) before the Laplacian -- "
                         "must match the notebook's TEXTURE_SMOOTH_SIGMA.")
    p.add_argument("--array-task-id", type=int, default=None,
                    help="0-based manifest line index; defaults to $SLURM_ARRAY_TASK_ID "
                         "(useful for manual testing outside SLURM).")
    p.add_argument("--frame-width", type=int, default=None,
                    help="Only needed for .dax input (raw byte reshape); ignored for "
                         ".zarr/.tiff, which carry their own shape.")
    p.add_argument("--frame-height", type=int, default=None)
    return p.parse_args(argv)


def _read_manifest_line(manifest: Path, index: int) -> Path:
    lines = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
    if not 0 <= index < len(lines):
        raise IndexError(f"Manifest {manifest} has {len(lines)} line(s); requested index {index}.")
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

    fpath         = _read_manifest_line(args.manifest, task_id)
    frame_indices = [int(x) for x in args.frame_indices.split(",")]

    # Same computation as the notebook's own local/sequential path (section 14):
    # Gaussian pre-smooth (suppresses pixel-level sensor noise that would
    # otherwise inflate a purely background frame's raw Laplacian variance),
    # then the Laplacian filter's variance, per z.
    profile = np.full(len(frame_indices), np.nan)
    for pos, (_, frame) in enumerate(iter_image_frames(
        fpath, frame_indices, frame_width=args.frame_width, frame_height=args.frame_height,
    )):
        smoothed     = gaussian_filter(frame.astype(np.float64), sigma=args.sigma)
        profile[pos] = laplace(smoothed).var()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{fpath.stem}_texture.npy"
    np.save(out_path, profile)
    print(f"Done: {fpath} -> {out_path}")


if __name__ == "__main__":
    main()
