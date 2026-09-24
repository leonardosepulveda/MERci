#!/usr/bin/env python
# MERci/analysis/cli_compute_channel_counters.py
"""
Standalone SLURM-array-task entry point for computing per-FOV channel
Counters -- the heaviest read step in ``measure_tissue_thickness``-style
notebooks (one full z-sweep of a single channel, per FOV). For ONE FOV,
reads every z-plane of the target channel and writes the resulting
``compute_channel_counters()`` result to the exact ``<image-stem>_counters.npz``
cache path that notebook's own calculation cell reads from.

Not part of the public MERci import surface -- meant to be invoked directly
as a script, one call per array task (one task per FOV):

    python /path/to/SAMPLE_DIR/MERci/src/MERci/analysis/cli_compute_channel_counters.py \\
        --manifest /path/to/pending_channel_counters_round003.txt \\
        --output-dir /path/to/SAMPLE_DIR/analysis/cache/measure_tissue_thickness/channel_counters \\
        --frame-indices 12,13,14,15,... \\
        --z-um-values 6.5,7.0,7.5,...

Manifest is a text file, one pending FOV image path per line (same
convention as ``cli_analyze_fov.py``/``cli_compute_texture_stats.py``).

Self-locates its own sibling ``src/`` root from ``__file__`` so MERci never
needs to be ``pip install``ed on the cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# .../MERci/src/MERci/analysis/cli_compute_channel_counters.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MERCI_SRC))

from MERci.analysis import _cli_common as cli  # noqa: E402
from MERci.analysis.fov import compute_channel_counters, save_channel_counters  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path,
                    help="Text file, one pending FOV image path per line.")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write '<image-stem>_counters.npz' into -- must match "
                         "the notebook's own channel_counters_dir.")
    p.add_argument("--frame-indices", required=True,
                    help="Comma-separated 0-based frame indices, in z order (this round's "
                         "z_frame_indices for CHANNEL_NM).")
    p.add_argument("--z-um-values", required=True,
                    help="Comma-separated z (um) values, same order/length as --frame-indices.")
    cli.add_task_args(p, frame_size=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    task_id = cli.task_id(args)
    fpath           = Path(cli.manifest_line(args.manifest, task_id))
    frame_indices   = [int(x) for x in args.frame_indices.split(",")]
    z_um_values     = [float(x) for x in args.z_um_values.split(",")]
    z_frame_indices = list(zip(frame_indices, z_um_values))

    counters = compute_channel_counters(
        fpath, z_frame_indices, frame_width=args.frame_width, frame_height=args.frame_height,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{fpath.stem}_counters.npz"
    save_channel_counters(out_path, counters)
    print(f"Done: {fpath} -> {out_path}")


if __name__ == "__main__":
    main()
