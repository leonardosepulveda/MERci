#!/usr/bin/env python
# MERci/analysis/cli_check_fov_completeness.py
"""
Standalone SLURM-array-task entry point for
:func:`MERci.analysis.completeness.check_one_file` -- one task per FOV,
checking every expected file for that FOV across a given set of rounds.

A serial scan (``MERci.analysis.completeness.check_dataset_completeness``)
takes roughly 1 second per ``.zarr`` file on this pipeline's cluster
storage (dominated by per-chunk filesystem-metadata latency, not CPU), so a
full experiment (~1000+ FOVs x a dozen-plus rounds) would take hours run
serially -- the same "reading every real FOV serially would take hours"
concern that motivated ``cli_compute_fov_elevation.py``'s own SLURM-array
design. One task per FOV keeps each task's own runtime small (all of that
FOV's rounds, not the whole dataset) while spreading the per-chunk
filesystem latency across the array's concurrency.

Not part of the public MERci import surface -- meant to be invoked
directly as a script, one call per array task (one task per FOV).
Required flags: ``--round-info-csv``, ``--positions-txt``, ``--data-dir``
(to rebuild the same ``ExperimentMetadata`` the submitting notebook used),
``--round-ids`` (comma-separated round ids to check -- the submitting
notebook decides which, e.g. only fully-written ones), ``--manifest`` (a
CSV with one ``fov_id`` column, one row per pending FOV), and
``--output-dir`` -- writes ``fov<fov_id>_completeness.csv`` (one row per
checked file, :func:`MERci.analysis.completeness.check_one_file`'s columns
plus ``round_id``/``fov_id``).

Self-locates its own sibling ``src/`` root from ``__file__`` (same
convention as the other ``cli_compute_*.py`` scripts) so MERci never needs
to be ``pip install``ed on the cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# .../MERci/src/MERci/analysis/cli_check_fov_completeness.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MERCI_SRC))

from MERci.analysis import _cli_common as cli  # noqa: E402
from MERci.common.metadata import ExperimentMetadata          # noqa: E402
from MERci.analysis.completeness import check_one_file         # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-info-csv", required=True, type=Path)
    p.add_argument("--positions-txt", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--image-suffix", default=".zarr")
    p.add_argument("--round-ids", required=True,
                    help="Comma-separated round ids to check for this FOV.")
    p.add_argument("--manifest", required=True, type=Path,
                    help="CSV with a fov_id column -- one row per pending FOV.")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write 'fov<fov_id>_completeness.csv' into.")
    cli.add_task_args(p)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    task_id = cli.task_id(args)
    fov_id = int(cli.manifest_row(args.manifest, task_id)["fov_id"])
    round_ids = {int(r) for r in args.round_ids.split(",")}

    meta = ExperimentMetadata.load(args.round_info_csv, args.positions_txt, args.data_dir,
                                    image_suffix=args.image_suffix)

    rows = []
    for fpath in meta.files_for_fov(fov_id):
        round_id = meta.round_id_of_file(fpath)
        if round_id not in round_ids:
            continue
        row = check_one_file(fpath)
        row["round_id"] = round_id
        row["fov_id"] = fov_id
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"fov{fov_id:04d}_completeness.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    n_bad = sum(1 for r in rows if r["status"] != "ok" and r["status"] != "not_checked")
    print(f"Done: FOV {fov_id} -- {len(rows)} file(s) checked across {len(round_ids)} round(s), "
          f"{n_bad} flagged -> {out_path}")


if __name__ == "__main__":
    main()
