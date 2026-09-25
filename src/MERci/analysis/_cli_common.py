# MERci/analysis/_cli_common.py
"""
Shared pieces of the ``cli_*.py`` SLURM array-task scripts: which manifest
entry this task handles, and the camera-orientation flags.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Union


def add_task_args(p: argparse.ArgumentParser, frame_size: bool = False) -> None:
    """``--array-task-id``; with *frame_size*, also ``--frame-width``/``--frame-height``."""
    p.add_argument("--array-task-id", type=int, default=None,
                   help="0-based manifest entry; defaults to $SLURM_ARRAY_TASK_ID "
                        "(useful for manual testing outside SLURM).")
    if frame_size:
        p.add_argument("--frame-width", type=int, default=None,
                       help="Only needed for .dax input; ignored for .zarr/.tiff.")
        p.add_argument("--frame-height", type=int, default=None)


def add_orientation_args(p: argparse.ArgumentParser) -> None:
    """This microscope's orientation flags (see ``load_microscope_orientation``)."""
    p.add_argument("--flip-horizontal", action="store_true")
    p.add_argument("--flip-vertical", action="store_true")
    p.add_argument("--transpose", action="store_true")


def orientation(args: argparse.Namespace) -> Dict[str, bool]:
    """The orientation flags as ``apply_microscope_orientation`` keyword arguments."""
    return {"flip_horizontal": args.flip_horizontal,
            "flip_vertical":   args.flip_vertical,
            "transpose":       args.transpose}


def task_id(args: argparse.Namespace) -> int:
    """``--array-task-id``, else ``$SLURM_ARRAY_TASK_ID``."""
    if args.array_task_id is not None:
        return args.array_task_id
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise SystemExit("No --array-task-id given and $SLURM_ARRAY_TASK_ID is not set "
                         "(this script is meant to run as one task of a SLURM array job).")
    return int(env)


def manifest_line(manifest: Path, index: int) -> str:
    """Entry *index* of a one-entry-per-line manifest (blank lines ignored)."""
    lines = [ln.strip() for ln in Path(manifest).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return _pick(lines, manifest, index)


def manifest_row(manifest: Path, index: int, header: bool = True) -> Union[Dict[str, str], List[str]]:
    """Row *index* of a CSV manifest: a dict if it has a header row, else a list."""
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh)) if header else [r for r in csv.reader(fh) if r]
    return _pick(rows, manifest, index)


def _pick(entries: list, manifest: Path, index: int):
    if not 0 <= index < len(entries):
        raise IndexError(f"Manifest {manifest} has {len(entries)} entries; requested index {index}.")
    return entries[index]
