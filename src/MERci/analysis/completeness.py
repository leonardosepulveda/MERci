# MERci/analysis/completeness.py
"""
Cheap, decompression-free completeness/integrity checks for raw MERFISH
image files -- confirms every expected (round, FOV) file exists and, for
``.zarr`` files (the acquisition writer's own blosc-compressed chunks),
that every chunk's actual size on disk matches its own blosc header's
declared compressed size. Catches a write interrupted partway through one
chunk -- this is the failure mode behind a real incident where a downstream
MERci analysis job failed with "error during blosc decompression" on one
silently-truncated chunk of an otherwise-present FOV file -- without ever
reading or decompressing pixel data, so it stays fast even over a whole
dataset.

Only meaningful for zarr v2 arrays using the blosc compressor, which is
what this pipeline's acquisition writer (HAL) produces; other formats
(``.dax``, ``.tiff``) get an existence-only check from
:func:`check_dataset_completeness` since there's no equivalent cheap
per-chunk trick for them.

Built for ``notebooks/after_imaging/09_check_fov_completeness.ipynb``.
"""
from __future__ import annotations

import itertools
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ..common.io import open_zarr_array

_BLOSC_HEADER_SIZE = 16  # version, versionlz, flags, typesize (1B each) + nbytes, blocksize, cbytes (4B each)


@dataclass
class ChunkIntegrityResult:
    """Result of :func:`check_zarr_chunk_integrity` for one ``.zarr`` file."""
    exists:            bool
    n_frames_found:    Optional[int]                      # arr.shape[0]; None if the file doesn't exist
    missing_chunks:    List[str] = field(default_factory=list)
    truncated_chunks:  List[Tuple[str, int, int]] = field(default_factory=list)  # (key, declared_bytes, actual_bytes)

    @property
    def ok(self) -> bool:
        return self.exists and not self.missing_chunks and not self.truncated_chunks


def _blosc_declared_size(chunk_path: Path) -> Optional[int]:
    """
    Read a blosc-compressed chunk's own 16-byte header and return its
    declared compressed size (``cbytes``, including the header itself) --
    ``None`` if the file is too short to even hold a header (itself a form
    of truncation, reported as such by the caller).
    """
    with open(chunk_path, "rb") as fh:
        header = fh.read(_BLOSC_HEADER_SIZE)
    if len(header) < _BLOSC_HEADER_SIZE:
        return None
    _, _, cbytes = struct.unpack("<III", header[4:16])
    return cbytes


def check_zarr_chunk_integrity(zarr_path: Path) -> ChunkIntegrityResult:
    """
    Check one round/FOV ``.zarr`` file's chunks without decompressing any
    of them: every chunk the array's own shape/chunk-grid metadata says
    should exist is confirmed present on disk, and its actual file size is
    compared against the compressed size its own blosc header declares --
    a mismatch means the write was interrupted partway through that chunk.

    Raises ``ValueError`` if the file isn't a zarr v2 array with the blosc
    compressor -- call sites that need to handle other formats gracefully
    should catch that (see :func:`check_dataset_completeness`).
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        return ChunkIntegrityResult(exists=False, n_frames_found=None)

    arr = open_zarr_array(zarr_path)

    meta = arr.metadata
    if getattr(meta, "zarr_format", None) != 2:
        raise ValueError(f"{zarr_path}: only zarr v2 arrays are supported (got format {meta.zarr_format!r}).")
    codec_id = getattr(meta.compressor, "codec_id", None)
    if codec_id != "blosc":
        raise ValueError(f"{zarr_path}: only blosc-compressed arrays are supported (got {codec_id!r}).")

    root = Path(arr.store.root)
    n_chunks_per_axis = [math.ceil(s / c) for s, c in zip(arr.shape, arr.chunks)]

    missing, truncated = [], []
    for coord in itertools.product(*(range(n) for n in n_chunks_per_axis)):
        key = meta.encode_chunk_key(coord)
        chunk_path = root / arr.path / key
        if not chunk_path.exists():
            missing.append(key)
            continue
        declared = _blosc_declared_size(chunk_path)
        actual = chunk_path.stat().st_size
        if declared is None or actual < declared:
            truncated.append((key, -1 if declared is None else declared, actual))

    return ChunkIntegrityResult(
        exists=True, n_frames_found=arr.shape[0],
        missing_chunks=missing, truncated_chunks=truncated,
    )


def check_one_file(fpath: Path) -> dict:
    """
    Existence + (for ``.zarr``) chunk-integrity check for one image file,
    shared by :func:`check_dataset_completeness` (small/ad-hoc, serial) and
    ``cli_check_fov_completeness.py`` (one SLURM array task per FOV, for
    full-dataset scale -- see that script's own docstring for why a serial
    scan doesn't finish in reasonable time over a whole experiment).

    Returns a dict: ``path``, ``format`` (``"zarr"``/``"other"``),
    ``exists``, ``n_frames_found``, ``n_missing_chunks``,
    ``n_truncated_chunks`` (the latter three ``None`` for non-zarr files),
    ``status`` -- one of ``"ok"``, ``"missing_file"``, ``"missing_chunks"``,
    ``"truncated_chunks"``, ``"not_checked"`` (existence-only formats), or
    ``"error: ..."`` (a zarr file that isn't v2 blosc, so couldn't be
    chunk-checked).
    """
    fpath = Path(fpath)
    row = {"path": str(fpath)}

    if fpath.suffix != ".zarr":
        exists = fpath.exists()
        row.update(format="other", exists=exists, n_frames_found=None,
                   n_missing_chunks=None, n_truncated_chunks=None,
                   status="missing_file" if not exists else "not_checked")
        return row

    try:
        result = check_zarr_chunk_integrity(fpath)
    except ValueError as exc:
        row.update(format="zarr", exists=fpath.exists(), n_frames_found=None,
                   n_missing_chunks=None, n_truncated_chunks=None, status=f"error: {exc}")
        return row

    n_missing, n_truncated = len(result.missing_chunks), len(result.truncated_chunks)
    status = ("missing_file" if not result.exists else
               "missing_chunks" if n_missing else
               "truncated_chunks" if n_truncated else
               "ok")
    row.update(format="zarr", exists=result.exists, n_frames_found=result.n_frames_found,
               n_missing_chunks=n_missing, n_truncated_chunks=n_truncated, status=status)
    return row


def check_dataset_completeness(meta, round_ids=None, progress_reporter=None):
    """
    Check every expected image file across *round_ids* (default: every
    round in *meta*, via ``meta.valid_round_ids()``), via
    :func:`check_one_file`. Serial -- fine for a handful of FOVs/rounds
    (e.g. re-checking one suspect FOV), but far too slow over a whole
    dataset (see :func:`check_one_file`'s docstring); use the
    ``09_check_fov_completeness.ipynb`` SLURM-array path for that.

    *progress_reporter*, if given, should be a fresh
    ``MERci.progress_display.ProgressReporter`` (its ``total`` set to the
    number of files this call will check) -- wrapped around the per-file
    loop.

    Returns a ``pandas.DataFrame``, one row per (round_id, fov_id) --
    :func:`check_one_file`'s columns plus ``round_id``/``fov_id``.
    """
    import pandas as pd

    if round_ids is None:
        round_ids = meta.valid_round_ids()

    files = [(rid, fpath) for rid in round_ids for fpath in meta.files_for_round(rid)]
    iterable = progress_reporter.wrap(files) if progress_reporter is not None else files

    rows = []
    for round_id, fpath in iterable:
        row = check_one_file(fpath)
        row["round_id"] = round_id
        row["fov_id"] = meta.fov_id_of_file(fpath)
        rows.append(row)

    return pd.DataFrame(rows)


def load_completeness_results(output_dir: Path):
    """
    Concatenate every ``fov*_completeness.csv`` written by
    ``cli_check_fov_completeness.py`` array tasks under *output_dir* into
    one ``pandas.DataFrame`` (empty, same columns, if none exist yet).
    """
    import pandas as pd

    paths = sorted(Path(output_dir).glob("fov*_completeness.csv"))
    if not paths:
        return pd.DataFrame(columns=["round_id", "fov_id", "path", "format", "exists",
                                      "n_frames_found", "n_missing_chunks", "n_truncated_chunks", "status"])
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
