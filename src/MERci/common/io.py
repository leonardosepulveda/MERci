# MERci/common/io.py
"""
Low-level I/O utilities shared by acquisition-planning and analysis modules.

Functions
---------
load_round_info         – read round_info.csv
load_positions          – read comma-separated positions file
save_positions_array    – write (N,2) array to comma-separated file
parse_inf               – parse HAL .inf sidecar
read_dax                – load a raw .dax file into a numpy array  (uint16)
read_zarr               – load a HAL .zarr store into a numpy array (uint16)
read_tiff               – load a multi-page .tiff into a numpy array (uint16)
read_image              – format-agnostic dispatcher for the three formats above
get_dax_shape           – read .dax shape without loading pixel data
discover_image_files    – scan a directory for stable image files
                          (handles flat files and .zarr directory stores)
is_path_stable          – single-path stability check (used by
                          discover_image_files and by callers that already
                          have one specific path, not a directory to scan)
path_mtime              – effective last-write mtime of a file OR a directory
                          store (max mtime of its contents, zarr-aware)
read_dax_frames/read_zarr_frames/read_tiff_frames/read_image_frames
                        – read only the given frame_indices (real partial I/O)
iter_dax_frames/iter_zarr_frames/iter_tiff_frames/iter_image_frames
                        – same, but lazy: yields (frame_idx, frame) one at a
                          time, so a caller can stop partway through without
                          reading the remaining frames at all
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Round info & positions ────────────────────────────────────────────────────

def load_round_info(csv_path: Path) -> pd.DataFrame:
    """
    Load ``round_info.csv``.

    Required columns: ``round_id``, ``series``
    Optional columns: ``imaging_type``, ``hal_config``, ``shutter_file``, others
    """
    df = pd.read_csv(csv_path)
    for col in ("round_id", "series"):
        if col not in df.columns:
            raise ValueError(
                f"round_info.csv must contain a '{col}' column "
                f"(found: {list(df.columns)})"
            )
    df["series"]   = df["series"].astype(str).str.strip()
    df["round_id"] = df["round_id"].astype(int)
    return df


def load_positions(positions_path: Path) -> Dict[int, Tuple[float, float]]:
    """
    Load a comma-separated positions file (``x,y`` per line, one FOV per line).

    Lines beginning with ``#`` and blank lines are ignored.

    Returns
    -------
    {fov_id: (x, y)} — zero-indexed.
    """
    positions: Dict[int, Tuple[float, float]] = {}
    fov_id = 0

    with Path(positions_path).open() as fh:
        for raw in fh:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                log.warning("Skipping short positions line at FOV %d: %r", fov_id, raw)
                continue
            try:
                positions[fov_id] = (float(parts[0]), float(parts[1]))
                fov_id += 1
            except ValueError as exc:
                log.warning("Bad position at FOV %d: %s", fov_id, exc)

    return positions


def save_positions_array(coords: np.ndarray, output_path: Path) -> None:
    """
    Write an ``(N, 2)`` array of ``(x, y)`` stage coordinates to a
    comma-separated text file.
    """
    coords = np.asarray(coords, dtype=float)
    with Path(output_path).open("w") as fh:
        for row in coords:
            fh.write(f"{row[0]},{row[1]}\n")


# ── DAX / INF file I/O ────────────────────────────────────────────────────────

def parse_inf(inf_path: Path) -> Dict[str, Any]:
    """
    Parse a HAL-style ``.inf`` metadata sidecar file into a plain dict.

    Adds convenience keys ``frame_width``, ``frame_height``, ``n_frames``
    when the corresponding lines are found.
    """
    import re as _re
    info: Dict[str, Any] = {}

    with Path(inf_path).open() as fh:
        for raw in fh:
            line = raw.split(";")[0].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            info[key.strip().lower()] = val.strip()

    if "frame dimensions" in info:
        m = _re.search(r"(\d+)\s*[xX]\s*(\d+)", info["frame dimensions"])
        if m:
            info["frame_width"]  = int(m.group(1))
            info["frame_height"] = int(m.group(2))

    if "number of frames" in info:
        try:
            info["n_frames"] = int(info["number of frames"])
        except ValueError:
            pass

    return info


def read_dax(
    dax_path:     Path,
    frame_width:  Optional[int] = None,
    frame_height: Optional[int] = None,
    n_frames:     Optional[int] = None,
    dtype:        type          = np.uint16,
) -> np.ndarray:
    """
    Read a raw DAX file; return a ``(n_frames, height, width)`` array.

    Dimension resolution order (highest priority first):

    1. Explicit keyword arguments
    2. ``.inf`` sidecar (same directory, same stem)
    3. Infer ``n_frames`` from file size

    Raises
    ------
    ValueError  if width or height cannot be determined
    IOError     if the file is smaller than expected
    """
    dax_path = Path(dax_path)
    inf_path = dax_path.with_suffix(".inf")

    inf: Dict[str, Any] = {}
    if inf_path.exists():
        try:
            inf = parse_inf(inf_path)
        except Exception as exc:
            log.warning("Could not parse %s: %s", inf_path, exc)

    fw = frame_width  or inf.get("frame_width")
    fh = frame_height or inf.get("frame_height")
    nf = n_frames     or inf.get("n_frames")

    if fw is None or fh is None:
        raise ValueError(
            f"Frame dimensions unknown for '{dax_path}'. "
            "Provide frame_width/frame_height or ensure a .inf sidecar exists."
        )

    raw             = np.fromfile(str(dax_path), dtype=dtype)
    pixels_per_frame = int(fw) * int(fh)

    if nf is None:
        nf = len(raw) // pixels_per_frame
        log.debug("Inferred n_frames=%d for %s", nf, dax_path.name)

    needed = int(nf) * pixels_per_frame
    if len(raw) < needed:
        raise IOError(
            f"'{dax_path.name}' has {len(raw)} values; "
            f"expected {needed} ({nf} frames × {fh}×{fw})."
        )

    return raw[:needed].reshape(int(nf), int(fh), int(fw))


def get_dax_shape(
    dax_path:     Path,
    frame_width:  Optional[int] = None,
    frame_height: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Return ``(n_frames, height, width)`` without loading pixel data.
    Uses the ``.inf`` sidecar when available; falls back to file-size inference.
    """
    dax_path = Path(dax_path)
    inf_path = dax_path.with_suffix(".inf")
    inf      = parse_inf(inf_path) if inf_path.exists() else {}

    fw = frame_width  or inf.get("frame_width")
    fh = frame_height or inf.get("frame_height")
    nf = inf.get("n_frames")

    if fw is None or fh is None:
        raise ValueError(
            f"Cannot determine frame dimensions for '{dax_path}'. "
            "Ensure a .inf sidecar exists or supply frame_width/frame_height."
        )

    if nf is None:
        item_bytes = np.dtype(np.uint16).itemsize
        nf = dax_path.stat().st_size // (int(fw) * int(fh) * item_bytes)

    return int(nf), int(fh), int(fw)


# ── Multi-format readers ──────────────────────────────────────────────────────

def read_zarr(
    zarr_path: Path,
    dtype:     type = np.uint16,
) -> np.ndarray:
    """
    Load a HAL-written ``.zarr`` store and return a ``(n_frames, H, W)`` array.

    The store is expected to contain a single zarr Array at the root level
    (the format written by Storm Control / HAL2).  If a zarr Group is found,
    the first array child is used.

    Parameters
    ----------
    zarr_path : path to the ``.zarr`` directory store
    dtype     : cast output to this dtype (default ``uint16``)

    Returns
    -------
    numpy array of shape ``(n_frames, height, width)``
    """
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "The 'zarr' package is required for .zarr support. "
            "Install it with: pip install zarr"
        ) from exc

    store = zarr.open(str(zarr_path), mode="r")
    if isinstance(store, zarr.Array):
        arr = store[:]
    else:
        # Group: use the first (and typically only) array child
        keys = [k for k in store.keys() if isinstance(store[k], zarr.Array)]
        if not keys:
            raise ValueError(
                f"No zarr Array found inside group store: {zarr_path}"
            )
        arr = store[keys[0]][:]

    return arr.astype(dtype)


def read_tiff(
    tiff_path: Path,
    dtype:     type = np.uint16,
) -> np.ndarray:
    """
    Load a multi-page ``.tiff`` file and return a ``(n_frames, H, W)`` array.

    Parameters
    ----------
    tiff_path : path to the TIFF file
    dtype     : cast output to this dtype (default ``uint16``)

    Returns
    -------
    numpy array of shape ``(n_frames, height, width)``
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "The 'tifffile' package is required for .tiff support. "
            "Install it with: pip install tifffile"
        ) from exc

    arr = tifffile.imread(str(tiff_path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]   # single frame → (1, H, W)
    return arr.astype(dtype)


def read_image(
    path:         Path,
    frame_width:  Optional[int] = None,
    frame_height: Optional[int] = None,
    n_frames:     Optional[int] = None,
    dtype:        type          = np.uint16,
) -> np.ndarray:
    """
    Format-agnostic image reader.  Dispatches to :func:`read_dax`,
    :func:`read_zarr`, or :func:`read_tiff` based on the file extension.

    Returns a ``(n_frames, height, width)`` array of *dtype*.

    The *frame_width*, *frame_height*, and *n_frames* parameters are forwarded
    to :func:`read_dax` only; zarr and tiff embed their own metadata.
    """
    path   = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dax":
        return read_dax(path, frame_width=frame_width,
                        frame_height=frame_height, n_frames=n_frames, dtype=dtype)
    if suffix == ".zarr":
        return read_zarr(path, dtype=dtype)
    if suffix in (".tiff", ".tif"):
        return read_tiff(path, dtype=dtype)
    raise ValueError(
        f"Unsupported image format '{suffix}' for path: {path}. "
        "Supported formats: .dax, .zarr, .tiff"
    )


# ── Selective per-frame reading ──────────────────────────────────────────────
#
# read_image() above always loads the WHOLE stack. A caller that only needs a
# handful of frames out of a much larger stack (e.g. one channel's z-range
# out of a multi-color, 100+-frame acquisition) pays for reading and holding
# every frame it doesn't need -- real cost under many concurrent workers, and
# real cost even sequentially for a large batch. The read_*_frames functions
# below read only the requested frame indices, with genuine partial I/O where
# the format allows it (zarr fancy-indexes just the needed chunks; dax seeks
# directly to each frame's byte offset; tifffile decodes only the requested
# pages) rather than reading everything and throwing most of it away. Each
# has a lazy iter_*_frames counterpart yielding one frame at a time -- for a
# caller that might stop partway through frame_indices (e.g. a sequential
# scan with a per-frame stopping condition), the eager versions still read
# every requested frame even if the caller only ends up using the first few.

def iter_dax_frames(
    dax_path:      Path,
    frame_indices: List[int],
    frame_width:   Optional[int] = None,
    frame_height:  Optional[int] = None,
    dtype:         type          = np.uint16,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Lazily yield ``(frame_idx, frame)`` from a raw ``.dax`` file, one frame at
    a time, seeking directly to each frame's byte offset -- unlike
    :func:`read_dax_frames`, a caller can stop partway through *frame_indices*
    (e.g. once some per-frame condition is met) without paying for the reads
    it never asked for.

    Dimension resolution follows :func:`read_dax`: explicit kwargs, then the
    ``.inf`` sidecar.
    """
    dax_path = Path(dax_path)
    inf_path = dax_path.with_suffix(".inf")
    inf: Dict[str, Any] = {}
    if inf_path.exists():
        try:
            inf = parse_inf(inf_path)
        except Exception as exc:
            log.warning("Could not parse %s: %s", inf_path, exc)

    fw = frame_width  or inf.get("frame_width")
    fh = frame_height or inf.get("frame_height")
    if fw is None or fh is None:
        raise ValueError(
            f"Frame dimensions unknown for '{dax_path}'. "
            "Provide frame_width/frame_height or ensure a .inf sidecar exists."
        )

    pixels_per_frame = int(fw) * int(fh)
    frame_bytes      = pixels_per_frame * np.dtype(dtype).itemsize

    with open(dax_path, "rb") as fh_bin:
        for idx in frame_indices:
            fh_bin.seek(idx * frame_bytes)
            raw = fh_bin.read(frame_bytes)
            if len(raw) < frame_bytes:
                raise IOError(
                    f"'{dax_path.name}': frame {idx} truncated "
                    f"(got {len(raw)} of {frame_bytes} bytes -- file too short?)"
                )
            yield idx, np.frombuffer(raw, dtype=dtype).reshape(int(fh), int(fw))


def read_dax_frames(
    dax_path:      Path,
    frame_indices: List[int],
    frame_width:   Optional[int] = None,
    frame_height:  Optional[int] = None,
    dtype:         type          = np.uint16,
) -> np.ndarray:
    """
    Read only *frame_indices* from a raw ``.dax`` file (see
    :func:`iter_dax_frames`), stacked into one array.

    Returns
    -------
    ``(len(frame_indices), height, width)`` array, in the given order.
    """
    frames = [frame for _, frame in iter_dax_frames(
        dax_path, frame_indices, frame_width=frame_width, frame_height=frame_height, dtype=dtype,
    )]
    return np.stack(frames, axis=0)


def iter_zarr_frames(
    zarr_path:     Path,
    frame_indices: List[int],
    dtype:         type = np.uint16,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Lazily yield ``(frame_idx, frame)`` from a ``.zarr`` store, one frame at a
    time -- each single-index read only fetches the chunk(s) covering that
    frame, so a caller stopping early never pays for the remaining frames.
    """
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "The 'zarr' package is required for .zarr support. "
            "Install it with: pip install zarr"
        ) from exc

    store = zarr.open(str(zarr_path), mode="r")
    if isinstance(store, zarr.Array):
        arr = store
    else:
        keys = [k for k in store.keys() if isinstance(store[k], zarr.Array)]
        if not keys:
            raise ValueError(f"No zarr Array found inside group store: {zarr_path}")
        arr = store[keys[0]]

    for idx in frame_indices:
        yield idx, np.asarray(arr[idx]).astype(dtype)


def read_zarr_frames(
    zarr_path:     Path,
    frame_indices: List[int],
    dtype:         type = np.uint16,
) -> np.ndarray:
    """
    Read only *frame_indices* from a ``.zarr`` store (see
    :func:`iter_zarr_frames`), stacked into one array -- zarr only reads the
    chunks actually covering the requested indices, not the full array
    (unlike :func:`read_zarr`'s ``[:]``).

    Returns
    -------
    ``(len(frame_indices), height, width)`` array, in the given order.
    """
    frames = [frame for _, frame in iter_zarr_frames(zarr_path, frame_indices, dtype=dtype)]
    return np.stack(frames, axis=0)


def iter_tiff_frames(
    tiff_path:     Path,
    frame_indices: List[int],
    dtype:         type = np.uint16,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Lazily yield ``(frame_idx, frame)`` from a multi-page ``.tiff``, one frame
    at a time, via a single kept-open :class:`tifffile.TiffFile` handle -- a
    caller stopping early never decodes the remaining frames.

    Uses ``TiffFile.asarray(key=idx, series=None)`` -- the same key-based
    selection :func:`read_tiff_frames`/module-level ``tifffile.imread(...,
    key=...)`` use -- NOT ``tf.pages[idx]``. A stack written by
    ``tifffile.imwrite`` from one ``(n_frames, H, W)`` array is typically
    stored as a *single* IFD/page holding every frame (tifffile's "shaped"
    convention), not one page per frame, so indexing
    ``tf.pages`` directly reads the wrong data for exactly the files this
    package itself writes. ``series=None`` matters too: leaving it at its
    default resolved a *different* (incorrect) code path in testing.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "The 'tifffile' package is required for .tiff support. "
            "Install it with: pip install tifffile"
        ) from exc

    with tifffile.TiffFile(str(tiff_path)) as tf:
        for idx in frame_indices:
            yield idx, np.asarray(tf.asarray(key=idx, series=None)).astype(dtype)


def read_tiff_frames(
    tiff_path:     Path,
    frame_indices: List[int],
    dtype:         type = np.uint16,
) -> np.ndarray:
    """
    Read only *frame_indices* (pages) from a multi-page ``.tiff`` (see
    :func:`iter_tiff_frames`), stacked into one array.

    Returns
    -------
    ``(len(frame_indices), height, width)`` array, in the given order.
    """
    frames = [frame for _, frame in iter_tiff_frames(tiff_path, frame_indices, dtype=dtype)]
    return np.stack(frames, axis=0)


def iter_image_frames(
    path:          Path,
    frame_indices: List[int],
    frame_width:   Optional[int] = None,
    frame_height:  Optional[int] = None,
    dtype:         type          = np.uint16,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Format-agnostic lazy per-frame reader. Dispatches to
    :func:`iter_dax_frames`, :func:`iter_zarr_frames`, or
    :func:`iter_tiff_frames` based on the file extension. Unlike
    :func:`read_image_frames`, this never reads ahead of what the caller
    actually consumes -- for a caller that may stop partway through
    *frame_indices* (e.g. a sequential scan with an early-exit condition),
    every frame after the stopping point is never read from disk at all.
    """
    path   = Path(path)
    suffix = path.suffix.lower()
    frame_indices = list(frame_indices)
    if suffix == ".dax":
        yield from iter_dax_frames(path, frame_indices, frame_width=frame_width,
                                   frame_height=frame_height, dtype=dtype)
    elif suffix == ".zarr":
        yield from iter_zarr_frames(path, frame_indices, dtype=dtype)
    elif suffix in (".tiff", ".tif"):
        yield from iter_tiff_frames(path, frame_indices, dtype=dtype)
    else:
        raise ValueError(
            f"Unsupported image format '{suffix}' for path: {path}. "
            "Supported formats: .dax, .zarr, .tiff"
        )


def read_image_frames(
    path:          Path,
    frame_indices: List[int],
    frame_width:   Optional[int] = None,
    frame_height:  Optional[int] = None,
    dtype:         type          = np.uint16,
) -> np.ndarray:
    """
    Format-agnostic selective-frame reader: reads every requested frame (see
    :func:`iter_image_frames` instead if the caller might stop early).

    Returns
    -------
    ``(len(frame_indices), height, width)`` array, in the given order.
    """
    frames = [frame for _, frame in iter_image_frames(
        path, frame_indices, frame_width=frame_width, frame_height=frame_height, dtype=dtype,
    )]
    return np.stack(frames, axis=0)


def discover_image_files(
    data_dir:        Path,
    suffix:          str   = ".zarr",
    recursive:       bool  = True,
    stability_check: bool  = True,
    stability_delay: float = 0.1,
) -> List[Path]:
    """
    Return a sorted list of image paths under *data_dir* that are not still
    being written.

    Handles both flat-file formats (``.dax``, ``.tiff``) and directory-based
    stores (``.zarr``).

    Parameters
    ----------
    suffix          : file extension / directory suffix to search for
    stability_check : skip entries whose size changes within *stability_delay*
                      seconds (catches partially-written files).
                      For ``.zarr`` directories the total content size is used.
    stability_delay : seconds between the two size measurements
    """
    glob       = data_dir.rglob if recursive else data_dir.glob
    candidates = sorted(glob(f"*{suffix}"))

    if not stability_check:
        stable = []
        for p in candidates:
            try:
                if p.is_dir():
                    # zarr store: accept if non-empty
                    if any(p.iterdir()):
                        stable.append(p)
                elif p.stat().st_size > 0:
                    stable.append(p)
            except (FileNotFoundError, StopIteration):
                pass
        return stable

    return [p for p in candidates if is_path_stable(p, stability_delay)]


def is_path_stable(path: Path, stability_delay: float = 0.1) -> bool:
    """
    True iff *path* -- a flat file or a directory store (e.g. ``.zarr``) --
    has not changed size in the last *stability_delay* seconds, i.e. it looks
    done being written rather than still being actively written to (HAL
    writes are incremental, so an image file/store can already ``exist()``
    -- and even already hold some real frames -- well before every frame of
    its stack has landed on disk).

    False (not True) for a path that doesn't exist, or a zero-size/empty one
    -- "can't confirm it's stable" should never be treated as "stable."
    """
    path = Path(path)
    try:
        if path.is_dir():
            s0 = _dir_content_size(path)
            if s0 == 0:
                return False
            time.sleep(stability_delay)
            s1 = _dir_content_size(path)
        else:
            s0 = path.stat().st_size
            if s0 == 0:
                return False
            time.sleep(stability_delay)
            s1 = path.stat().st_size
        return s0 == s1
    except FileNotFoundError:
        return False


def _dir_content_size(path: Path) -> int:
    """Sum of file sizes for all files under *path* (non-recursive files only)."""
    return sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file()
    )


def path_mtime(path: Path) -> float:
    """
    Effective last-write time of *path*: its own mtime for a flat file, or the
    max mtime of every file inside it for a directory store (e.g. ``.zarr``).

    A directory's own mtime only updates when an entry is added/removed/renamed
    directly inside it -- it doesn't reliably bubble up when a chunk file nested
    a level or two deeper is written, so recursing into the actual files (same
    walk as :func:`_dir_content_size`, just mtime instead of size) is needed to
    get a meaningful "when did writing finish" timestamp for a directory store.
    """
    path = Path(path)
    if path.is_dir():
        mtimes = [f.stat().st_mtime for f in path.rglob("*") if f.is_file()]
        if not mtimes:
            raise FileNotFoundError(f"No files found under directory store: {path}")
        return max(mtimes)
    return path.stat().st_mtime