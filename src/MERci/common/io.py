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
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    stable = []
    for p in candidates:
        try:
            if p.is_dir():
                # Directory store (zarr): measure total content size
                s0 = _dir_content_size(p)
                if s0 == 0:
                    continue
                time.sleep(stability_delay)
                s1 = _dir_content_size(p)
            else:
                s0 = p.stat().st_size
                if s0 == 0:
                    continue
                time.sleep(stability_delay)
                s1 = p.stat().st_size
            if s0 == s1:
                stable.append(p)
        except FileNotFoundError:
            pass
    return stable


def _dir_content_size(path: Path) -> int:
    """Sum of file sizes for all files under *path* (non-recursive files only)."""
    return sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file()
    )