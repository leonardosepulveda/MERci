# MERci/fov.py
"""
FOV-level analyses.  All public functions accept a pre-loaded numpy array
(so the scheduler can read the file once and pass it to multiple functions).

Functions
---------
create_thumbnail            – downsample + contrast-stretch one frame → PNG
create_thumbnails_for_stack – batch version over many frames
measure_stats               – per-frame min/mean/median/max → CSV
get_histogram               – per-frame intensity histograms → .npz
get_histogram_for_frames    – histogram only selected frame indices (partial read) → .npz
load_stats                  – convenience: load a saved stats CSV
load_histogram              – convenience: load a saved histogram .npz
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Atomic helper ─────────────────────────────────────────────────────────────

def _atomic_save(path: Path, save_fn) -> None:
    """
    Call ``save_fn(tmp_path)`` then rename ``tmp_path`` → ``path`` atomically.
    Prevents other processes from reading a partially-written output file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    try:
        save_fn(tmp)
        tmp.replace(path)   # atomic on POSIX; overwrites destination
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


# ── Thumbnail ─────────────────────────────────────────────────────────────────

def create_thumbnail(
    frame: np.ndarray,
    output_path: Path,
    target_size: Tuple[int, int] = (200, 200),
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """
    Downsample and contrast-stretch a 2-D image frame; save as PNG.

    Parameters
    ----------
    frame          : 2-D array (H × W), any unsigned integer dtype
    output_path    : destination PNG path; parent directories are created
    target_size    : output (width, height) in pixels
    percentile_clip: (lo_pct, hi_pct) for contrast stretching

    Returns
    -------
    uint8 array of shape (height, width)

    Notes
    -----
    Uses ``skimage.transform.resize`` with ``anti_aliasing=True`` for
    high-quality downsampling, then writes via Pillow for broad compatibility.
    """
    from PIL import Image
    from skimage.transform import resize as sk_resize

    if frame.ndim != 2:
        raise ValueError(f"Expected 2-D frame, got shape {frame.shape}")

    # Contrast stretch to [0, 1]
    lo, hi = np.percentile(frame, [percentile_clip[0], percentile_clip[1]])
    if hi > lo:
        stretched = np.clip(
            (frame.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0
        )
    else:
        stretched = np.zeros(frame.shape, dtype=np.float32)

    # Downsample: target_size is (W, H); skimage expects (H, W)
    tw, th = target_size
    resized = sk_resize(
        stretched, (th, tw),
        anti_aliasing=True,
        preserve_range=True,
    )
    thumb = (resized * 255).clip(0, 255).astype(np.uint8)

    _atomic_save(
        Path(output_path),
        lambda tmp: Image.fromarray(thumb).save(str(tmp)),
    )
    log.debug("Thumbnail saved: %s", output_path)
    return thumb


def create_thumbnails_for_stack(
    stack: np.ndarray,
    stem: str,
    output_dir: Path,
    frame_indices: Optional[List[int]] = None,
    target_size: Tuple[int, int] = (200, 200),
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
) -> List[Path]:
    """
    Create thumbnails for one or more frames in an image stack.

    Parameters
    ----------
    stack         : 3-D array (n_frames, H, W)
    stem          : base filename stem (no extension) for naming outputs
    output_dir    : directory to write PNG files
    frame_indices : which frames to process; ``None`` → all frames
    target_size   : output thumbnail size (width, height) in pixels
    percentile_clip: contrast stretch percentiles

    Returns
    -------
    List of Paths for the written (or pre-existing) PNG files
    """
    if frame_indices is None:
        frame_indices = list(range(len(stack)))

    paths = []
    for fi in frame_indices:
        out = Path(output_dir) / f"{stem}_frame{fi:03d}.png"
        if not out.exists():
            create_thumbnail(stack[fi], out, target_size, percentile_clip)
        paths.append(out)
    return paths


# ── Per-frame statistics ──────────────────────────────────────────────────────

def measure_stats(
    stack: np.ndarray,
    output_path: Path,
    source_filename: str = "",
) -> pd.DataFrame:
    """
    Compute per-frame statistics and write to a CSV file.

    Columns
    -------
    file, frame, min, max, mean, median, std, p01, p99

    Parameters
    ----------
    stack           : 3-D array (n_frames, H, W)
    output_path     : destination CSV path
    source_filename : written in the ``file`` column for traceability

    Returns
    -------
    pd.DataFrame with one row per frame
    """
    records = []
    for fi, frame in enumerate(stack):
        flat = frame.ravel().astype(np.float64)
        records.append({
            "file":   source_filename,
            "frame":  fi,
            "min":    int(flat.min()),
            "max":    int(flat.max()),
            "mean":   float(flat.mean()),
            "median": float(np.median(flat)),
            "std":    float(flat.std()),
            "p01":    float(np.percentile(flat,  1)),
            "p99":    float(np.percentile(flat, 99)),
        })

    df = pd.DataFrame(records)
    _atomic_save(
        Path(output_path),
        lambda tmp: df.to_csv(str(tmp), index=False),
    )
    log.debug("Stats saved (%d frames): %s", len(records), output_path)
    return df


# ── Intensity histograms ──────────────────────────────────────────────────────

def get_histogram(
    stack: np.ndarray,
    output_path: Path,
    bins: int = 512,
    hist_range: Tuple[int, int] = (0, 65535),
) -> Dict:
    """
    Compute per-frame intensity histograms; save to a compressed .npz file.

    Saved arrays
    ------------
    counts      : shape (n_frames, bins) – integer bin counts per frame
    bin_centers : shape (bins,)  – bin centre values
    bin_edges   : shape (bins+1,)

    Parameters
    ----------
    stack       : 3-D array (n_frames, H, W)
    output_path : destination; extension is replaced with ``.npz``
    bins        : number of histogram bins
    hist_range  : (lo, hi) intensity range

    Returns
    -------
    dict with keys ``counts``, ``bin_centers``, ``bin_edges``
    """
    output_path = Path(output_path).with_suffix(".npz")

    n_frames = len(stack)
    all_counts = np.zeros((n_frames, bins), dtype=np.int64)
    edges: Optional[np.ndarray] = None

    for fi, frame in enumerate(stack):
        counts, edges = np.histogram(frame.ravel(), bins=bins, range=hist_range)
        all_counts[fi] = counts

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    result = {
        "counts":      all_counts,
        "bin_centers": bin_centers,
        "bin_edges":   edges,
    }
    _atomic_save(
        output_path,
        lambda tmp: np.savez_compressed(str(tmp), **result),
    )
    log.debug("Histogram saved (%d frames, %d bins): %s", n_frames, bins, output_path)
    return result


# ── Whole-file driver (parallel worker) ────────────────────────────────────────

def analyze_file(
    image_path:                Path,
    *,
    thumbnails_dir:            Path,
    stats_path:                Path,
    histogram_path:            Path,
    sentinel_path:             Path,
    frame_width:               Optional[int]            = None,
    frame_height:              Optional[int]            = None,
    thumbnail_frames:          Optional[List[int]]      = None,
    thumbnail_size:            Tuple[int, int]          = (200, 200),
    thumbnail_percentile_clip: Tuple[float, float]      = (1.0, 99.0),
    histogram_bins:            int                      = 512,
    histogram_range:           Tuple[int, int]          = (0, 65535),
) -> str:
    """
    Run every FOV-level analysis for one image file: read the stack **once**,
    then write thumbnails, per-frame stats, and histograms, and finally touch the
    ``sentinel_path`` to mark the FOV complete.

    This is a top-level, picklable function so it can be dispatched to a
    :class:`concurrent.futures.ProcessPoolExecutor` worker — the whole file is
    handled inside a single process (one disk read, all analyses), which is the
    unit of cross-FOV parallelism used by :class:`MERci.scheduler.FOVScheduler`.

    All output paths are passed in (computed deterministically by the parent from
    the file stem) so the worker needs neither the config nor the tracker object.

    Returns
    -------
    The image filename (for logging by the parent).
    """
    # Local import keeps process-spawn overhead off the module top level and
    # avoids importing the reader in parents that only need the array helpers.
    from MERci.common.io import read_image

    image_path = Path(image_path)
    stack = read_image(image_path, frame_width=frame_width, frame_height=frame_height)
    try:
        create_thumbnails_for_stack(
            stack,
            stem=image_path.stem,
            output_dir=Path(thumbnails_dir),
            frame_indices=thumbnail_frames,
            target_size=thumbnail_size,
            percentile_clip=thumbnail_percentile_clip,
        )
        if not Path(stats_path).exists():
            measure_stats(stack, Path(stats_path), source_filename=image_path.name)
        if not Path(histogram_path).exists():
            get_histogram(
                stack, Path(histogram_path),
                bins=histogram_bins, hist_range=histogram_range,
            )
    finally:
        del stack   # release ~200 MB per file promptly

    sentinel = Path(sentinel_path)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    return image_path.name


def compute_histogram_only(
    image_path:      Path,
    histogram_path:  Path,
    frame_width:     Optional[int]      = None,
    frame_height:    Optional[int]      = None,
    histogram_bins:  int                = 512,
    histogram_range: Tuple[int, int]    = (0, 65535),
) -> str:
    """
    Read one image file's stack and save its per-frame histogram --
    unlike :func:`analyze_file`, does NOT also compute thumbnails/stats,
    for callers that only need the histogram (e.g. a process-pool-
    parallelized backfill outside the standard FOV-analysis pipeline).

    Top-level, picklable function so it can be dispatched to a
    :class:`concurrent.futures.ProcessPoolExecutor` worker, same
    convention as :func:`analyze_file`.

    Returns
    -------
    The image filename (for logging by the parent).
    """
    from MERci.common.io import read_image

    image_path = Path(image_path)
    stack = read_image(image_path, frame_width=frame_width, frame_height=frame_height)
    try:
        if not Path(histogram_path).exists():
            get_histogram(stack, Path(histogram_path), bins=histogram_bins, hist_range=histogram_range)
    finally:
        del stack
    return image_path.name


def get_histogram_for_frames(
    image_path:      Path,
    histogram_path:  Path,
    frame_indices:   List[int],
    frame_width:     Optional[int]      = None,
    frame_height:    Optional[int]      = None,
    histogram_bins:  int                = 512,
    histogram_range: Tuple[int, int]    = (0, 65535),
) -> Dict:
    """
    Histogram only the requested *frame_indices* out of a stack (e.g. one
    channel's z-range out of a much larger multi-color acquisition), reading
    only those frames off disk via :func:`MERci.common.io.read_image_frames`
    instead of the whole stack -- unlike :func:`compute_histogram_only`,
    which always reads and histograms every frame.

    Saved arrays (in the .npz at *histogram_path*)
    ------------------------------------------------
    counts        : shape (len(frame_indices), bins) -- rows in the SAME
                    order as frame_indices, not the stack's own frame order.
    frame_indices : the original (global, whole-stack) frame indices each
                    row of ``counts`` corresponds to -- needed since this
                    histogram covers a subset, not every frame 0..n-1.
    bin_centers, bin_edges : same as :func:`get_histogram`.

    Returns
    -------
    dict with keys ``counts``, ``frame_indices``, ``bin_centers``, ``bin_edges``
    """
    from MERci.common.io import read_image_frames

    image_path     = Path(image_path)
    histogram_path = Path(histogram_path).with_suffix(".npz")
    frame_indices  = list(frame_indices)

    stack = read_image_frames(image_path, frame_indices,
                              frame_width=frame_width, frame_height=frame_height)
    try:
        all_counts = np.zeros((len(frame_indices), histogram_bins), dtype=np.int64)
        edges: Optional[np.ndarray] = None
        for row, frame in enumerate(stack):
            counts, edges = np.histogram(frame.ravel(), bins=histogram_bins, range=histogram_range)
            all_counts[row] = counts
    finally:
        del stack

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    result = {
        "counts":        all_counts,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "bin_centers":   bin_centers,
        "bin_edges":     edges,
    }
    _atomic_save(histogram_path, lambda tmp: np.savez_compressed(str(tmp), **result))
    log.debug("Histogram saved (%d/%d frames, %d bins): %s",
              len(frame_indices), len(frame_indices), histogram_bins, histogram_path)
    return result


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_stats(stats_csv: Path) -> pd.DataFrame:
    """Load a previously saved stats CSV into a DataFrame."""
    return pd.read_csv(stats_csv)


def load_histogram(hist_npz: Path) -> Dict:
    """Load a previously saved histogram .npz into a plain dict."""
    data = np.load(str(hist_npz))
    return {k: data[k] for k in data.files}