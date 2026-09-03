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
compute_channel_counters    – EXACT, bin-width-1 histogram ("Counter": sparse
                              (value, count) pairs via numpy.unique) of every
                              z-plane in a given channel's frame range
save_channel_counters/load_channel_counters – persist/reload a compute_channel_counters() result
counter_mean/counter_percentile/rebin_counter/tpc_from_counter
                            – derive a mean / percentile / re-binned histogram /
                              true-pixel count from a (values, counts) Counter,
                              no raw pixel re-read needed
tpc_profile_from_counters  – per-z true-pixel-count profile (z_first_um/z_last_um/
                              is_contiguous) purely from a compute_channel_counters() result
resolve_round_by_imaging_type – (imaging_round, frame_table) for round_info.csv's first
                              row matching a given imaging_type (e.g. "cells")
compute_tissue_fraction     – per-FOV true-pixel-count tissue coverage (0-1), same
                              method/estimator as misc/measure_tissue_thickness_test.ipynb
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


def compute_channel_counters(
    image_path:      Path,
    z_frame_indices: List[Tuple[int, float]],
    frame_width:     Optional[int] = None,
    frame_height:    Optional[int] = None,
) -> Dict:
    """
    Read every z-plane in *z_frame_indices* for one FOV (e.g. one channel's
    z-range out of a much larger multi-color acquisition -- only these frames
    are ever read, via :func:`MERci.common.io.iter_image_frames`, not the
    whole stack) and build an EXACT, bin-width-1 histogram of each frame: a
    true Counter over observed pixel intensities, stored *sparsely* (only
    intensity values that actually occur, via :func:`numpy.unique`) rather
    than a dense fixed-width array.

    Exact, per-intensity-value counts mean any downstream view -- log-scale,
    linear-scale, an arbitrary ``[lo, hi]`` range, a percentile cutoff, a
    true-pixel count against any threshold -- can be derived later
    (:func:`rebin_counter`, :func:`counter_mean`, :func:`counter_percentile`,
    :func:`tpc_from_counter`) without re-reading pixels or losing precision to
    a fixed binning choice picked up front.

    Returns
    -------
    dict with keys:
      z_um          : (n_z,) z of each frame, in the given order
      frame_indices : (n_z,) global frame index of each frame
      values_per_z  : list of length n_z; values_per_z[i] = sorted unique
                       intensity values observed in frame i
      counts_per_z  : list of length n_z; counts_per_z[i] = pixel count for
                       each of values_per_z[i] (same order/length)
    """
    from MERci.common.io import iter_image_frames

    z_frame_indices = list(z_frame_indices)
    frame_indices   = [idx for idx, _ in z_frame_indices]
    z_um            = np.asarray([z for _, z in z_frame_indices], dtype=np.float64)

    values_per_z: List[np.ndarray] = []
    counts_per_z: List[np.ndarray] = []
    for _, frame in iter_image_frames(image_path, frame_indices,
                                      frame_width=frame_width, frame_height=frame_height):
        values, counts = np.unique(frame, return_counts=True)
        values_per_z.append(values.astype(np.int32))
        counts_per_z.append(counts.astype(np.int64))

    return {
        "z_um":          z_um,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "values_per_z":  values_per_z,
        "counts_per_z":  counts_per_z,
    }


def save_channel_counters(path: Path, data: Dict) -> None:
    """Persist a :func:`compute_channel_counters` result to a compressed .npz."""
    path = Path(path).with_suffix(".npz")
    payload = {
        "z_um":          np.asarray(data["z_um"], dtype=np.float64),
        "frame_indices": np.asarray(data["frame_indices"], dtype=np.int64),
        # Ragged (different length per z) -- stored as object arrays; np.load
        # needs allow_pickle=True to read these back (see load_channel_counters).
        "values_per_z":  np.array(data["values_per_z"], dtype=object),
        "counts_per_z":  np.array(data["counts_per_z"], dtype=object),
    }
    _atomic_save(path, lambda tmp: np.savez_compressed(str(tmp), **payload))


def load_channel_counters(path: Path) -> Dict:
    """Load a previously saved :func:`compute_channel_counters` result."""
    data = np.load(Path(path), allow_pickle=True)
    return {
        "z_um":          data["z_um"],
        "frame_indices": data["frame_indices"],
        "values_per_z":  list(data["values_per_z"]),
        "counts_per_z":  list(data["counts_per_z"]),
    }


def counter_mean(values: np.ndarray, counts: np.ndarray) -> float:
    """Exact mean intensity from a ``(values, counts)`` Counter -- weighted average, no raw pixel re-read."""
    counts = counts.astype(np.float64)
    return float(np.sum(values.astype(np.float64) * counts) / np.sum(counts))


def counter_percentile(values: np.ndarray, counts: np.ndarray, pct: float) -> float:
    """
    Exact *pct*-th percentile of the pixel-intensity distribution represented
    by a ``(values, counts)`` Counter. Assumes *values* is sorted ascending
    (true of :func:`numpy.unique`'s output, which :func:`compute_channel_counters`
    always uses).
    """
    cum    = np.cumsum(counts.astype(np.float64))
    target = pct / 100.0 * cum[-1]
    idx    = min(int(np.searchsorted(cum, target)), len(values) - 1)
    return float(values[idx])


def rebin_counter(values: np.ndarray, counts: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Re-bin an exact ``(values, counts)`` Counter into arbitrary *bin_edges*
    (log-spaced, linear over a percentile-derived range, ...) -- exact
    re-derivation from the Counter, no raw pixel re-read needed.
    """
    hist, _ = np.histogram(values, bins=bin_edges, weights=counts.astype(np.float64))
    return hist


def tpc_from_counter(values: np.ndarray, counts: np.ndarray, threshold: float) -> int:
    """True-pixel count (# pixels with intensity >= *threshold*) from a ``(values, counts)`` Counter."""
    return int(counts[values >= threshold].sum())


def _summarize_tpc_profile(z_um: np.ndarray, tpc: np.ndarray, tpc_threshold: float) -> Dict:
    """
    Shared by :func:`tpc_profile_from_counters`: given every z's true-pixel
    count, find the shallowest and deepest z with signal, and flag whether
    every z in between also had signal.

    Real data showed tissue signal doesn't always start at the shallowest
    requested z -- some FOVs are blank at first and only start showing
    signal partway down, and a small subset show it turn off and back on
    more than once (debris, folded tissue, noise). So both a first and a
    last z are reported (not just the deepest), and ``is_contiguous`` flags
    the off-then-on case for anyone reviewing results.

    Returns
    -------
    dict with keys ``z_first_um``, ``z_last_um`` (``None`` if no z passed),
    ``is_contiguous`` (``True`` vacuously when no z passed).
    """
    passing = tpc > tpc_threshold
    if not passing.any():
        return {"z_first_um": None, "z_last_um": None, "is_contiguous": True}
    idx = np.flatnonzero(passing)
    return {
        "z_first_um":    float(z_um[idx[0]]),
        "z_last_um":     float(z_um[idx[-1]]),
        "is_contiguous": bool(passing[idx[0]:idx[-1] + 1].all()),
    }


def tpc_profile_from_counters(channel_counters: Dict, threshold: float, tpc_threshold: float) -> Dict:
    """
    Derive a per-z true-pixel-count profile (z_first_um/z_last_um/is_contiguous,
    see :func:`_summarize_tpc_profile`) purely from an already-computed
    :func:`compute_channel_counters` result -- pure in-memory arithmetic
    (:func:`tpc_from_counter` per z), no raw pixel or disk read at all, since
    every z's exact per-intensity counts are already available.

    Returns
    -------
    dict with keys ``z_um``, ``tpc``, ``z_first_um``, ``z_last_um``, ``is_contiguous``.
    """
    z_um = channel_counters["z_um"]
    tpc = np.asarray(
        [tpc_from_counter(values, counts, threshold)
         for values, counts in zip(channel_counters["values_per_z"], channel_counters["counts_per_z"])],
        dtype=np.int64,
    )
    return {"z_um": z_um, "tpc": tpc, **_summarize_tpc_profile(z_um, tpc, tpc_threshold)}


# ── Tissue fraction (per-FOV, single-channel true-pixel-count coverage) ────────

def resolve_round_by_imaging_type(config, imaging_type: str) -> Tuple[int, pd.DataFrame]:
    """
    ``(imaging_round, frame_table)`` for ``round_info.csv``'s first row whose
    ``imaging_type`` matches (e.g. ``"cells"``).

    Raises
    ------
    ValueError if no round has that ``imaging_type``.
    """
    from ..acquisition.configs import find_frame_table_for_hal_config

    round_info = pd.read_csv(config.round_info_csv)
    match = round_info.loc[round_info["imaging_type"] == imaging_type]
    if match.empty:
        raise ValueError(f"No round with imaging_type={imaging_type!r} in {config.round_info_csv}")
    row = match.iloc[0]
    round_id = int(row["imaging_round"])
    hal_path = config.settings_dir / row["hal_config"]
    ft_path  = find_frame_table_for_hal_config(hal_path, config.metadata_dir)
    frame_table = pd.read_csv(ft_path, index_col=0)
    return round_id, frame_table


def compute_tissue_fraction(
    config, meta, cache_dir: Path, label: str,
    *, channel_nm: float = 405.0, n_background_frames: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """
    Per-FOV ``tissue_fraction`` (0-1): what fraction of the frame is actually
    covered by tissue, using the same true-pixel-count (TPC) method as
    ``misc/measure_tissue_thickness_test.ipynb`` -- reused here as a proxy
    for "is this FOV mostly empty of tissue" (as opposed to genuinely low
    barcode/foci density in a FOV that IS covered by tissue).

    1. For every FOV, build an exact per-z pixel-intensity Counter
       (:func:`compute_channel_counters`) for *channel_nm* in the ``cells``
       round -- cached per FOV under ``cache_dir/channel_counters/``.
    2. Estimate a THRESHOLD as the highest pixel value observed among the
       *n_background_frames* lowest-mean frames across every FOV/z -- the
       highest value that can plausibly occur as background noise.
    3. Per FOV, ``tissue_fraction`` = the largest single-z true-pixel
       fraction (pixels >= THRESHOLD, divided by total pixels) seen across
       that FOV's z-stack -- not one fixed z, since real tissue can sit at
       a different z per FOV.

    Parameters
    ----------
    config, meta : ExperimentConfig, ExperimentMetadata for the dataset
    cache_dir    : per-FOV Counters are cached under ``cache_dir/channel_counters/``
    label        : prefix for progress/print messages (e.g. a sample name)
    channel_nm   : which color channel is DAPI/cells (default 405.0)
    n_background_frames : how many lowest-mean frames to treat as background
                   for THRESHOLD estimation

    Returns
    -------
    (results_df[fov_id, tissue_fraction], threshold)
    """
    from ..common.io import read_image_frames
    from ..progress_display import ProgressReporter

    round_id, frame_table = resolve_round_by_imaging_type(config, "cells")
    channel_frames  = frame_table[frame_table["color"].round(0) == round(channel_nm)].sort_values("z")
    z_frame_indices = list(zip(channel_frames.index.tolist(), channel_frames["z"].tolist()))
    if not z_frame_indices:
        raise ValueError(f"No frames found for channel {channel_nm} nm in the 'cells' round's frame table.")

    counters_dir = Path(cache_dir) / "channel_counters"
    counters_dir.mkdir(parents=True, exist_ok=True)

    def counters_path(fpath):
        return counters_dir / f"{Path(fpath).stem}_counters.npz"

    files = [f for f in meta.files_for_round(round_id) if f.exists()]
    channel_counters, to_compute = {}, []
    for fpath in files:
        if counters_path(fpath).exists():
            channel_counters[meta.fov_id_of_file(fpath)] = load_channel_counters(counters_path(fpath))
        else:
            to_compute.append(fpath)
    print(f"[{label}] {len(channel_counters)} / {len(files)} DAPI channel Counter(s) already cached; "
          f"computing {len(to_compute)} more ({len(z_frame_indices)} z-step(s) each).")

    if to_compute:
        reporter = ProgressReporter(total=len(to_compute), label=f"[{label}] Computing DAPI Counters")
        for fpath in reporter.wrap(to_compute):
            counters = compute_channel_counters(fpath, z_frame_indices)
            save_channel_counters(counters_path(fpath), counters)
            channel_counters[meta.fov_id_of_file(fpath)] = counters

    # THRESHOLD (measure_tissue_thickness_test.ipynb's estimator): highest pixel
    # value among the n_background_frames lowest-mean frames across every FOV/z.
    frame_records = [
        (counter_mean(values, counts), fov_id, pos)
        for fov_id, counters in channel_counters.items()
        for pos, (values, counts) in enumerate(zip(counters["values_per_z"], counters["counts_per_z"]))
    ]
    frame_records.sort(key=lambda r: r[0])
    worst = frame_records[:n_background_frames]
    threshold = max(
        float(channel_counters[fov_id]["values_per_z"][pos].max()) for _, fov_id, pos in worst
    )

    # One frame read, just for its pixel dimensions (Counters discard shape).
    sample_frame = read_image_frames(files[0], [int(z_frame_indices[0][0])])[0]
    total_pixels = sample_frame.shape[0] * sample_frame.shape[1]

    rows = []
    for fov_id, counters in channel_counters.items():
        fractions = [
            tpc_from_counter(values, counts, threshold) / total_pixels
            for values, counts in zip(counters["values_per_z"], counters["counts_per_z"])
        ]
        rows.append({"fov_id": fov_id, "tissue_fraction": max(fractions)})

    results_df = pd.DataFrame(rows).sort_values("fov_id").reset_index(drop=True)
    print(f"[{label}] THRESHOLD={threshold:.1f}  tissue_fraction: "
          f"mean={results_df['tissue_fraction'].mean():.3f}  "
          f"median={results_df['tissue_fraction'].median():.3f}")
    return results_df, threshold


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_stats(stats_csv: Path) -> pd.DataFrame:
    """Load a previously saved stats CSV into a DataFrame."""
    return pd.read_csv(stats_csv)


def load_histogram(hist_npz: Path) -> Dict:
    """Load a previously saved histogram .npz into a plain dict."""
    data = np.load(str(hist_npz))
    return {k: data[k] for k in data.files}