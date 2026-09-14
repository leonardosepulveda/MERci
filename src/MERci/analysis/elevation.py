# MERci/analysis/elevation.py
"""
Per-pixel tissue-elevation mapping: how deep (in z) real tissue signal
extends, across the FOV grid -- a digital-elevation-model-style heatmap,
plus a z-sweep GIF of the downsampled, flat-field-corrected DAPI signal.

Promoted from ``notebooks/tests/tissue_thickness/01_elevation_heatmap.ipynb``'s
own investigation (see that notebook's own docstring/Review note for the
full algorithm rationale, the camera-orientation fix, and why the FFC field
is built from INTERIOR FOVs rather than boundary/exterior ones) and
``notebooks/tests/calculate_ffc/01_compare_ffc_methods.ipynb`` (why "min"
is the default per-FOV z-projection statistic). Now this repo's own
production tissue-thickness measurement
(``after_imaging/08_measure_tissue_thickness.ipynb``), replacing the
earlier per-FOV Counter/true-pixel-count scalar approach
(:mod:`MERci.analysis.fov`'s ``compute_channel_counters``/
``tpc_profile_from_counters``, still used elsewhere).

Pipeline (see ``08_measure_tissue_thickness.ipynb`` for the full notebook
wiring, including which steps offer a SLURM array option):

1. :func:`identify_boundary_fovs`     -- exterior vs. interior FOVs + grid indices
2. :func:`calculate_ffc`               -- FFC field from every interior FOV's
                                          own full-z-stack projection (default: min)
3. :func:`estimate_background_threshold` -- background/foreground intensity cutoff
4. :func:`compute_fov_elevation` (per FOV -- typically via a SLURM array, see
   ``MERci.acquisition.cluster_submit.build_fov_elevation_array_script``)
5. :func:`create_elevation_heatmap`    -- crop + stitch into one grid heatmap
6. :func:`create_gif`                   -- z-sweep GIF of the same per-FOV
                                          downsampled stacks, with a scale
                                          bar + z label
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ── Boundary/interior FOVs + grid indices ───────────────────────────────────

def identify_boundary_fovs(
    positions: Dict[int, Tuple[float, float]],
    step_size_um: float,
    connectivity: str = "8",
    tolerance_fraction: float = 0.25,
) -> Tuple[Set[int], Set[int], Dict[int, Tuple[int, int]]]:
    """
    Split *positions* into boundary (exterior-grid) vs. interior FOVs, plus
    each FOV's integer ``(row, col)`` grid index.

    Boundary FOVs = the exterior FOVs of the imaged grid (outer perimeter +
    any hole edges), via :func:`MERci.acquisition.positions.find_exterior_fovs`
    -- the same set :mod:`MERci.analysis.ffc`'s ``"exterior_grid"`` strategy
    uses. Used here only to bootstrap a rough background threshold (see
    :func:`estimate_background_threshold`) -- the real FFC field is built
    from INTERIOR FOVs instead (:func:`calculate_ffc`): a large fraction of
    "boundary" FOVs are not actually tissue-free, so a field built from them
    partly captures real anatomy instead of pure illumination/vignette (see
    the Review note in ``01_elevation_heatmap.ipynb``).

    ``grid_indices`` (row = y-based, col = x-based, rounded to the nearest
    integer step from the grid's own minimum x/y) is what
    :func:`stitch_by_grid` places tiles by.

    Parameters
    ----------
    positions          : {fov_id: (x, y)} stage coordinates (µm), scoped to
                         one round's own real imaged FOVs
    step_size_um       : grid step size (µm), e.g. ``ExperimentConfig.step_size_um``
    connectivity, tolerance_fraction : passed through to ``find_exterior_fovs``

    Returns
    -------
    boundary_fov_ids, interior_fov_ids, grid_indices
    """
    from MERci.acquisition.positions import find_exterior_fovs

    boundary_fov_ids = find_exterior_fovs(
        positions, step_size_um, connectivity=connectivity, tolerance_fraction=tolerance_fraction,
    )
    interior_fov_ids = set(positions) - boundary_fov_ids

    xy = np.array([positions[f] for f in positions])
    x0, y0 = xy[:, 0].min(), xy[:, 1].min()
    grid_indices = {
        fov_id: (int(round((y - y0) / step_size_um)), int(round((x - x0) / step_size_um)))
        for fov_id, (x, y) in positions.items()
    }
    return boundary_fov_ids, interior_fov_ids, grid_indices


# ── FFC field from interior FOVs' own z-projections ─────────────────────────

_VALID_PROJECTION_STATISTICS = ("min", "max", "median", "mean")


def project_stack(stack: np.ndarray, statistics: Iterable[str]) -> Dict[str, np.ndarray]:
    """
    Per-pixel projection(s) of a ``(n_z, H, W)`` z-stack -- one 2-D array per
    requested statistic in ``{"min", "max", "median", "mean"}``, computed
    from the SAME in-memory stack (one raw read serves every requested
    statistic -- shared by ``cli_compute_fov_projections.py`` and any local
    fallback loop).
    """
    unknown = set(statistics) - set(_VALID_PROJECTION_STATISTICS)
    if unknown:
        raise ValueError(f"Unknown statistic(s) {sorted(unknown)} -- must be a "
                          f"subset of {_VALID_PROJECTION_STATISTICS}.")
    out = {}
    if "max" in statistics:
        out["max"] = np.max(stack, axis=0)
    if "min" in statistics:
        out["min"] = np.min(stack, axis=0)
    if "mean" in statistics:
        out["mean"] = np.mean(stack, axis=0)
    if "median" in statistics:
        out["median"] = np.median(stack, axis=0)
    return out


def compute_fov_projection(
    fpath: Path,
    frame_indices: List[int],
    statistic: str = "min",
    orientation: Optional[dict] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> np.ndarray:
    """
    Read *fpath*'s full z-stack at *frame_indices* once and return its
    per-pixel *statistic* projection (see :func:`project_stack`), reoriented
    if *orientation* is given. One-statistic convenience wrapper around
    :func:`project_stack` for a local (non-SLURM) fallback loop --
    ``cli_compute_fov_elevation.py``-style SLURM tasks that need several
    statistics from one read should call :func:`project_stack` directly.
    """
    from MERci.acquisition.merlin_config import apply_microscope_orientation
    from MERci.common.io import read_image_frames

    stack = read_image_frames(fpath, frame_indices, frame_width, frame_height).astype(np.float32)
    img = project_stack(stack, [statistic])[statistic]
    if orientation:
        img = apply_microscope_orientation(img, **orientation)
    return img.astype(np.float32)


def build_ffc_field_from_projections(
    paths: List[Path],
    smooth_sigma_px: float = 50.0,
    normalize_percentile: float = 99.99,
    ffc_min_value: float = 0.10,
) -> Tuple[np.ndarray, dict]:
    """
    Mean-pool already-computed per-FOV projections (see :func:`calculate_ffc`)
    -> optional Gaussian smoothing -> percentile-normalize -> floor-clip.
    Same tail recipe as :func:`MERci.analysis.ffc.compute_ffc_field_for_color`,
    streamed one already-computed image at a time (never loading every one
    into memory at once -- a real interior-FOV population can number in the
    hundreds).

    Per ``notebooks/tests/calculate_ffc/01_compare_ffc_methods.ipynb``'s own
    real comparison: smoothing (at that sample size) barely changes an
    already-clean min-projection field, and can mask a genuinely
    contaminated one -- it does not automatically make the field "more
    correct". ``smooth_sigma_px=0`` skips smoothing entirely.
    """
    from scipy.ndimage import gaussian_filter

    total = None
    for p in paths:
        img = np.load(p).astype(np.float64)
        total = img if total is None else total + img
    field = (total / len(paths)).astype(np.float32)
    if smooth_sigma_px and smooth_sigma_px > 0:
        field = gaussian_filter(field, sigma=smooth_sigma_px)
    norm_value = np.percentile(field, normalize_percentile)
    if norm_value > 0:
        field = field / norm_value
    field = np.clip(field, ffc_min_value, None).astype(np.float32)
    meta = {
        "n_samples": len(paths), "smooth_sigma_px": smooth_sigma_px,
        "normalize_percentile": normalize_percentile, "ffc_min_value": ffc_min_value,
    }
    return field, meta


def calculate_ffc(
    fov_ids: List[int],
    projections_dir: Path,
    method: str = "min",
    smooth_sigma_px: float = 50.0,
    normalize_percentile: float = 99.99,
    ffc_min_value: float = 0.10,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Build the FFC field from every FOV in *fov_ids* (typically the
    experiment's INTERIOR FOVs -- see :func:`identify_boundary_fovs`) own
    full-z-stack *method* projection.

    Loads each FOV's already-cached per-pixel projection from
    ``<projections_dir>/fov<id>_<method>.npy`` -- the SAME file convention
    :func:`compute_fov_projection`/``cli_compute_fov_projections.py`` write,
    so a SLURM array job submitted via
    :func:`MERci.acquisition.cluster_submit.build_fov_projections_array_script`
    can be picked up here directly; there is no separate code path for the
    SLURM vs. local case.

    Parameters
    ----------
    fov_ids               : FOV ids to pool -- every one needs its own
                            cached projection already on disk to build the
                            field this call (see Returns for the pending case)
    projections_dir       : directory holding ``fov<id>_<method>.npy`` files
    method                 : which per-pixel z-projection statistic --
                            ``"min"`` (default, recommended -- see
                            ``notebooks/tests/calculate_ffc/01_compare_ffc_
                            methods.ipynb``'s own real comparison), ``"max"``,
                            ``"median"``, or ``"mean"``
    smooth_sigma_px, normalize_percentile, ffc_min_value : see
                            :func:`build_ffc_field_from_projections`

    Returns
    -------
    ``(field, meta)`` once every FOV in *fov_ids* has a cached projection
    ready (``meta`` includes ``"method"`` plus
    :func:`build_ffc_field_from_projections`'s own keys).

    ``(None, {"missing_fov_ids": [...]})`` if any FOV's projection is not
    yet on disk -- submit a SLURM array job for exactly those ids (or
    compute them locally via :func:`compute_fov_projection`) and call this
    again once they're ready.
    """
    if method not in _VALID_PROJECTION_STATISTICS:
        raise ValueError(f"method must be one of {_VALID_PROJECTION_STATISTICS}, got {method!r}")

    projections_dir = Path(projections_dir)
    paths = {fov_id: projections_dir / f"fov{fov_id:04d}_{method}.npy" for fov_id in fov_ids}
    missing = sorted(fov_id for fov_id, p in paths.items() if not p.exists())
    if missing:
        return None, {"missing_fov_ids": missing}

    field, meta = build_ffc_field_from_projections(
        [paths[f] for f in fov_ids], smooth_sigma_px, normalize_percentile, ffc_min_value,
    )
    meta["method"] = method
    return field, meta


# ── Background threshold ────────────────────────────────────────────────────

def ffc_correct_and_downsample(
    raw_frame: np.ndarray,
    ffc_field: np.ndarray,
    downsample_factor: int,
    orientation: Optional[dict] = None,
) -> np.ndarray:
    """
    Reorient (if *orientation* given -- a raw camera frame does not match
    the real stage layout otherwise, see
    ``MERci.acquisition.merlin_config.apply_microscope_orientation``) ->
    divide out *ffc_field* -> block-average downsample by
    *downsample_factor*. The one raw-frame operation shared by background-
    threshold estimation and :func:`compute_fov_elevation`.
    """
    from skimage.measure import block_reduce

    from MERci.acquisition.merlin_config import apply_microscope_orientation
    from MERci.analysis.ffc import apply_ffc

    frame = raw_frame
    if orientation:
        frame = apply_microscope_orientation(frame, **orientation)
    corrected = apply_ffc(frame, ffc_field)
    return block_reduce(corrected, (downsample_factor, downsample_factor), func=np.mean)


def estimate_background_threshold(
    images: Dict[int, np.ndarray],
    n_background_frames: int,
    percentile: float = 100.0,
) -> float:
    """
    Highest pixel value (at *percentile* -- default 100, the literal max)
    observed among the *n_background_frames* lowest-mean images in *images*
    -- the highest value that can plausibly occur as background noise,
    estimated only from frames confidently known to be background (same
    convention as :func:`MERci.analysis.fov.compute_tissue_fraction`).

    *images* should already be in whatever (FFC-corrected + downsampled)
    space thresholding will actually happen in.
    """
    means = {key: float(img.mean()) for key, img in images.items()}
    lowest_mean_keys = sorted(means, key=means.get)[:n_background_frames]
    return float(max(np.percentile(images[key], percentile) for key in lowest_mean_keys))


# ── Per-FOV elevation ────────────────────────────────────────────────────────

def compute_fov_elevation(
    fpath: Path,
    frame_indices: List[int],
    z_um_values: List[float],
    ffc_field: np.ndarray,
    threshold: float,
    downsample_factor: int,
    orientation: Optional[dict] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel tissue elevation for one FOV: for every z-plane (ascending),
    FFC-correct + downsample + threshold, overwriting ``M[i, j] = z_um``
    wherever the downsampled pixel is foreground -- ``M`` ends up holding
    each pixel's topmost foreground z (µm). ``M`` starts at 0, which
    unambiguously means "never foreground at any z" as long as
    *z_um_values* never includes exactly 0 (true of every real MERFISH
    z-grid, which starts at some positive step, e.g. 0.5 µm).

    Also returns the same z-stack's FFC-corrected, downsampled frames --
    reused directly by :func:`create_gif` (computed once here, never
    re-read).

    Parameters
    ----------
    fpath              : this FOV's image file (one full z-stack read)
    frame_indices      : 0-based frame indices, in ascending z order
    z_um_values        : z (µm) for each of *frame_indices*, same order
    ffc_field          : flat-field-correction map (already oriented -- see
                         :func:`calculate_ffc`/the Review note in
                         ``01_elevation_heatmap.ipynb`` on why)
    threshold          : background/foreground intensity cutoff, in the same
                         FFC-corrected + downsampled space
    downsample_factor  : block-average downsample factor
    orientation        : optional flip_horizontal/flip_vertical/transpose
                         dict (``load_microscope_orientation``'s output) --
                         applied to every raw frame before FFC
    frame_width, frame_height : only needed for .dax input

    Returns
    -------
    M        : float32 ``(h, w)`` elevation matrix, downsampled resolution
    ds_stack : float32 ``(n_z, h, w)`` FFC-corrected, downsampled z-stack
    """
    from MERci.common.io import read_image_frames

    M = None
    ds_stack = []
    for idx, z_um in zip(frame_indices, z_um_values):
        raw = read_image_frames(fpath, [int(idx)], frame_width, frame_height)[0]
        ds = ffc_correct_and_downsample(raw, ffc_field, downsample_factor, orientation).astype(np.float32)
        if M is None:
            M = np.zeros(ds.shape, dtype=np.float32)
        M[ds >= threshold] = z_um
        ds_stack.append(ds)
    return M, np.stack(ds_stack, axis=0)


# ── Stitching ────────────────────────────────────────────────────────────────

def center_crop(arr: np.ndarray, crop_px: int) -> np.ndarray:
    """Symmetric crop of *crop_px* pixels from every edge (0 = no-op)."""
    if crop_px == 0:
        return arr
    return arr[crop_px:-crop_px, crop_px:-crop_px]


def stitch_by_grid(
    tiles: Dict[int, np.ndarray],
    grid_indices: Dict[int, Tuple[int, int]],
    r0: int,
    c0: int,
    n_rows: int,
    n_cols: int,
    crop_px: int,
    fill: float = np.nan,
) -> np.ndarray:
    """
    Center-crop every tile in *tiles* to its non-overlap footprint (see
    :func:`MERci.analysis.ffc.compute_mosaic_crop_px`), then place it into
    an ``(n_rows * tile_h, n_cols * tile_w)`` canvas indexed by
    *grid_indices* (window starting at ``(r0, c0)``) -- missing grid cells
    (no real FOV there, e.g. a hole, or outside the requested window) are
    left as *fill*. Shared by :func:`create_elevation_heatmap` and
    :func:`create_gif` so both stitch identically.
    """
    cropped = {f: center_crop(t, crop_px) for f, t in tiles.items()}
    tile_h, tile_w = next(iter(cropped.values())).shape
    canvas = np.full((n_rows * tile_h, n_cols * tile_w), fill, dtype=np.float32)
    for fov_id, tile in cropped.items():
        r, c = grid_indices[fov_id]
        rr, cc = r - r0, c - c0
        if not (0 <= rr < n_rows and 0 <= cc < n_cols):
            continue
        canvas[rr * tile_h:(rr + 1) * tile_h, cc * tile_w:(cc + 1) * tile_w] = tile
    return canvas


def _resolve_grid_window(
    grid_indices: Dict[int, Tuple[int, int]],
    r0: int,
    c0: int,
    n_rows: Optional[int],
    n_cols: Optional[int],
) -> Tuple[int, int]:
    """Fill in n_rows/n_cols from *grid_indices*' own full extent when not
    given explicitly (production, full-grid use) -- pass them explicitly
    (with r0/c0) to stitch a smaller representative window instead."""
    if n_rows is not None and n_cols is not None:
        return n_rows, n_cols
    rows = [r for r, _ in grid_indices.values()]
    cols = [c for _, c in grid_indices.values()]
    if n_rows is None:
        n_rows = max(rows) - r0 + 1
    if n_cols is None:
        n_cols = max(cols) - c0 + 1
    return n_rows, n_cols


def create_elevation_heatmap(
    elevation_matrices: Dict[int, np.ndarray],
    grid_indices: Dict[int, Tuple[int, int]],
    config,                                     # ExperimentConfig
    downsample_factor: int = 16,
    r0: int = 0,
    c0: int = 0,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
) -> np.ndarray:
    """
    Crop + stitch every FOV's own elevation matrix
    (:func:`compute_fov_elevation`'s own ``M``) into one grid-indexed
    heatmap. Defaults to the FULL extent of *grid_indices* (production use
    over the whole real FOV grid) -- pass a smaller ``(r0, c0, n_rows,
    n_cols)`` window for a quick representative-block preview instead (see
    ``01_elevation_heatmap.ipynb``'s own scope note on why a full-grid run
    needs a SLURM array job).
    """
    from MERci.analysis.ffc import compute_mosaic_crop_px

    crop_px = compute_mosaic_crop_px(config) // downsample_factor
    n_rows, n_cols = _resolve_grid_window(grid_indices, r0, c0, n_rows, n_cols)
    return stitch_by_grid(elevation_matrices, grid_indices, r0, c0, n_rows, n_cols, crop_px, fill=np.nan)


# ── z-sweep GIF ──────────────────────────────────────────────────────────────

def _to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = (arr.astype(np.float64) - vmin) / max(vmax - vmin, 1e-9) * 255
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _draw_scale_bar(draw, canvas_width: int, canvas_height: int, bar_px: int, label: str, fill: int = 255) -> None:
    """Bottom-left physical scale bar: a horizontal line *bar_px* pixels
    long, with *label* (e.g. "1 mm") drawn just above its left end -- same
    PIL-default-font convention as the z label (no font-file dependency)."""
    from PIL import ImageFont

    font = ImageFont.load_default(size=max(14, canvas_width // 40))
    margin = max(10, canvas_width // 50)
    x0 = margin
    y0 = canvas_height - margin
    x1 = min(x0 + bar_px, canvas_width - margin)
    draw.line([(x0, y0), (x1, y0)], fill=fill, width=max(2, canvas_height // 200))
    draw.text((x0, y0 - font.size - 4), label, fill=fill, font=font)


def create_gif(
    stack_paths: Dict[int, Path],
    z_um_values: List[float],
    grid_indices: Dict[int, Tuple[int, int]],
    config,                                     # ExperimentConfig
    output_path: Path,
    downsample_factor: int = 16,
    r0: int = 0,
    c0: int = 0,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    z_stride: int = 1,
    frame_duration_ms: int = 300,
    scalebar_um: float = 1000.0,
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
) -> Path:
    """
    Assemble a z-sweep GIF of the same FFC-corrected, downsampled z-stacks
    :func:`compute_fov_elevation` already produced (one ``.npy`` path per
    FOV, shape ``(n_z, h, w)`` -- reused directly, no re-read of raw data).
    Each frame is the whole grid (or a smaller window -- see
    :func:`create_elevation_heatmap`) stitched at its native downsampled
    resolution (no further resize), with a shared intensity scale across
    every frame (so brightness changes reflect real signal fading, not
    per-frame auto-contrast), a per-frame ``"z = <value> um"`` label, and a
    physical scale bar (*scalebar_um*, default 1000 -> "1 mm").

    Streams one z-plane at a time from each FOV's own memory-mapped
    ``.npy`` (never loading a whole per-FOV stack, let alone every FOV's
    stack, into memory at once) -- needed at full-grid scale, where every
    FOV's whole stack together can run into the tens of GB.

    Parameters
    ----------
    stack_paths        : ``{fov_id: path}`` to that FOV's ``(n_z, h, w)``
                         float32 ``.npy`` z-stack -- e.g.
                         :func:`compute_fov_elevation`'s own ``ds_stack``,
                         saved via ``np.save`` (NOT ``np.savez`` -- memory-
                         mapping needs a plain ``.npy``)
    z_um_values        : z (µm) for each of the stack's ``n_z`` planes
    grid_indices, config, downsample_factor, r0, c0, n_rows, n_cols : same
                         as :func:`create_elevation_heatmap`
    output_path        : where to save the finished GIF
    z_stride           : take every Nth z-plane (default 1 = every frame)
    frame_duration_ms  : GIF frame duration
    scalebar_um        : physical scale-bar length in µm (default 1000 =
                         1 mm); label is ``"<value> mm"`` for >=1000 µm,
                         else ``"<value> um"``
    percentile_clip    : ``(lo_pct, hi_pct)`` shared display-intensity
                         scale, estimated from one representative (middle)
                         z-plane pooled across every FOV -- not the whole
                         stack, to avoid reading everything twice at
                         full-grid scale

    Returns
    -------
    output_path
    """
    from PIL import Image, ImageDraw, ImageFont

    from MERci.analysis.ffc import compute_mosaic_crop_px
    from MERci.progress_display import ProgressReporter

    crop_px = compute_mosaic_crop_px(config) // downsample_factor
    n_rows, n_cols = _resolve_grid_window(grid_indices, r0, c0, n_rows, n_cols)
    fov_ids = [f for f in stack_paths if f in grid_indices]
    if not fov_ids:
        raise ValueError("No FOV in stack_paths has a matching entry in grid_indices")

    n_z = len(z_um_values)
    z_positions = list(range(0, n_z, z_stride))

    # Shared display scale from one representative (middle) z-plane per FOV
    # only -- cheap, and representative of the stack's overall brightness
    # range without reading every plane of every FOV twice.
    mid_z = n_z // 2
    mid_pixels = np.concatenate([
        np.asarray(np.load(stack_paths[f], mmap_mode="r")[mid_z]).ravel() for f in fov_ids
    ])
    vmin, vmax = np.percentile(mid_pixels, [percentile_clip[0], percentile_clip[1]])
    del mid_pixels

    pixel_size_ds_um = config.pixel_size_um * downsample_factor
    bar_px = max(1, round(scalebar_um / pixel_size_ds_um))
    bar_label = f"{scalebar_um / 1000:.3g} mm" if scalebar_um >= 1000 else f"{scalebar_um:.0f} um"

    frames = []
    reporter = ProgressReporter(total=len(z_positions), label="Assembling GIF frames")
    for z_pos in reporter.wrap(z_positions):
        tiles = {
            f: _to_uint8(np.load(stack_paths[f], mmap_mode="r")[z_pos], vmin, vmax)
            for f in fov_ids
        }
        canvas = stitch_by_grid(tiles, grid_indices, r0, c0, n_rows, n_cols, crop_px, fill=0)
        img = Image.fromarray(canvas.astype(np.uint8), mode="L")   # native canvas resolution -- no resize
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=max(14, img.width // 40))
        draw.text((8, 8), f"z = {z_um_values[z_pos]:.1f} um", fill=255, font=font)
        _draw_scale_bar(draw, img.width, img.height, bar_px, bar_label)
        frames.append(img)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=frame_duration_ms, loop=0)
    log.info("GIF saved: %s  (%d frame(s), %d x %d px)",
              output_path, len(frames), frames[0].width, frames[0].height)
    return output_path
