# MERci/analysis/elevation.py
"""
Per-pixel tissue elevation: how deep (in z) tissue signal extends across
the FOV grid, as an elevation-map heatmap, plus z-sweep GIF/MP4s of the
downsampled, flat-field-corrected DAPI signal.

Production tissue-thickness measurement for
``after_imaging/08_measure_tissue_thickness.ipynb`` (replacing the per-FOV
Counter/true-pixel-count approach in :mod:`MERci.analysis.fov`, which other
notebooks still use). Algorithm rationale, including why the FFC field
comes from INTERIOR FOVs and why "min" is the default z-projection, is in
``notebooks/tests/tissue_thickness/01_elevation_heatmap.ipynb`` and
``notebooks/tests/calculate_ffc/01_compare_ffc_methods.ipynb``.

Pipeline (notebook 08 shows the wiring and SLURM options):

1. :func:`identify_boundary_fovs`        -- exterior vs. interior FOVs + grid indices
2. :func:`calculate_ffc`                 -- FFC field from the interior FOVs'
                                            full-z projections (default: min)
3. :func:`estimate_background_threshold` -- background/foreground cutoff
4. :func:`compute_fov_elevation`         -- per FOV, usually as a SLURM array
   (``MERci.acquisition.cluster_submit.build_fov_elevation_array_script``)
5. :func:`create_elevation_heatmap`      -- crop + stitch into one heatmap
6. :func:`create_gif`/:func:`create_movie` -- z-sweep with scale bar and z label
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
    from MERci.acquisition.configs import apply_microscope_orientation
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
    from .ffc import mean_field_to_ffc

    return mean_field_to_ffc((np.load(p) for p in paths),
                             smooth_sigma_px, normalize_percentile, ffc_min_value)


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
    ``MERci.acquisition.configs.apply_microscope_orientation``) ->
    divide out *ffc_field* -> block-average downsample by
    *downsample_factor*. The one raw-frame operation shared by background-
    threshold estimation and :func:`compute_fov_elevation`.
    """
    from skimage.measure import block_reduce

    from MERci.acquisition.configs import apply_microscope_orientation
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
    from MERci.common.io import iter_image_frames

    M = None
    ds_stack = []
    frames = iter_image_frames(fpath, [int(i) for i in frame_indices],
                               frame_width=frame_width, frame_height=frame_height)
    for (_, raw), z_um in zip(frames, z_um_values):
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
    long, with *label* (e.g. "1 mm") drawn above its left end, clear of the
    line by 1% of the canvas height -- same PIL-default-font convention as
    the z label (no font-file dependency)."""
    from PIL import ImageFont

    font = ImageFont.load_default(size=max(14, canvas_width // 40))
    margin = max(10, canvas_width // 50)
    label_gap = max(4, round(canvas_height * 0.01))
    x0 = margin
    y0 = canvas_height - margin
    x1 = min(x0 + bar_px, canvas_width - margin)
    draw.line([(x0, y0), (x1, y0)], fill=fill, width=max(2, canvas_height // 200))
    draw.text((x0, y0 - font.size - label_gap), label, fill=fill, font=font)


def _scalebar_px_and_label(config, downsample_factor: int, scalebar_um: float) -> Tuple[int, str]:
    """Physical scale-bar length in downsampled pixels + its label -- shared
    by :func:`create_gif` and :func:`create_z_mosaic`. Label is
    ``"<value> mm"`` for *scalebar_um* >= 1000, else ``"<value> um"``."""
    pixel_size_ds_um = config.pixel_size_um * downsample_factor
    bar_px = max(1, round(scalebar_um / pixel_size_ds_um))
    bar_label = f"{scalebar_um / 1000:.3g} mm" if scalebar_um >= 1000 else f"{scalebar_um:.0f} um"
    return bar_px, bar_label


def _render_stitched_frame(
    stack_paths: Dict[int, Path],
    fov_ids: List[int],
    z_pos: int,
    grid_indices: Dict[int, Tuple[int, int]],
    r0: int, c0: int, n_rows: int, n_cols: int, crop_px: int,
    vmin: float, vmax: float,
    z_um_value: float,
    bar_px: int, bar_label: str,
):
    """
    Stitch one z-plane (index *z_pos*, read lazily via memory-map from each
    FOV's own ``.npy`` stack) into one grid-indexed canvas, then annotate it
    with the ``"z = <value> um"`` label + physical scale bar -- the single-
    frame building block shared by :func:`create_gif` (one call per z-plane)
    and :func:`create_z_mosaic` (one call, at one chosen z).

    Returns a PIL ``"L"`` (grayscale) Image at the canvas's native
    resolution (no resize).
    """
    from PIL import Image, ImageDraw, ImageFont

    tiles = {f: _to_uint8(np.load(stack_paths[f], mmap_mode="r")[z_pos], vmin, vmax) for f in fov_ids}
    canvas = stitch_by_grid(tiles, grid_indices, r0, c0, n_rows, n_cols, crop_px, fill=0)
    img = Image.fromarray(canvas.astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(14, img.width // 40))

    # Lower-right corner, 1% of each dimension clear of the image edge --
    # measured from the glyphs' own ink extent (textbbox at origin), not the
    # font's nominal line box, since PIL pads the top of that box by several
    # px and drawing from it would run the text past the bottom edge.
    z_label = f"z = {z_um_value:.1f} um"
    margin_x = max(4, round(img.width * 0.01))
    margin_y = max(4, round(img.height * 0.01))
    label_bbox = draw.textbbox((0, 0), z_label, font=font)
    draw.text((img.width - margin_x - label_bbox[2], img.height - margin_y - label_bbox[3]),
              z_label, fill=255, font=font)

    _draw_scale_bar(draw, img.width, img.height, bar_px, bar_label)
    return img


def _render_setup(stack_paths, grid_indices, config, downsample_factor, r0, c0,
                  n_rows, n_cols, scalebar_um, percentile_clip, scale_z_pos):
    """
    Grid window, crop size, display scale (percentiles over every FOV's
    plane *scale_z_pos*) and scale bar shared by every stitched render.
    Returns ``(fov_ids, n_rows, n_cols, crop_px, vmin, vmax, bar_px, bar_label)``.
    """
    from MERci.analysis.ffc import compute_mosaic_crop_px

    crop_px = compute_mosaic_crop_px(config) // downsample_factor
    n_rows, n_cols = _resolve_grid_window(grid_indices, r0, c0, n_rows, n_cols)
    fov_ids = [f for f in stack_paths if f in grid_indices]
    if not fov_ids:
        raise ValueError("No FOV in stack_paths has a matching entry in grid_indices")
    pixels = np.concatenate([
        np.asarray(np.load(stack_paths[f], mmap_mode="r")[scale_z_pos]).ravel() for f in fov_ids
    ])
    vmin, vmax = np.percentile(pixels, [percentile_clip[0], percentile_clip[1]])
    bar_px, bar_label = _scalebar_px_and_label(config, downsample_factor, scalebar_um)
    return fov_ids, n_rows, n_cols, crop_px, vmin, vmax, bar_px, bar_label


def _cached_frame(cache_path: Optional[Path], render):
    """Load *cache_path* if it exists, else ``render()`` and save it there."""
    from PIL import Image

    if cache_path is not None and cache_path.exists():
        frame = Image.open(cache_path)
        frame.load()
        return frame
    frame = render()
    if cache_path is not None:
        frame.save(cache_path)
    return frame


def _prepare_sweep_render(
    stack_paths: Dict[int, Path],
    z_um_values: List[float],
    grid_indices: Dict[int, Tuple[int, int]],
    config,
    downsample_factor: int,
    r0: int, c0: int, n_rows: Optional[int], n_cols: Optional[int],
    z_stride: int,
    scalebar_um: float,
    percentile_clip: Tuple[float, float],
):
    """
    Shared setup for :func:`create_gif`/:func:`create_movie`: resolve the
    grid window and crop size, the shared display-intensity scale (from one
    representative z-plane, see those functions' own docstrings), the scale
    bar, and the z-plane indices to render.
    """
    n_z = len(z_um_values)
    z_positions = list(range(0, n_z, z_stride))
    setup = _render_setup(stack_paths, grid_indices, config, downsample_factor, r0, c0,
                          n_rows, n_cols, scalebar_um, percentile_clip, n_z // 2)
    return (*setup, z_positions)


def _stitched_frames(
    stack_paths, fov_ids, z_positions, grid_indices, r0, c0, n_rows, n_cols, crop_px,
    vmin, vmax, z_um_values, bar_px, bar_label,
    frame_cache_dir: Optional[Path] = None,
    progress_label: str = "Assembling frames",
):
    """
    Yield one annotated, stitched frame per entry in *z_positions* (see
    :func:`_render_stitched_frame`), optionally caching each to
    *frame_cache_dir* as ``z<index>.png`` and reloading instead of
    re-rendering on a later call -- shared by :func:`create_gif` and
    :func:`create_movie` so a GIF and a movie of the same sweep (same cache
    directory) render every frame's expensive stitching only once between
    them, and so a crash during either one's final encode/save step (after
    this generator has already finished) doesn't force a redo.
    """
    from MERci.progress_display import ProgressReporter

    if frame_cache_dir is not None:
        frame_cache_dir = Path(frame_cache_dir)
        frame_cache_dir.mkdir(parents=True, exist_ok=True)

    reporter = ProgressReporter(total=len(z_positions), label=progress_label)
    for z_pos in reporter.wrap(z_positions):
        cache_path = frame_cache_dir / f"z{z_pos:04d}.png" if frame_cache_dir is not None else None
        yield _cached_frame(cache_path, lambda: _render_stitched_frame(
            stack_paths, fov_ids, z_pos, grid_indices, r0, c0, n_rows, n_cols, crop_px,
            vmin, vmax, z_um_values[z_pos], bar_px, bar_label,
        ))


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
    frame_cache_dir: Optional[Path] = None,
) -> Path:
    """
    Z-sweep GIF of the FFC-corrected, downsampled per-FOV stacks
    :func:`compute_fov_elevation` produced (one ``(n_z, h, w)`` ``.npy`` per FOV;
    raw data is not re-read). Each frame is the grid (or a window, as in
    :func:`create_elevation_heatmap`) stitched at the downsampled resolution,
    with one intensity scale for all frames (so fading is real signal), a
    ``"z = <value> um"`` label and a scale bar.

    Reads one z-plane at a time from memory-mapped ``.npy`` files, since all
    stacks together can reach tens of GB.

    For slides prefer :func:`create_movie`: PowerPoint treats a GIF as a
    picture, and autoplay depends on the version.

    Parameters
    ----------
    stack_paths        : ``{fov_id: path}`` to a float32 ``(n_z, h, w)``
                         ``.npy`` (saved with ``np.save``, not ``np.savez``,
                         so it can be memory-mapped)
    z_um_values        : z (µm) of each of the ``n_z`` planes
    grid_indices, config, downsample_factor, r0, c0, n_rows, n_cols : as in
                         :func:`create_elevation_heatmap`
    output_path        : where to save the GIF
    z_stride           : take every Nth z-plane (default 1)
    frame_duration_ms  : GIF frame duration
    scalebar_um        : scale-bar length (µm, default 1000); labelled in mm
                         from 1000 µm up
    percentile_clip    : ``(lo_pct, hi_pct)`` of the shared intensity scale,
                         taken from the middle z-plane of every FOV
    frame_cache_dir    : if given, each rendered frame is saved as
                         ``z<index>.png`` and reused on later calls (so a crash
                         while encoding doesn't redo the rendering). Keyed on
                         z index only: clear it after changing any display
                         parameter or the grid window. Can be shared with a
                         :func:`create_movie` call over the same sweep.

    Returns
    -------
    output_path
    """
    fov_ids, n_rows, n_cols, crop_px, vmin, vmax, bar_px, bar_label, z_positions = _prepare_sweep_render(
        stack_paths, z_um_values, grid_indices, config, downsample_factor,
        r0, c0, n_rows, n_cols, z_stride, scalebar_um, percentile_clip,
    )

    frames = list(_stitched_frames(
        stack_paths, fov_ids, z_positions, grid_indices, r0, c0, n_rows, n_cols, crop_px,
        vmin, vmax, z_um_values, bar_px, bar_label,
        frame_cache_dir=frame_cache_dir, progress_label="Assembling GIF frames",
    ))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=frame_duration_ms, loop=0)
    log.info("GIF saved: %s  (%d frame(s), %d x %d px)",
              output_path, len(frames), frames[0].width, frames[0].height)
    return output_path


def create_movie(
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
    fps: Optional[float] = None,
    frame_duration_ms: int = 300,
    scalebar_um: float = 1000.0,
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
    frame_cache_dir: Optional[Path] = None,
) -> Path:
    """
    MP4 (H.264 video, ``yuv420p`` pixel format) version of the same z-sweep
    :func:`create_gif` produces -- identical cropped/annotated/shared-scale
    frames (see that function's own docstring), written via ``imageio``'s
    ffmpeg backend instead of PIL's GIF encoder. H.264 + ``yuv420p`` in an
    ``.mp4`` container is the combination PowerPoint (Windows and Mac)
    embeds as a real, "Insert > Video"-able object -- unlike a GIF, which
    PowerPoint only ever treats as a picture. Silent (no audio track).

    Pass the *same* `frame_cache_dir` used for a `create_gif` call over the
    same sweep (same `z_stride`/`downsample_factor`/etc.) to reuse its
    already-rendered frames instead of re-stitching them.

    Parameters
    ----------
    fps                : frames per second; defaults to
                         ``1000 / frame_duration_ms`` so the movie plays at
                         the same speed as a `create_gif` call with the same
                         `frame_duration_ms`, unless overridden
    (all other parameters : same as :func:`create_gif`)

    Returns
    -------
    output_path
    """
    import imageio

    fov_ids, n_rows, n_cols, crop_px, vmin, vmax, bar_px, bar_label, z_positions = _prepare_sweep_render(
        stack_paths, z_um_values, grid_indices, config, downsample_factor,
        r0, c0, n_rows, n_cols, z_stride, scalebar_um, percentile_clip,
    )
    if fps is None:
        fps = 1000.0 / frame_duration_ms

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p")
    try:
        for frame in _stitched_frames(
            stack_paths, fov_ids, z_positions, grid_indices, r0, c0, n_rows, n_cols, crop_px,
            vmin, vmax, z_um_values, bar_px, bar_label,
            frame_cache_dir=frame_cache_dir, progress_label="Assembling movie frames",
        ):
            writer.append_data(np.asarray(frame.convert("RGB")))
            n_frames += 1
    finally:
        writer.close()
    log.info("Movie saved: %s  (%d frame(s) @ %.1f fps)", output_path, n_frames, fps)
    return output_path


# libx264's own peak (steady-state, not leaked) memory use for a
# rawvideo-piped encode scales with frame resolution -- empirically, during
# development: 1920x1080 plateaus around 0.6 GB, 4096x2944 (12 MP) around
# 2.8 GB; a single acquisition's own full-grid frame (6512x4560, 30 MP) is
# already close to the edge of what a modest interactive session's memory
# allows, and this function's own combined (two full grids wide) frame before
# any resizing is roughly double that again (60 MP) -- reliably exceeded a
# 4 GB SLURM cgroup and got the ffmpeg child OOM-killed partway through the
# encode (confirmed via dmesg) well before finishing. Cap the ENCODED frame's
# own pixel count so its steady-state memory stays modest regardless of how
# many FOVs either side's own grid has.
_DEFAULT_MAX_OUTPUT_PIXELS = 12_000_000   # ~2.8 GB peak, verified empirically


def _fit_frame_for_movie_encode(img, max_pixels=_DEFAULT_MAX_OUTPUT_PIXELS):
    """
    Downscale *img* (if needed) so its own pixel count stays under
    *max_pixels* -- see :data:`_DEFAULT_MAX_OUTPUT_PIXELS`'s own note on why
    -- then pad up to a multiple of 16 px (libx264's macroblock size);
    otherwise imageio_ffmpeg inserts its own auto ``-vf scale`` filter for a
    non-aligned size, which destabilized the encode at this function's own
    frame sizes just as reliably as skipping the resize entirely.
    """
    from PIL import Image

    if img.width * img.height > max_pixels:
        scale = (max_pixels / (img.width * img.height)) ** 0.5
        img = img.resize((max(16, round(img.width * scale)), max(16, round(img.height * scale))), Image.LANCZOS)

    width  = -(-img.width // 16) * 16
    height = -(-img.height // 16) * 16
    if (width, height) != img.size:
        padded = Image.new(img.mode, (width, height))
        padded.paste(img, (0, 0))
        img = padded
    return img


def create_paired_movie(
    stack_paths_a: Dict[int, Path],
    z_um_values_a: List[float],
    grid_indices_a: Dict[int, Tuple[int, int]],
    config_a,                                    # ExperimentConfig
    stack_paths_b: Dict[int, Path],
    z_um_values_b: List[float],
    grid_indices_b: Dict[int, Tuple[int, int]],
    config_b,                                    # ExperimentConfig
    target_z_um_values: List[float],
    output_path: Path,
    downsample_factor: int = 16,
    fps: Optional[float] = None,
    frame_duration_ms: int = 300,
    scalebar_um: float = 1000.0,
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
    frame_cache_dir_a: Optional[Path] = None,
    frame_cache_dir_b: Optional[Path] = None,
    max_output_pixels: int = _DEFAULT_MAX_OUTPUT_PIXELS,
) -> Path:
    """
    Side-by-side MP4 of two z-sweeps (rotated 90° CCW, panel *a* on the left),
    e.g. two sibling acquisitions of one sample (see
    :func:`MERci.common.experiment_info.resolve_sample_identity`), matched by
    physical depth. For each value in *target_z_um_values* each side shows its
    own nearest z-plane (as :func:`create_z_mosaic`), so different z steps or
    offsets never let the panels drift apart. The caller picks the depths
    (typically the overlap of both z ranges).

    Each side keeps its own intensity scale, fixed from its middle frame (two
    acquisitions' intensities aren't comparable). Both share one physical
    scale bar, so it has the same length in µm even if pixel sizes differ.

    Parameters
    ----------
    stack_paths_a/b, z_um_values_a/b, grid_indices_a/b, config_a/b : per-side
        versions of :func:`create_movie`'s parameters
    target_z_um_values : depths (µm), one output frame each
    frame_cache_dir_a/b : per-side frame cache (keyed on that side's nearest
        z index), as :func:`create_movie`'s *frame_cache_dir*
    max_output_pixels : downscale the combined frame below this pixel count
        (see :data:`_DEFAULT_MAX_OUTPUT_PIXELS`); raise it for sharper output
        if memory allows
    (other parameters : as in :func:`create_movie`)

    Returns
    -------
    output_path
    """
    import imageio
    from PIL import Image

    from MERci.progress_display import ProgressReporter

    def _prepare_side(stack_paths, z_um_values, grid_indices, config):
        keys = ("fov_ids", "n_rows", "n_cols", "crop_px", "vmin", "vmax", "bar_px", "bar_label")
        setup = _render_setup(stack_paths, grid_indices, config, downsample_factor, 0, 0,
                              None, None, scalebar_um, percentile_clip, len(z_um_values) // 2)
        return dict(zip(keys, setup), z_arr=np.asarray(z_um_values, dtype=float))

    side_a = _prepare_side(stack_paths_a, z_um_values_a, grid_indices_a, config_a)
    side_b = _prepare_side(stack_paths_b, z_um_values_b, grid_indices_b, config_b)

    def _side_frame(stack_paths, grid_indices, side, target_z, frame_cache_dir):
        z_pos = int(np.argmin(np.abs(side["z_arr"] - target_z)))
        cache_path = Path(frame_cache_dir) / f"z{z_pos:04d}.png" if frame_cache_dir is not None else None
        return _cached_frame(cache_path, lambda: _render_stitched_frame(
            stack_paths, side["fov_ids"], z_pos, grid_indices,
            0, 0, side["n_rows"], side["n_cols"], side["crop_px"],
            side["vmin"], side["vmax"], side["z_arr"][z_pos], side["bar_px"], side["bar_label"],
        ))

    if frame_cache_dir_a is not None:
        Path(frame_cache_dir_a).mkdir(parents=True, exist_ok=True)
    if frame_cache_dir_b is not None:
        Path(frame_cache_dir_b).mkdir(parents=True, exist_ok=True)
    if fps is None:
        fps = 1000.0 / frame_duration_ms

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    reporter = ProgressReporter(total=len(target_z_um_values), label="Assembling paired movie frames")
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p")
    try:
        for target_z in reporter.wrap(target_z_um_values):
            img_a = _side_frame(stack_paths_a, grid_indices_a, side_a, target_z, frame_cache_dir_a).rotate(90, expand=True)
            img_b = _side_frame(stack_paths_b, grid_indices_b, side_b, target_z, frame_cache_dir_b).rotate(90, expand=True)

            combined = Image.new("L", (img_a.width + img_b.width, max(img_a.height, img_b.height)))
            combined.paste(img_a, (0, 0))
            combined.paste(img_b, (img_a.width, 0))
            combined = _fit_frame_for_movie_encode(combined, max_output_pixels)
            writer.append_data(np.asarray(combined.convert("RGB")))
            n_frames += 1
    finally:
        writer.close()
    log.info("Paired movie saved: %s  (%d frame(s) @ %.1f fps)", output_path, n_frames, fps)
    return output_path


def create_z_mosaic(
    stack_paths: Dict[int, Path],
    z_um_values: List[float],
    grid_indices: Dict[int, Tuple[int, int]],
    config,                                     # ExperimentConfig
    output_path: Path,
    z_um: float,
    downsample_factor: int = 16,
    r0: int = 0,
    c0: int = 0,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    scalebar_um: float = 1000.0,
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
) -> Path:
    """
    Static single-z-plane sibling of :func:`create_gif`: one stitched,
    FFC-corrected, downsampled mosaic at the z-plane closest to *z_um* --
    same shared-scale/z-label/scale-bar convention, same ``stack_paths``
    input (:func:`compute_fov_elevation`'s own ``ds_stack``), saved as a
    single PNG instead of a GIF. Useful as a standalone figure at one
    representative depth, without opening/paging through the full sweep.

    Parameters
    ----------
    z_um  : target depth (µm) -- the closest available z-plane in
            *z_um_values* is used (exact match not required)
    (all other parameters same as :func:`create_gif`, minus the animation-
    only ones)

    Returns
    -------
    output_path
    """
    z_pos = int(np.argmin(np.abs(np.asarray(z_um_values, dtype=float) - z_um)))
    fov_ids, n_rows, n_cols, crop_px, vmin, vmax, bar_px, bar_label = _render_setup(
        stack_paths, grid_indices, config, downsample_factor, r0, c0, n_rows, n_cols,
        scalebar_um, percentile_clip, z_pos,
    )
    img = _render_stitched_frame(
        stack_paths, fov_ids, z_pos, grid_indices, r0, c0, n_rows, n_cols, crop_px,
        vmin, vmax, z_um_values[z_pos], bar_px, bar_label,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    log.info("Single-z mosaic saved: %s  (z=%.1f um, %d x %d px)",
              output_path, z_um_values[z_pos], img.width, img.height)
    return output_path
