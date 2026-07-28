# MERci/analysis/round.py
"""
Round-level analysis: assemble a spatial mosaic from per-FOV thumbnails.

The mosaic is a single large image in which each thumbnail is placed at a
pixel position proportional to the FOV's stage coordinates.  The scale
(pixels per stage unit) is estimated automatically from the median
nearest-neighbour distance between stage positions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple  # noqa: F401 (Optional/List used in annotations)

import numpy as np

log = logging.getLogger(__name__)


def create_mosaic(
    thumbnails: Dict[int, np.ndarray],
    positions: Dict[int, Tuple[float, float]],
    output_path: Path,
    thumbnail_size: Tuple[int, int] = (200, 200),
    padding: int = 4,
    background: int = 0,
    pixels_per_unit: Optional[float] = None,
    flip_y: bool = False,
    labels: Optional[Dict[int, str]] = None,
    label_color: int = 255,
    highlight_fov_ids: Optional[Set[int]] = None,
    highlight_color: int = 255,
    highlight_width: int = 3,
    return_tile_bboxes: bool = False,
):
    """
    Assemble a mosaic image from per-FOV thumbnails placed at stage coordinates.

    Layout
    ------
    Each thumbnail cell is placed at a canvas position proportional to its
    stage (x, y) coordinate.  The scale factor ``pixels_per_unit`` maps from
    stage units (e.g. µm) to canvas pixels.  If not supplied it is estimated
    from the median nearest-neighbour distance between all FOV positions so
    that adjacent thumbnails end up side-by-side with ``padding`` pixels gap.

    Parameters
    ----------
    thumbnails      : {fov_id: 2-D uint8 array (H, W)}
    positions       : {fov_id: (x, y)} stage coordinates (any consistent unit)
    output_path     : where to save the finished mosaic PNG
    thumbnail_size  : (width, height) of each thumbnail cell in the canvas
    padding         : pixel gap around each thumbnail (also used for border)
    background      : canvas fill value for empty pixels (0–255)
    pixels_per_unit : spatial scale; auto-estimated when ``None``
    flip_y          : if True, mirror the y-axis so that stage +y points up
                      in the image (depends on your microscope convention)
    labels          : optional {fov_id: text} drawn in the top-left corner of
                      that FOV's tile (e.g. a z value) using PIL's built-in
                      default font -- no font file dependency. FOVs not in
                      this dict (or omitted entirely) get no label.
    label_color     : pixel value (0-255) for the label text
    highlight_fov_ids : optional set of FOV ids to draw a rectangle border
                      around (e.g. to flag FOVs classified as a different
                      category than the rest, for a visual sanity check).
                      Stays single-channel grayscale like the rest of the
                      mosaic (source data here is grayscale, not RGB) --
                      the border is a pixel intensity, not a color.
    highlight_color : pixel value (0-255) for the highlight border (default
                      255 = white, for max contrast against typically dim
                      background/tissue thumbnails)
    highlight_width : border thickness in pixels
    return_tile_bboxes : if True, also return {fov_id: (x0, y0, x1, y1)}
                      pixel bounding boxes for every placed tile -- lets a
                      caller draw its own overlay (e.g. a matplotlib
                      ``Rectangle`` patch on top of an ``imshow`` of this
                      canvas) instead of a border baked into the raster,
                      without re-deriving the scale/offset math here.

    Returns
    -------
    canvas : uint8 ndarray -- (H, W) grayscale
    tile_bboxes : {fov_id: (x0, y0, x1, y1)}, only when
                  *return_tile_bboxes* is True (then the return value is the
                  tuple ``(canvas, tile_bboxes)``)
    """
    from PIL import Image, ImageDraw
    from skimage.transform import resize as sk_resize

    if not thumbnails:
        raise ValueError("thumbnails dict is empty")

    fov_ids = sorted(thumbnails.keys())
    tw, th  = thumbnail_size

    pixel_xs, pixel_ys, canvas_w, canvas_h, pixels_per_unit = _layout_tiles(
        fov_ids, positions, tw, th, padding, pixels_per_unit, flip_y,
    )
    log.debug("Mosaic: %d FOVs, scale=%.4f px/unit", len(fov_ids), pixels_per_unit)

    canvas   = np.full((canvas_h, canvas_w), background, dtype=np.uint8)

    # ── Place thumbnails ──────────────────────────────────────────────────────
    tile_bboxes = {}
    for i, fov_id in enumerate(fov_ids):
        thumb = thumbnails[fov_id]

        if thumb.dtype != np.uint8:
            thumb = thumb.clip(0, 255).astype(np.uint8)
        if thumb.shape[:2] != (th, tw):
            thumb = (
                sk_resize(thumb, (th, tw), anti_aliasing=True, preserve_range=True)
                .clip(0, 255)
                .astype(np.uint8)
            )

        x0, y0 = pixel_xs[i], pixel_ys[i]
        x1 = min(x0 + tw, canvas_w)
        y1 = min(y0 + th, canvas_h)
        canvas[y0:y1, x0:x1] = thumb[: y1 - y0, : x1 - x0]
        tile_bboxes[fov_id] = (int(x0), int(y0), int(x1), int(y1))

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_image = Image.fromarray(canvas)
    if labels or highlight_fov_ids:
        draw = ImageDraw.Draw(canvas_image)
        for i, fov_id in enumerate(fov_ids):
            x0, y0 = pixel_xs[i], pixel_ys[i]
            if highlight_fov_ids and fov_id in highlight_fov_ids:
                x1 = min(x0 + tw, canvas_w) - 1
                y1 = min(y0 + th, canvas_h) - 1
                draw.rectangle([x0, y0, x1, y1], outline=highlight_color, width=highlight_width)
            if labels and fov_id in labels:
                draw.text((x0 + 2, y0 + 2), str(labels[fov_id]), fill=label_color)
        canvas = np.asarray(canvas_image)
    canvas_image.save(str(output_path))
    log.info(
        "Mosaic saved: %s  (%d × %d px, %d FOVs)",
        output_path, canvas_w, canvas_h, len(fov_ids),
    )
    if return_tile_bboxes:
        return canvas, tile_bboxes
    return canvas


def create_mosaic_ffc(
    raw_frames: Dict[int, np.ndarray],
    positions: Dict[int, Tuple[float, float]],
    output_path: Path,
    ffc_field: Optional[np.ndarray] = None,
    crop_px: int = 0,
    thumbnail_size: Tuple[int, int] = (200, 200),
    padding: int = 4,
    background: int = 0,
    pixels_per_unit: Optional[float] = None,
    flip_y: bool = False,
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
    labels: Optional[Dict[int, str]] = None,
    label_color: int = 255,
    highlight_fov_ids: Optional[Set[int]] = None,
    highlight_color: int = 255,
    highlight_width: int = 3,
    return_tile_bboxes: bool = False,
):
    """
    Flat-field-corrected sibling of :func:`create_mosaic`. Takes RAW per-FOV
    frames (not pre-made, already-8-bit, already-independently-contrast-
    stretched thumbnails) so it can (1) divide out a flat-field/vignette
    correction, (2) crop each FOV's overlap border, and (3) apply ONE shared
    contrast stretch across the WHOLE assembled canvas at the end, instead of
    each tile being stretched independently -- the two things that made
    ``create_mosaic`` alone look visibly worse than a properly flat-field-
    corrected mosaic. Kept as a separate function (not a flag on
    ``create_mosaic``) since existing callers of ``create_mosaic`` rely on
    its ``{fov_id: uint8 array}`` contract unchanged.

    Parameters
    ----------
    raw_frames      : {fov_id: 2-D array (H, W)}, any unsigned integer dtype
                      -- e.g. from :func:`load_raw_frames_for_round`
    positions       : {fov_id: (x, y)} stage coordinates
    output_path     : where to save the finished mosaic PNG
    ffc_field       : optional per-pixel flat-field correction map (float,
                      same H×W as the raw frames); each frame is divided by
                      this before cropping/downsampling. ``None`` skips FFC.
    crop_px         : pixels to crop from every edge before downsampling --
                      drops the overlap border shared with neighbouring FOVs
                      (see :func:`MERci.analysis.ffc.compute_mosaic_crop_px`)
    thumbnail_size  : (width, height) of each thumbnail cell in the canvas
    padding         : pixel gap around each thumbnail (also used for border)
    background      : canvas fill value for empty pixels (0-255)
    pixels_per_unit : spatial scale; auto-estimated when ``None``
    flip_y          : if True, mirror the y-axis (microscope convention)
    percentile_clip : (lo_pct, hi_pct) contrast stretch computed ONCE over
                      the whole assembled float canvas's non-background
                      pixels, not per tile
    labels, label_color, highlight_fov_ids, highlight_color, highlight_width,
    return_tile_bboxes : same as :func:`create_mosaic`

    Returns
    -------
    canvas : uint8 ndarray -- (H, W) grayscale
    tile_bboxes : {fov_id: (x0, y0, x1, y1)}, only when
                  *return_tile_bboxes* is True
    """
    from PIL import Image, ImageDraw
    from skimage.transform import resize as sk_resize

    if not raw_frames:
        raise ValueError("raw_frames dict is empty")

    fov_ids = sorted(raw_frames.keys())
    tw, th  = thumbnail_size

    pixel_xs, pixel_ys, canvas_w, canvas_h, pixels_per_unit = _layout_tiles(
        fov_ids, positions, tw, th, padding, pixels_per_unit, flip_y,
    )
    log.debug("FFC mosaic: %d FOVs, scale=%.4f px/unit", len(fov_ids), pixels_per_unit)

    canvas    = np.full((canvas_h, canvas_w), float(background), dtype=np.float32)
    is_filled = np.zeros((canvas_h, canvas_w), dtype=bool)

    # ── Place FFC-corrected, cropped, downsampled tiles ───────────────────────
    tile_bboxes = {}
    for i, fov_id in enumerate(fov_ids):
        frame = raw_frames[fov_id].astype(np.float32)

        if ffc_field is not None:
            frame = np.clip(frame / ffc_field, 0, None)

        if crop_px > 0:
            frame = frame[crop_px:-crop_px, crop_px:-crop_px]

        if frame.shape[:2] != (th, tw):
            frame = sk_resize(frame, (th, tw), anti_aliasing=True, preserve_range=True)

        x0, y0 = pixel_xs[i], pixel_ys[i]
        x1 = min(x0 + tw, canvas_w)
        y1 = min(y0 + th, canvas_h)
        canvas[y0:y1, x0:x1] = frame[: y1 - y0, : x1 - x0]
        is_filled[y0:y1, x0:x1] = True
        tile_bboxes[fov_id] = (int(x0), int(y0), int(x1), int(y1))

    # ── One shared contrast stretch over every filled pixel ───────────────────
    filled_pixels = canvas[is_filled]
    lo, hi = np.percentile(filled_pixels, [percentile_clip[0], percentile_clip[1]])
    if hi > lo:
        stretched = np.clip((canvas - lo) / (hi - lo), 0.0, 1.0)
    else:
        stretched = np.zeros_like(canvas)
    canvas_u8 = (stretched * 255).astype(np.uint8)
    canvas_u8[~is_filled] = background

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_image = Image.fromarray(canvas_u8)
    if labels or highlight_fov_ids:
        draw = ImageDraw.Draw(canvas_image)
        for i, fov_id in enumerate(fov_ids):
            x0, y0 = pixel_xs[i], pixel_ys[i]
            if highlight_fov_ids and fov_id in highlight_fov_ids:
                x1 = min(x0 + tw, canvas_w) - 1
                y1 = min(y0 + th, canvas_h) - 1
                draw.rectangle([x0, y0, x1, y1], outline=highlight_color, width=highlight_width)
            if labels and fov_id in labels:
                draw.text((x0 + 2, y0 + 2), str(labels[fov_id]), fill=label_color)
        canvas_u8 = np.asarray(canvas_image)
    canvas_image.save(str(output_path))
    log.info(
        "FFC mosaic saved: %s  (%d x %d px, %d FOVs)",
        output_path, canvas_w, canvas_h, len(fov_ids),
    )
    if return_tile_bboxes:
        return canvas_u8, tile_bboxes
    return canvas_u8


def load_raw_frames_for_round(
    round_id:       int,
    metadata,                           # ExperimentMetadata
    frame_idx:      int,
    series_idx:     int = 0,            # which series per FOV when multiple exist
    fov_subset:     Optional[List[int]] = None,
    frame_width:    Optional[int] = None,
    frame_height:   Optional[int] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[float, float]]]:
    """
    Read the raw *frame_idx* frame directly from every FOV's own image file
    in *round_id* -- NOT the pre-made, already-8-bit, already-independently-
    contrast-stretched PNG thumbnails :func:`load_thumbnails_for_round` loads.
    Needed for :func:`create_mosaic_ffc`, since FFC division and a shared
    whole-canvas contrast stretch are both meaningless once a per-FOV
    percentile stretch has already discarded the very brightness differences
    FFC corrects and quantized to 8 bits.

    Same FOV/file-resolution walk as :func:`load_thumbnails_for_round`.

    Parameters
    ----------
    round_id      : which imaging round
    metadata      : ExperimentMetadata instance
    frame_idx     : which raw frame index to read from each FOV's file
    series_idx    : when a round has multiple series per FOV, pick this one
    fov_subset    : if given, only load frames for these FOV ids
    frame_width, frame_height : passed to the raw-frame reader (only needed
                    for dax files without a frame-shape-carrying sidecar)

    Returns
    -------
    raw_frames : {fov_id: uint16 array (H, W)}
    positions  : {fov_id: (x, y)}
    """
    from MERci.common.io import read_image_frames

    round_info = metadata.rounds.get(round_id)
    if round_info is None:
        raise KeyError(f"Round {round_id} not found in metadata")

    fov_set = set(fov_subset) if fov_subset is not None else None

    raw_frames: Dict[int, np.ndarray] = {}
    positions:  Dict[int, Tuple[float, float]] = {}

    for fov_id, file_list in round_info.fov_files.items():
        if fov_set is not None and fov_id not in fov_set:
            continue
        if not file_list:
            continue
        idx   = min(series_idx, len(file_list) - 1)
        fpath = file_list[idx]
        if not fpath.exists():
            log.warning("Image file missing: %s", fpath)
            continue

        frames = read_image_frames(fpath, [frame_idx], frame_width, frame_height)
        raw_frames[fov_id] = frames[0]
        positions[fov_id]  = metadata.fovs[fov_id].position

    return raw_frames, positions


def load_thumbnails_for_round(
    round_id:       int,
    metadata,                           # ExperimentMetadata
    thumbnails_dir: Path,
    frame_idx:      int = 0,
    series_idx:     int = 0,            # which series per FOV when multiple exist
    fov_subset:     Optional[List[int]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[float, float]]]:
    """
    Load existing thumbnail PNGs for every FOV in *round_id*.

    Parameters
    ----------
    round_id       : which imaging round
    metadata       : ExperimentMetadata instance
    thumbnails_dir : directory that contains ``{stem}_frame{n:03d}.png`` files
    frame_idx      : which frame index thumbnail to load (default 0)
    series_idx     : when a round has multiple series per FOV, pick this one
    fov_subset     : if given, only load thumbnails for these FOV ids

    Returns
    -------
    thumbnails : {fov_id: uint8 array}
    positions  : {fov_id: (x, y)}
    """
    from PIL import Image

    round_info = metadata.rounds.get(round_id)
    if round_info is None:
        raise KeyError(f"Round {round_id} not found in metadata")

    fov_set = set(fov_subset) if fov_subset is not None else None

    thumbnails: Dict[int, np.ndarray] = {}
    positions:  Dict[int, Tuple[float, float]] = {}

    for fov_id, file_list in round_info.fov_files.items():
        if fov_set is not None and fov_id not in fov_set:
            continue
        if not file_list:
            continue
        idx   = min(series_idx, len(file_list) - 1)
        fpath = file_list[idx]
        thumb_path = (
            Path(thumbnails_dir) / f"{fpath.stem}_frame{frame_idx:03d}.png"
        )
        if not thumb_path.exists():
            log.warning("Thumbnail missing: %s", thumb_path)
            continue

        thumbnails[fov_id] = np.array(Image.open(str(thumb_path)))
        positions[fov_id]  = metadata.fovs[fov_id].position

    return thumbnails, positions


# ── Private helpers ───────────────────────────────────────────────────────────

def _layout_tiles(
    fov_ids: List[int],
    positions: Dict[int, Tuple[float, float]],
    tw: int,
    th: int,
    padding: int,
    pixels_per_unit: Optional[float],
    flip_y: bool,
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """
    Compute per-tile pixel placement + overall canvas size for *fov_ids* at
    their stage *positions*. Shared by :func:`create_mosaic` and
    :func:`create_mosaic_ffc` so both place tiles identically.

    Returns
    -------
    pixel_xs, pixel_ys : int arrays, one entry per fov_id (same order as
                         *fov_ids*) -- top-left pixel of that tile
    canvas_w, canvas_h  : overall canvas size in pixels
    pixels_per_unit     : resolved scale (in case it was auto-estimated)
    """
    xs = np.array([positions[f][0] for f in fov_ids], dtype=float)
    ys = np.array([positions[f][1] for f in fov_ids], dtype=float)

    if flip_y:
        ys = -ys

    x_min, y_min = xs.min(), ys.min()

    if pixels_per_unit is None:
        pixels_per_unit = _estimate_pixels_per_unit(xs, ys, tw, th, padding)

    pixel_xs = ((xs - x_min) * pixels_per_unit + padding).astype(int)
    pixel_ys = ((ys - y_min) * pixels_per_unit + padding).astype(int)

    canvas_w = int(pixel_xs.max()) + tw + padding
    canvas_h = int(pixel_ys.max()) + th + padding

    return pixel_xs, pixel_ys, canvas_w, canvas_h, pixels_per_unit


def _estimate_pixels_per_unit(
    xs: np.ndarray,
    ys: np.ndarray,
    tw: int,
    th: int,
    padding: int,
) -> float:
    """
    Estimate the pixels-per-unit scale from the median nearest-neighbour
    distance between stage positions.  The scale is set so that thumbnails
    corresponding to adjacent FOVs are placed side-by-side with ``padding``
    pixels of gap.
    """
    if len(xs) < 2:
        return 1.0

    coords = np.stack([xs, ys], axis=1)        # (N, 2)
    diff   = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
    sq_d   = (diff ** 2).sum(axis=2)           # (N, N)
    np.fill_diagonal(sq_d, np.inf)
    nn_dist = np.sqrt(sq_d.min(axis=1)).mean()

    if nn_dist == 0:
        return 1.0

    # One cell + one padding per nearest-neighbour step
    cell_size = max(tw, th) + padding
    return cell_size / nn_dist