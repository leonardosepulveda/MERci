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
from typing import Dict, List, Optional, Tuple  # noqa: F401 (Optional/List used in annotations)

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
) -> np.ndarray:
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

    Returns
    -------
    canvas : uint8 ndarray – the complete mosaic
    """
    from PIL import Image, ImageDraw
    from skimage.transform import resize as sk_resize

    if not thumbnails:
        raise ValueError("thumbnails dict is empty")

    fov_ids = sorted(thumbnails.keys())
    tw, th  = thumbnail_size

    xs = np.array([positions[f][0] for f in fov_ids], dtype=float)
    ys = np.array([positions[f][1] for f in fov_ids], dtype=float)

    if flip_y:
        ys = -ys

    x_min, y_min = xs.min(), ys.min()

    # ── Estimate scale ────────────────────────────────────────────────────────
    if pixels_per_unit is None:
        pixels_per_unit = _estimate_pixels_per_unit(xs, ys, tw, th, padding)
    log.debug("Mosaic: %d FOVs, scale=%.4f px/unit", len(fov_ids), pixels_per_unit)

    # ── Canvas size ───────────────────────────────────────────────────────────
    pixel_xs = ((xs - x_min) * pixels_per_unit + padding).astype(int)
    pixel_ys = ((ys - y_min) * pixels_per_unit + padding).astype(int)

    canvas_w = int(pixel_xs.max()) + tw + padding
    canvas_h = int(pixel_ys.max()) + th + padding
    canvas   = np.full((canvas_h, canvas_w), background, dtype=np.uint8)

    # ── Place thumbnails ──────────────────────────────────────────────────────
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

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_image = Image.fromarray(canvas)
    if labels:
        draw = ImageDraw.Draw(canvas_image)
        for i, fov_id in enumerate(fov_ids):
            if fov_id not in labels:
                continue
            x0, y0 = pixel_xs[i], pixel_ys[i]
            draw.text((x0 + 2, y0 + 2), str(labels[fov_id]), fill=label_color)
        canvas = np.asarray(canvas_image)
    canvas_image.save(str(output_path))
    log.info(
        "Mosaic saved: %s  (%d × %d px, %d FOVs)",
        output_path, canvas_w, canvas_h, len(fov_ids),
    )
    return canvas


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