# MERci/live_round_mosaic.py
"""
Logic behind ``notebooks/during_imaging/round_mosaics.ipynb`` -- a live
quick-look mosaic (one per real imaging color) built from a single frame per
FOV near a target stage z, for whichever rounds are selected. Distinct from
:func:`MERci.scheduler.build_round_mosaics`'s production mosaics (mid-z,
optional FFC, built only once a round is 100% done): this one is meant to
run continuously alongside an active acquisition and show partial progress.

``LiveRoundMosaicBuilder`` holds the per-session state a live run needs
(FFC fields, contrast ranges, tile-placement geometry, in-progress canvases)
across repeated polls, so nothing gets recomputed or re-read once cached.

**Tile-by-tile, not batch-then-display.** Each round/color gets a
persistent canvas (:meth:`get_canvas`), placed once via
:meth:`get_round_layout` at the exact pixel positions
``analysis.round.create_mosaic``/``create_mosaic_ffc`` would use for EVERY
FOV *planned* for that round (not just the ones imaged so far, so a tile's
position never shifts as more FOVs arrive). :meth:`build_round_mosaic` first
bulk-loads every FOV whose tile is already cached on disk (no per-tile
redraw), shows that accumulated state once, then reads/corrects/places any
remaining FOV one at a time with a throttled redraw after each -- so a round
already mostly processed from a prior kernel session shows that state
immediately instead of re-crawling through FOVs that were already done.

**The redraw itself must stay cheap regardless of real mosaic size.** On a
real ~1166-FOV, ~10600x6700 px canvas, matplotlib's own ``imshow``+draw of
the full-resolution array measured at ~6s, and a full-resolution PNG disk
write at ~3s -- confirmed directly, and enough to dominate wall-clock time
if paid on every redraw. Fixed by decoupling the two costs: the on-screen
preview is downsampled first (:func:`MERci.plots.round_mosaic_plots.show_round_mosaic`,
under 0.1s regardless of canvas size) so it can redraw every
``live_redraw_min_interval_sec``; the full-resolution PNG on disk is only
rewritten at the coarser ``disk_save_min_interval_sec``.

**Camera orientation.** Every raw frame is re-oriented via
``apply_microscope_orientation`` immediately after reading, before
FFC/contrast-stretch/placement -- otherwise tile CONTENT can appear
rotated/mirrored relative to its neighbours even though each tile lands at
the geometrically correct stage position (see
``notebooks/misc/correct_camera_rotation.ipynb``). A cached FFC field is
itself estimated from un-oriented raw pixels (cheaper: orient the small
resulting field once rather than every sample frame) -- mathematically
equivalent either way, since averaging, isotropic Gaussian smoothing,
percentile normalization, and clipping all commute exactly with a fixed
transpose/flip.

**A FOV's file existing is not the same as it being safe to read.** HAL
creates a FOV's image store as soon as it starts writing that stack, well
before every frame has actually landed on disk, so :meth:`round_imaged_fov_ids`'s
``.exists()`` check (fast, no I/O, safe every poll) can legitimately call a
FOV "imaged" while it's still being written. ``is_path_stable`` is checked
right before the actual read (not for every already-imaged FOV up front,
which would cost its own delay per FOV every poll) -- an unstable FOV is
simply left pending and retried on the next poll.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize as sk_resize

from .common.config import ExperimentConfig
from .common.io import is_path_stable, read_image_frames
from .common.metadata import ExperimentMetadata, RoundInfo, SeriesInfo
from .acquisition.configs import find_frame_table_for_hal_config
from .acquisition.merlin_config import apply_microscope_orientation, load_microscope_orientation
from .acquisition.positions import find_exterior_fovs
from .analysis.ffc import apply_ffc, compute_ffc_field_for_color, load_ffc_field, save_ffc_field
from .analysis.fov import create_thumbnail
from .analysis.round import _layout_tiles
from .scheduler import resolve_round_flip_y
from .plots.round_mosaic_plots import show_round_mosaic

# Real imaging_round values start at 1 (acquisition.dave.create_round_info),
# so 0 is never a real round id.
FOCUSTEST_ROUND_ID = 0


def register_focustest_round(
    metadata: ExperimentMetadata,
    config: ExperimentConfig,
    microscope: str,
    focustest_round_id: int = FOCUSTEST_ROUND_ID,
) -> bool:
    """
    Register the focus-lock test (``before_imaging``'s
    ``create_focus_test_dave_config``) as a synthetic round in *metadata*,
    if this experiment has one.

    It's a standalone calibration procedure, not a real imaging round in
    ``round_info.csv``, so :meth:`ExperimentMetadata.load` never sees it.
    Registering it here (as *focustest_round_id*, default 0) lets it show up
    in a round-selection UI exactly like any other round, with no further
    setup, whenever this experiment actually has a focus-test HAL config and
    its own ``data/focus_test/`` movies.

    Returns True iff a round was registered (False if already present, or no
    focus-test HAL config was found).
    """
    if focustest_round_id in metadata.rounds:
        return False
    mic = microscope.lower()
    hal_configs = sorted(config.settings_dir.glob(f"hal-config-{mic}-focustest-*.xml"))
    if not hal_configs:
        return False
    # Same per-FOV zero-pad width Dave itself uses for these movies' filenames
    # (v2Generator's copyChildren pads to len(str(n_fovs)) -- confirmed
    # directly against the real source -- distinct from round_info.csv series
    # patterns' fov_pad_width convention used everywhere else).
    pad = len(str(metadata.n_fovs))
    series = SeriesInfo(
        name         = f"hal-{mic}-focustest_{{fov:0{pad}d}}",
        round_id     = focustest_round_id,
        imaging_type = "focustest",
        hal_config   = hal_configs[-1].name,
        data_dir     = config.data_dir / "focus_test",
    )
    series.candidate_dirs = [series.data_dir]
    round_info = RoundInfo(round_id=focustest_round_id, series=[series])
    for fov_id in sorted(metadata.fovs):
        round_info.fov_files[fov_id] = [series.resolve_path(fov_id, config.image_suffix)]
    metadata.rounds[focustest_round_id] = round_info
    metadata.all_series.append(series)
    return True


def stretch_to_uint8(
    frame: np.ndarray, target_size: Tuple[int, int], vmin_vmax: Tuple[float, float],
) -> np.ndarray:
    """
    Contrast-stretch *frame* using a FIXED ``(vmin, vmax)`` (see
    :meth:`LiveRoundMosaicBuilder.get_or_compute_contrast_range` -- not a
    per-frame percentile) and resize to *target_size*. Returns the array
    only; callers cache it themselves via
    :meth:`LiveRoundMosaicBuilder.tile_cache_path` when appropriate.
    """
    vmin, vmax = vmin_vmax
    if vmax > vmin:
        stretched = np.clip((frame.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        stretched = np.zeros(frame.shape, dtype=np.float32)
    tw, th = target_size
    resized = sk_resize(stretched, (th, tw), anti_aliasing=True, preserve_range=True)
    return (resized * 255).clip(0, 255).astype(np.uint8)


class LiveRoundMosaicBuilder:
    """
    Per-session state + logic for one live-mosaic notebook run. See module
    docstring for the overall design.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
        microscope: str,
        microscope_orientation_dir: Path,
        cache_dir: Path,
        figures_dir: Path,
        thumbnails_dir: Path,
        notebook_name: str,
        target_z_um: float,
        excluded_colors: List[float],
        focustest_round_id: Optional[int] = FOCUSTEST_ROUND_ID,
        enable_ffc: bool = False,
        ffc_n_fovs: int = 10,
        contrast_sample_n_fovs: int = 10,
        contrast_low_pct: float = 5.0,
        contrast_high_pct: float = 99.0,
        live_preview_max_px: int = 1400,
        live_redraw_min_interval_sec: float = 0.5,
        disk_save_min_interval_sec: float = 20.0,
        catchup_read_delay_sec: float = 0.02,
    ):
        self.config = config
        self.metadata = metadata
        self.microscope = microscope
        self.cache_dir = Path(cache_dir)
        self.figures_dir = Path(figures_dir)
        self.thumbnails_dir = Path(thumbnails_dir)
        self.notebook_name = notebook_name
        self.target_z_um = target_z_um
        self.excluded_colors = excluded_colors
        self.focustest_round_id = focustest_round_id
        self.enable_ffc = enable_ffc
        self.ffc_n_fovs = ffc_n_fovs
        self.contrast_sample_n_fovs = contrast_sample_n_fovs
        self.contrast_low_pct = contrast_low_pct
        self.contrast_high_pct = contrast_high_pct
        self.live_preview_max_px = live_preview_max_px
        self.live_redraw_min_interval_sec = live_redraw_min_interval_sec
        self.disk_save_min_interval_sec = disk_save_min_interval_sec
        self.catchup_read_delay_sec = catchup_read_delay_sec

        self.microscope_orientation = load_microscope_orientation(microscope, microscope_orientation_dir)

        # Exterior FOVs (outer edge of the imaged grid) are used both for FFC
        # sampling (likely near-empty) and excluded from contrast sampling
        # (likely NOT to carry real tissue signal) -- computed once from the
        # full planned FOV grid, independent of what's imaged so far.
        positions = {f: metadata.fovs[f].position for f in metadata.fovs}
        coords_arr = np.array([positions[f] for f in sorted(positions)])
        from scipy.spatial import KDTree
        nn_dist, _ = KDTree(coords_arr).query(coords_arr, k=2)
        self.step_size_um = float(np.median(nn_dist[:, 1]))
        self.exterior_fov_ids: Set[int] = find_exterior_fovs(positions, self.step_size_um)

        self._ffc_fields: Dict[float, np.ndarray] = {}
        self._contrast_ranges: Dict[Tuple[int, float], Tuple[float, float]] = {}
        self._round_layouts: Dict[int, dict] = {}
        self._mosaic_canvases: Dict[Tuple[int, float], np.ndarray] = {}
        self._placed_tiles: Dict[Tuple[int, float], Set[int]] = {}

    # ── Round/color/frame resolution ────────────────────────────────────────

    def resolve_round_color_frames(self, round_id: int) -> Dict[float, int]:
        """
        Return ``{color_nm: frame_idx}`` for *round_id*'s own frame table --
        the frame closest to ``target_z_um`` for every real color it has,
        except ``excluded_colors`` (and blanks, which have no color at all).

        The focus-test round is exempt from ``excluded_colors``: its only
        real channel is normally 488 nm (the bead/focus-lock color, excluded
        everywhere else as "not real tissue signal"), but for the focus-test
        round itself 488 nm IS the signal being checked -- excluding it would
        leave the round with no resolved colors at all.
        """
        color_frames: Dict[float, int] = {}
        for s in self.metadata.series_for_round(round_id):
            if not s.hal_config:
                continue
            frame_table_path = find_frame_table_for_hal_config(
                self.config.settings_dir / s.hal_config, self.config.metadata_dir)
            if frame_table_path is None:
                continue
            frame_table = pd.read_csv(frame_table_path)
            for color in sorted(frame_table["color"].dropna().unique()):
                if round_id != self.focustest_round_id and any(
                    round(color) == round(excluded) for excluded in self.excluded_colors
                ):
                    continue
                candidates = frame_table[frame_table["color"].round(0) == round(color)]
                frame_idx = int((candidates["z"] - self.target_z_um).abs().idxmin())
                resolved_z = float(candidates.loc[frame_idx, "z"])
                color_frames[float(color)] = frame_idx
                if abs(resolved_z - self.target_z_um) > 5.0:
                    print(f"  round {round_id}, color {color:.0f} nm: nearest available z is "
                          f"{resolved_z:.1f} um (requested {self.target_z_um:.1f} um) -- frame {frame_idx}")
        return color_frames

    def resolve_round_by_imaging_type(self, imaging_type: str) -> Optional[int]:
        """Return the round id whose series has ``imaging_type == imaging_type``, else None."""
        target = imaging_type.strip().lower()
        for round_id in sorted(self.metadata.rounds):
            for s in self.metadata.series_for_round(round_id):
                if (s.imaging_type or "").strip().lower() == target:
                    return round_id
        return None

    def resolve_round_token(self, token) -> int:
        """``int`` -> that round id directly; ``str`` -> resolved by imaging_type."""
        if isinstance(token, str):
            round_id = self.resolve_round_by_imaging_type(token)
            if round_id is None:
                raise ValueError(f"No round has a series with imaging_type={token!r} "
                                  f"-- check round_info.csv, or use an explicit round id.")
            return round_id
        return int(token)

    def round_label_for(self, round_id: int):
        """``"cells"`` if *round_id* has a cells series, else *round_id* itself."""
        for s in self.metadata.series_for_round(round_id):
            if (s.imaging_type or "").strip().lower() == "cells":
                return "cells"
        return round_id

    # ── FOV / processed-state bookkeeping ───────────────────────────────────

    def round_imaged_fov_ids(self, round_id: int) -> List[int]:
        series = self.metadata.series_for_round(round_id)
        return [
            fov_id for fov_id in sorted(self.metadata.fovs)
            if any(s.resolve_path(fov_id, self.config.image_suffix).exists() for s in series)
        ]

    def thumbnail_path_for(self, image_path: Path, frame_idx: int) -> Path:
        return self.thumbnails_dir / f"{image_path.stem}_frame{frame_idx:03d}.png"

    def fov_is_processed(
        self, round_id: int, fov_id: int, color_frames: Dict[float, int], series: List[SeriesInfo],
    ) -> bool:
        """True iff *fov_id* has a cached thumbnail for EVERY color of this round."""
        existing = [s.resolve_path(fov_id, self.config.image_suffix) for s in series]
        existing = [p for p in existing if p.exists()]
        if not existing:
            return False
        image_path = existing[0]
        return all(self.thumbnail_path_for(image_path, frame_idx).exists()
                   for frame_idx in color_frames.values())

    def build_state_df(self, round_ids: List[int], round_color_frames: Dict[int, Dict[float, int]]) -> pd.DataFrame:
        """
        One row per round in *round_ids*: how many of its FOVs are imaged
        right now, how many already have a computed thumbnail for every one
        of that round's colors ("processed"), and the experiment's total FOV
        count.
        """
        rows = []
        for round_id in round_ids:
            color_frames = round_color_frames[round_id]
            series = self.metadata.series_for_round(round_id)
            imaged_ids = self.round_imaged_fov_ids(round_id)
            processed_ids = [f for f in imaged_ids if self.fov_is_processed(round_id, f, color_frames, series)]
            rows.append({
                "round": self.round_label_for(round_id), "round_id": round_id,
                "imaged_fovs": len(imaged_ids), "processed_fovs": len(processed_ids),
                "total_fovs": self.metadata.n_fovs,
            })
        return pd.DataFrame(rows, columns=["round", "round_id", "imaged_fovs", "processed_fovs", "total_fovs"])

    # ── Canvas / tile-cache paths ────────────────────────────────────────────

    def mosaic_path(self, round_id: int, color_nm: float) -> Path:
        return self.figures_dir / f"{self.notebook_name}.round{round_id:03d}_{color_nm:.0f}nm.png"

    def tile_cache_path(self, round_id: int, color_nm: float, fov_id: int) -> Path:
        """
        Dedicated per-round/color/FOV canvas-tile cache -- distinct from
        :meth:`thumbnail_path_for`'s plain, per-frame-percentile "processed"
        marker: this one is oriented, FFC-divided (if enabled), and
        stretched with the SHARED contrast range, i.e. exactly the pixels
        placed into the canvas -- loading it back skips a raw read entirely,
        regardless of ``enable_ffc``.
        """
        d = self.cache_dir / "tiles"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"round{round_id:03d}_{color_nm:.0f}nm_fov{fov_id:04d}.png"

    def get_round_layout(self, round_id: int) -> dict:
        """
        Tile pixel-placement geometry for EVERY FOV *planned* for this round
        (``metadata.rounds[...].fov_files`` covers the full positions-file
        grid, independent of which files actually exist on disk yet),
        computed once and cached -- so a tile never shifts position as more
        FOVs arrive, and matches exactly where
        ``analysis.round.create_mosaic``/``create_mosaic_ffc`` would place
        the same FOV.
        """
        if round_id not in self._round_layouts:
            round_info = self.metadata.rounds[round_id]
            all_fov_ids = sorted(round_info.fov_files)
            positions = {f: self.metadata.fovs[f].position for f in all_fov_ids}
            flip_y = resolve_round_flip_y(round_id, self.config, self.metadata)
            tw, th = self.config.thumbnail_size
            pixel_xs, pixel_ys, canvas_w, canvas_h, _ = _layout_tiles(
                all_fov_ids, positions, tw, th, self.config.mosaic_padding, None, flip_y)
            self._round_layouts[round_id] = {
                "pixel_xy": {f: (int(x), int(y)) for f, x, y in zip(all_fov_ids, pixel_xs, pixel_ys)},
                "w": canvas_w, "h": canvas_h,
            }
        return self._round_layouts[round_id]

    def get_canvas(self, round_id: int, color_nm: float) -> np.ndarray:
        key = (round_id, color_nm)
        if key not in self._mosaic_canvases:
            layout = self.get_round_layout(round_id)
            self._mosaic_canvases[key] = np.zeros((layout["h"], layout["w"]), dtype=np.uint8)
            self._placed_tiles[key] = set()
        return self._mosaic_canvases[key]

    def place_tile(self, round_id: int, color_nm: float, fov_id: int, tile: np.ndarray) -> None:
        layout = self.get_round_layout(round_id)
        canvas = self.get_canvas(round_id, color_nm)
        tw, th = self.config.thumbnail_size
        if tile.dtype != np.uint8:
            tile = tile.clip(0, 255).astype(np.uint8)
        if tile.shape[:2] != (th, tw):
            tile = (sk_resize(tile, (th, tw), anti_aliasing=True, preserve_range=True)
                    .clip(0, 255).astype(np.uint8))
        x0, y0 = layout["pixel_xy"][fov_id]
        x1, y1 = min(x0 + tw, canvas.shape[1]), min(y0 + th, canvas.shape[0])
        canvas[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]
        self._placed_tiles[(round_id, color_nm)].add(fov_id)

    # ── FFC / contrast range ─────────────────────────────────────────────────

    def get_or_compute_ffc_field(
        self, color_nm: float, frame_idx: int, available_fov_ids: List[int], series: List[SeriesInfo],
    ) -> Optional[np.ndarray]:
        """
        Returns the field already oriented via ``microscope_orientation`` --
        the cache on disk stays in RAW (un-oriented) pixel space either way,
        since averaging + isotropic Gaussian smoothing + percentile
        normalization + clipping all commute exactly with a fixed
        transpose/flip, so orienting the finished field once here is
        identical to orienting every raw sample frame before computing it,
        just cheaper. Samples come from real exterior FOVs (the outer edge
        of the imaged grid) already imaged in whatever round first needs
        this color's field -- not a fixed reference round, since the
        first-processed round might not yet have all ``ffc_n_fovs`` exterior
        FOVs done. Cached to disk and reused for every round afterward,
        since vignetting is a fixed property of the microscope/channel, not
        of any one round.
        """
        if not self.enable_ffc:
            return None
        if color_nm in self._ffc_fields:
            return self._ffc_fields[color_nm]

        cache_path = self.cache_dir / f"ffc_field_{color_nm:.0f}nm.npz"
        if cache_path.exists():
            field, _ = load_ffc_field(cache_path)
            field = apply_microscope_orientation(field, **self.microscope_orientation)
            self._ffc_fields[color_nm] = field
            print(f"Loaded cached FFC field ({color_nm:.0f} nm): {cache_path}")
            return field

        candidate_ids = sorted(self.exterior_fov_ids & set(available_fov_ids))[:self.ffc_n_fovs]
        if not candidate_ids:
            print(f"  FFC ({color_nm:.0f} nm): no exterior FOVs available yet -- will retry later.")
            return None

        samples = []
        for fov_id in candidate_ids:
            paths = [s.resolve_path(fov_id, self.config.image_suffix) for s in series]
            existing = [p for p in paths if p.exists() and is_path_stable(p)]
            if existing:
                samples.append((existing[0], frame_idx))
        if not samples:
            return None

        field, field_meta = compute_ffc_field_for_color(
            samples, frame_width=self.config.frame_width, frame_height=self.config.frame_height)
        save_ffc_field(cache_path, field, field_meta)   # cache stays RAW-space
        field = apply_microscope_orientation(field, **self.microscope_orientation)
        self._ffc_fields[color_nm] = field
        print(f"Computed FFC field ({color_nm:.0f} nm) from {len(samples)} exterior FOV(s): {cache_path}")
        return field

    def get_or_compute_contrast_range(
        self, round_id: int, color_nm: float, frame_idx: int,
        series: List[SeriesInfo], available_fov_ids: List[int],
    ) -> Optional[Tuple[float, float]]:
        """
        ``(vmin, vmax)`` from the pooled histogram of ``contrast_sample_n_fovs``
        random interior FOVs, in the SAME space tiles are placed in (oriented,
        and FFC-divided if enabled) -- returns None if no interior FOV is
        imaged yet (very early in a round), so the caller can fall back to a
        per-tile percentile stretch temporarily rather than caching a
        meaningless range. A single shared range per (round, color) keeps
        every tile visually consistent with its neighbours without needing
        the whole round done first, unlike a single shared whole-canvas
        stretch computed only after the fact.
        """
        key = (round_id, color_nm)
        if key in self._contrast_ranges:
            return self._contrast_ranges[key]

        cache_path = self.cache_dir / f"contrast_range_round{round_id:03d}_{color_nm:.0f}nm.npz"
        if cache_path.exists():
            npz = np.load(cache_path)
            vmin_vmax = (float(npz["vmin"]), float(npz["vmax"]))
            self._contrast_ranges[key] = vmin_vmax
            print(f"Loaded cached contrast range (round {round_id}, {color_nm:.0f} nm): "
                  f"vmin={vmin_vmax[0]:.1f}, vmax={vmin_vmax[1]:.1f}")
            return vmin_vmax

        interior_ids = sorted(set(available_fov_ids) - self.exterior_fov_ids)
        if not interior_ids:
            return None

        rng = np.random.default_rng()
        sample_ids = rng.choice(interior_ids, size=min(self.contrast_sample_n_fovs, len(interior_ids)),
                                 replace=False)

        ffc_field = self.get_or_compute_ffc_field(color_nm, frame_idx, available_fov_ids, series) \
            if self.enable_ffc else None
        pooled = []
        for fov_id in sample_ids:
            paths = [s.resolve_path(fov_id, self.config.image_suffix) for s in series]
            # is_path_stable, not just .exists(): a sampled FOV's file can
            # already exist while HAL is still mid-write -- reading a
            # not-yet-written frame from it would raise the same
            # IndexError/truncated-read build_round_mosaic otherwise guards
            # against. Skipping it here just shrinks this one-time sample
            # pool by one; a later cycle recomputes and caches the real range.
            existing = [p for p in paths if p.exists() and is_path_stable(p)]
            if not existing:
                continue
            try:
                frame = read_image_frames(existing[0], [frame_idx],
                                           frame_width=self.config.frame_width,
                                           frame_height=self.config.frame_height)[0]
            except Exception as exc:
                print(f"  round {round_id}, {color_nm:.0f} nm, FOV {fov_id}: "
                      f"contrast-sample read failed ({exc!r}) -- skipping this sample.")
                continue
            frame = apply_microscope_orientation(frame, **self.microscope_orientation)
            if ffc_field is not None:
                frame = apply_ffc(frame, ffc_field)
            pooled.append(frame.ravel())
        if not pooled:
            return None

        pooled_arr = np.concatenate(pooled)
        vmin, vmax = np.percentile(pooled_arr, [self.contrast_low_pct, self.contrast_high_pct])
        vmin_vmax = (float(vmin), float(vmax))
        np.savez_compressed(cache_path, vmin=vmin_vmax[0], vmax=vmin_vmax[1])
        self._contrast_ranges[key] = vmin_vmax
        print(f"Contrast range (round {round_id}, {color_nm:.0f} nm): vmin={vmin_vmax[0]:.1f}, "
              f"vmax={vmin_vmax[1]:.1f} (from {len(sample_ids)} interior FOV(s))")
        return vmin_vmax

    # ── Main build step ──────────────────────────────────────────────────────

    def build_round_mosaic(
        self, round_id: int, color_frames: Dict[float, int], fov_ids: List[int], label,
    ) -> Dict[float, np.ndarray]:
        """
        For each color: bulk-load whatever's already in ``tile_cache_path``
        (fast, no raw reads) with no per-tile redraw, show that accumulated
        state once, then read, orient, correct, stretch, cache, and place
        any remaining FOV one at a time with a throttled redraw after each.
        So a round that's already mostly processed (e.g. from a prior kernel
        session) shows that state immediately instead of re-crawling through
        already-done FOVs. Returns ``{color_nm: canvas}`` (the same mutable
        arrays :meth:`get_canvas` holds).
        """
        series = self.metadata.series_for_round(round_id)
        last_redraw = [0.0]      # mutable cells so maybe_redraw can update them
        last_disk_save = [0.0]

        def maybe_redraw(force: bool = False) -> None:
            now = time.time()
            if not (force or (now - last_redraw[0]) >= self.live_redraw_min_interval_sec):
                return
            save_full_res = force or (now - last_disk_save[0]) >= self.disk_save_min_interval_sec
            show_round_mosaic(
                round_id, {c: self.get_canvas(round_id, c) for c in color_frames}, label,
                mosaic_paths={c: self.mosaic_path(round_id, c) for c in color_frames},
                live_preview_max_px=self.live_preview_max_px, save_full_res=save_full_res,
            )
            last_redraw[0] = now
            if save_full_res:
                last_disk_save[0] = now

        for color_nm, frame_idx in color_frames.items():
            ffc_field = self.get_or_compute_ffc_field(color_nm, frame_idx, fov_ids, series)
            placed = self._placed_tiles.get((round_id, color_nm), set())
            pending = [f for f in fov_ids if f not in placed]

            still_pending = []
            for fov_id in pending:
                cpath = self.tile_cache_path(round_id, color_nm, fov_id)
                if cpath.exists():
                    self.place_tile(round_id, color_nm, fov_id, np.array(Image.open(str(cpath))))
                else:
                    still_pending.append(fov_id)
            if len(still_pending) < len(pending):
                maybe_redraw(force=True)   # show the bulk-loaded state immediately
            if not still_pending:
                continue   # nothing new for this color -- skip the contrast-range lookup entirely

            vmin_vmax = self.get_or_compute_contrast_range(round_id, color_nm, frame_idx, series, fov_ids)

            for fov_id in still_pending:
                existing = [s.resolve_path(fov_id, self.config.image_suffix) for s in series]
                existing = [p for p in existing if p.exists()]
                if not existing:
                    continue
                image_path = existing[0]

                if not is_path_stable(image_path):
                    print(f"  round {round_id}, {color_nm:.0f} nm, FOV {fov_id}: "
                          f"file still being written -- will retry next poll.")
                    continue

                thumb_path = self.thumbnail_path_for(image_path, frame_idx)

                # is_path_stable only confirms the store's on-disk size
                # hasn't changed in the last stability_delay (default 0.1s)
                # -- for a zarr store HAL grows one frame at a time, that's
                # also true BETWEEN two frame writes whenever HAL's per-frame
                # cadence is slower than stability_delay, so a genuinely
                # still-growing store can read as "stable" mid-round and get
                # read here before frame_idx actually exists yet (confirmed
                # directly against a real experiment: an out-of-bounds error
                # reading a not-yet-written frame; the analogous .dax failure
                # is a truncated read). Same remedy as the is_path_stable
                # check just above: leave this FOV pending and retry next
                # poll, rather than letting one FOV's read crash the whole
                # live loop.
                try:
                    frame = read_image_frames(
                        image_path, [frame_idx],
                        frame_width=self.config.frame_width, frame_height=self.config.frame_height,
                    )[0]
                except Exception as exc:
                    print(f"  round {round_id}, {color_nm:.0f} nm, FOV {fov_id}: "
                          f"read failed ({exc!r}) -- still being written? will retry next poll.")
                    continue
                time.sleep(self.catchup_read_delay_sec)
                frame = apply_microscope_orientation(frame, **self.microscope_orientation)

                # Plain (non-FFC, per-frame-percentile) thumbnail is always
                # written as the "this FOV is processed" marker --
                # fov_is_processed checks THIS file, kept independent of
                # enable_ffc/contrast range so the cache stays reusable if
                # either changes later.
                marker_tile = None
                if not thumb_path.exists():
                    marker_tile = create_thumbnail(frame, thumb_path, target_size=self.config.thumbnail_size,
                                                    percentile_clip=self.config.thumbnail_percentile_clip)

                corrected = apply_ffc(frame, ffc_field) if ffc_field is not None else frame

                if vmin_vmax is not None:
                    tile = stretch_to_uint8(corrected, self.config.thumbnail_size, vmin_vmax)
                    Image.fromarray(tile).save(str(self.tile_cache_path(round_id, color_nm, fov_id)))
                elif ffc_field is None:
                    # No shared contrast range yet (too few interior FOVs so
                    # far) and no FFC correction needed -- the marker
                    # thumbnail (a per-frame percentile stretch) IS the right
                    # tile content; not cached under tile_cache_path, so it's
                    # redone with the real shared range once enough interior
                    # FOVs exist.
                    tile = marker_tile if marker_tile is not None else np.array(Image.open(str(thumb_path)))
                else:
                    # FFC enabled but no shared range yet -- temporary
                    # per-frame percentile stretch of the FFC-corrected frame,
                    # not cached.
                    lo, hi = np.percentile(corrected, [self.contrast_low_pct, self.contrast_high_pct])
                    tile = stretch_to_uint8(corrected, self.config.thumbnail_size, (float(lo), float(hi)))

                self.place_tile(round_id, color_nm, fov_id, tile)
                maybe_redraw()

        maybe_redraw(force=True)   # always show this cycle's final state, even if throttled
        return {c: self.get_canvas(round_id, c) for c in color_frames}
