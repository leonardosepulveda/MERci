# merfish_pipeline/common/config.py
"""
Central configuration dataclass.  One instance is shared by both the
acquisition-planning modules and the online-analysis modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ExperimentConfig:
    """
    All tuneable parameters for one experiment's acquisition and analysis.

    Acquisition parameters
    ----------------------
    microscope            : microscope identifier, e.g. ``"MF3"``, ``"MF5"``
    pixel_size_um         : camera pixel size in µm
    image_size_px         : number of pixels along one side of a raw frame
    non_overlap_fraction  : fraction of the FOV covered per stage step
                            (step_size_um = pixel_size_um × image_size_px
                                           × non_overlap_fraction)
    hal_templates_dir     : directory that contains HAL config XML templates

    Analysis parameters
    -------------------
    t_min / t_max                : analysis window within the fluidics interval
    imaging_idle_threshold       : seconds with no new file → imaging is done
    thumbnail_frames             : which frame indices to thumbnail (None = all)
    thumbnail_size               : (width, height) for PNG thumbnails
    thumbnail_percentile_clip    : (lo_pct, hi_pct) for contrast stretching
    histogram_bins / range       : histogram parameters
    mosaic_frame_idx             : frame index used when assembling mosaics
    mosaic_padding               : pixel gap between thumbnails in the mosaic
    mosaic_flip_y                : mirror y-axis (depends on microscope convention)
    """

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir:      Path
    metadata_dir:  Path
    analysis_dir:  Path
    round_info_csv: Path      # replaces the old imaging_info_csv
    positions_txt:  Path

    # ── Microscope / acquisition ───────────────────────────────────────────────
    microscope:             str            = "MF3"
    image_suffix:           str            = ".dax"
    image_dtype:            str            = "uint16"
    frame_width:            Optional[int]  = None
    frame_height:           Optional[int]  = None
    pixel_size_um:          float          = 0.109
    image_size_px:          int            = 2048
    non_overlap_fraction:   float          = 0.9
    hal_templates_dir:      Optional[Path] = None

    # ── Timing (seconds) ──────────────────────────────────────────────────────
    t_min:                   float = 300.0
    t_max:                   float = 3300.0
    imaging_idle_threshold:  float = 180.0
    poll_interval:           float = 60.0

    # ── Analysis ──────────────────────────────────────────────────────────────
    thumbnail_frames:           Optional[List[int]]         = None
    thumbnail_size:             Tuple[int, int]             = (200, 200)
    thumbnail_percentile_clip:  Tuple[float, float]         = (1.0, 99.0)
    histogram_bins:             int                         = 512
    histogram_range:            Tuple[int, int]             = (0, 65535)
    mosaic_frame_idx:           int                         = 0
    mosaic_padding:             int                         = 4
    mosaic_flip_y:              bool                        = False

    # ── Derived properties ─────────────────────────────────────────────────────

    @property
    def step_size_um(self) -> float:
        """Stage step size in µm, derived from pixel/FOV parameters."""
        return self.pixel_size_um * self.image_size_px * self.non_overlap_fraction

    # ── Initialisation ─────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        for attr in (
            "data_dir", "metadata_dir", "analysis_dir",
            "round_info_csv", "positions_txt",
        ):
            setattr(self, attr, Path(getattr(self, attr)))

        if self.hal_templates_dir is not None:
            self.hal_templates_dir = Path(self.hal_templates_dir)

        for sub in ("thumbnails", "stats", "histograms", "mosaics", "done", "logs"):
            (self.analysis_dir / sub).mkdir(parents=True, exist_ok=True)