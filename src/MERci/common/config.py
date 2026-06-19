# MERci/common/config.py
"""
Central configuration dataclass.  One instance is shared by both the
acquisition-planning modules and the online-analysis modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ── Fluidics t_max defaults (seconds) ─────────────────────────────────────────
T_MAX_ADAPTOR = 6000.0   # 100 min — adaptor-based fluidics
T_MAX_DIRECT  = 3000.0   # 50 min  — direct-readout fluidics


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
    fluidics_type                : ``"adaptor"`` (t_max=100 min) or
                                   ``"direct"`` (t_max=50 min); sets t_max
                                   automatically when t_max is left as None
    t_min / t_max                : analysis window within the fluidics interval
                                   (t_max defaults to fluidics_type preset)
    imaging_idle_threshold       : seconds with no new file → imaging is done
    thumbnail_frames             : which frame indices to thumbnail (None = all)
    thumbnail_size               : (width, height) for PNG thumbnails
    thumbnail_percentile_clip    : (lo_pct, hi_pct) for contrast stretching
    histogram_bins / range       : histogram parameters
    mosaic_padding               : pixel gap between thumbnails in the mosaic
    mosaic_flip_y                : mirror y-axis; if None, auto-read from HAL config
    fov_subset                   : limit analysis to these FOV ids (None = all)
    """

    # ── Required paths ─────────────────────────────────────────────────────────
    data_dir:      Path
    metadata_dir:  Path
    analysis_dir:  Path
    round_info_csv: Path
    positions_txt:  Path

    # ── Optional paths ─────────────────────────────────────────────────────────
    settings_dir:       Optional[Path] = None   # SAMPLE_DIR/settings/ for HAL XMLs
    hal_templates_dir:  Optional[Path] = None

    # ── Microscope / acquisition ───────────────────────────────────────────────
    microscope:             str            = "MF3"
    image_suffix:           str            = ".zarr"
    image_dtype:            str            = "uint16"
    frame_width:            Optional[int]  = None
    frame_height:           Optional[int]  = None
    pixel_size_um:          float          = 0.109
    image_size_px:          int            = 2048
    non_overlap_fraction:   float          = 0.9

    # ── Timing (seconds) ──────────────────────────────────────────────────────
    fluidics_type:           str            = "adaptor"  # "adaptor" or "direct"
    t_min:                   float          = 300.0      # 5 min
    t_max:                   Optional[float] = None      # set from fluidics_type if None
    imaging_idle_threshold:  float          = 180.0
    poll_interval:           float          = 60.0

    # ── Analysis ──────────────────────────────────────────────────────────────
    thumbnail_frames:           Optional[List[int]]          = None
    thumbnail_size:             Tuple[int, int]              = (200, 200)
    thumbnail_percentile_clip:  Tuple[float, float]          = (1.0, 99.0)
    histogram_bins:             int                          = 512
    histogram_range:            Tuple[int, int]              = (0, 65535)
    mosaic_padding:             int                          = 4
    mosaic_flip_y:              Optional[bool]               = None   # None = auto from HAL config
    fov_subset:                 Optional[List[int]]          = None   # None = all FOVs

    # ── Data transfer ──────────────────────────────────────────────────────────
    transfer_dest:      Optional[Path]  = None    # network destination root; None = no transfer
    transfer_min_time:  float           = 600.0   # min seconds remaining in fluidics window to start transfer

    # ── Analysis scheduling ──────────────────────────────────────────────────────
    # Analysis now runs CONTINUOUSLY (during acquisition and fluidics), not only in
    # the fluidics window.  Two modes control where it reads image data from:
    #   "same_drive"  (mode B): analyse straight from data_dir on the acquisition
    #                  drive, during both phases. Simplest; analysis I/O shares the
    #                  microscope drive (possible contention on slow HDDs).
    #   "mirror_drive" (mode A): during fluidics, incrementally mirror data_dir to
    #                  analysis_source_dir on a second drive; analyse continuously
    #                  from that mirror, so analysis I/O never touches the
    #                  acquisition drive while the microscope is writing.
    analysis_mode:        str            = "same_drive"
    analysis_source_dir:  Optional[Path] = None   # mode A: second-drive mirror to analyse from
    n_analysis_workers:   Optional[int]  = None   # FOV process-pool size; None → cpu_count - 2

    # ── Derived properties ─────────────────────────────────────────────────────

    @property
    def step_size_um(self) -> float:
        """Stage step size in µm, derived from pixel/FOV parameters."""
        return self.pixel_size_um * self.image_size_px * self.non_overlap_fraction

    @property
    def analysis_data_dir(self) -> Path:
        """Directory the FOV scheduler discovers and reads image files from.

        ``data_dir`` in same-drive mode; ``analysis_source_dir`` (the second-drive
        mirror) in mirror mode.
        """
        if self.analysis_mode == "mirror_drive":
            if self.analysis_source_dir is None:
                raise ValueError(
                    "analysis_mode='mirror_drive' requires analysis_source_dir to be set."
                )
            return self.analysis_source_dir
        return self.data_dir

    @property
    def resolved_n_workers(self) -> int:
        """Number of FOV worker processes to use (>= 1)."""
        if self.n_analysis_workers is not None:
            return max(1, int(self.n_analysis_workers))
        import os
        return max(1, (os.cpu_count() or 2) - 2)

    # ── Initialisation ─────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        for attr in (
            "data_dir", "metadata_dir", "analysis_dir",
            "round_info_csv", "positions_txt",
        ):
            setattr(self, attr, Path(getattr(self, attr)))

        if self.settings_dir is not None:
            self.settings_dir = Path(self.settings_dir)
        if self.hal_templates_dir is not None:
            self.hal_templates_dir = Path(self.hal_templates_dir)
        if self.transfer_dest is not None:
            self.transfer_dest = Path(self.transfer_dest)
        if self.analysis_source_dir is not None:
            self.analysis_source_dir = Path(self.analysis_source_dir)

        if self.analysis_mode not in ("same_drive", "mirror_drive"):
            raise ValueError(
                f"analysis_mode must be 'same_drive' or 'mirror_drive', "
                f"got {self.analysis_mode!r}"
            )
        if self.analysis_mode == "mirror_drive" and self.analysis_source_dir is None:
            raise ValueError(
                "analysis_mode='mirror_drive' requires analysis_source_dir "
                "(a directory on a second drive to mirror data into and analyse from)."
            )

        if self.t_max is None:
            self.t_max = (
                T_MAX_ADAPTOR if self.fluidics_type == "adaptor" else T_MAX_DIRECT
            )

        for sub in ("thumbnails", "stats", "histograms", "mosaics", "done", "logs"):
            (self.analysis_dir / sub).mkdir(parents=True, exist_ok=True)