# MERci/common/config.py
"""
Central configuration dataclass.  One instance is shared by both the
acquisition-planning modules and the online-analysis modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ── Fluidics t_max defaults (seconds) ─────────────────────────────────────────
T_MAX_ADAPTOR = 6000.0   # 100 min — adaptor-based fluidics
T_MAX_DIRECT  = 3000.0   # 50 min  — direct-readout fluidics


def default_n_workers() -> int:
    """
    Usable CPUs minus 2 (>= 1). Counts the CPUs this process may run on
    (``os.sched_getaffinity``: a SLURM job's allocation, not the whole
    node) where available, else ``os.cpu_count()``. Capped at 61 on
    Windows, the most ``ProcessPoolExecutor`` accepts there.
    """
    import os
    import sys
    try:
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:          # not available on Windows/macOS
        n_cpus = os.cpu_count() or 2
    n = max(1, n_cpus - 2)
    return min(n, 61) if sys.platform == "win32" else n


@dataclass
class ExperimentConfig:
    """
    All tuneable parameters for one experiment's acquisition and analysis.

    Acquisition parameters
    ----------------------
    microscope            : microscope identifier, e.g. ``"MF3"``, ``"MF5"``
    pixel_size_um         : camera pixel size in µm; None → the microscope's MERlin JSON
    image_size_px         : number of pixels along one side of a raw frame; None → the
                            microscope's MERlin JSON
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
    pixel_size_um:          Optional[float] = None   # None → the microscope's MERlin JSON value
    image_size_px:          Optional[int]  = None   # None → the microscope's MERlin JSON value
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

    # ── Flat-field correction (FFC) for round mosaics ──────────────────────────
    # Divides each raw FOV frame by a per-channel, per-pixel illumination/vignette
    # profile before assembling a round mosaic, then applies one shared contrast
    # stretch across the whole assembled canvas instead of per-tile -- see
    # analysis/ffc.py. Computed once per experiment per color and cached; does
    # NOT affect the per-FOV analyze_file/create_thumbnail pipeline.
    mosaic_ffc_enabled:              bool                = True
    mosaic_contrast_percentile_clip: Tuple[float, float] = (1.0, 99.0)
    ffc_fov_selection_strategy:      str                 = "exterior_grid"  # "exterior_grid" | "emptiest_stats" | "single_fov_all_frames"
    ffc_connectivity:                str                 = "8"    # "4" or "8" -- only used by "exterior_grid"
    ffc_neighbor_tolerance:          float                = 0.25   # fraction of step_size_um
    ffc_smooth_sigma_px:             float                = 50.0
    ffc_normalize_percentile:        float                = 99.99
    ffc_min_value:                   float                = 0.10   # floor clip before division
    ffc_min_samples:                 int                  = 8      # below this, skip FFC for that color/round (warn, don't fail)
    ffc_emptiest_n_fovs:             int                  = 10      # candidate count for "emptiest_stats" strategy

    # ── Data transfer ──────────────────────────────────────────────────────────
    transfer_dest:      Optional[Path]  = None    # network destination root; None = no transfer
    transfer_min_time:  float           = 600.0   # min seconds remaining in fluidics window to start transfer

    # ── Analysis scheduling ──────────────────────────────────────────────────────
    # Analysis now runs CONTINUOUSLY (during acquisition and fluidics), not only in
    # the fluidics window.  Two modes control where it reads image data from:
    #   "same_drive"  (mode B): analyse straight from data_dir on the acquisition
    #                  drive, during both phases. Simplest; analysis I/O shares the
    #                  microscope drive (possible contention on slow HDDs). Also the
    #                  mode to use when SAMPLE_DIR/data_dir itself is a NAS-mounted
    #                  path (direct-to-NAS acquisition) — there is only one location,
    #                  so no mirroring/round-robin step is needed.
    #   "mirror_drive" (mode A): during fluidics, incrementally mirror data_dir to
    #                  analysis_source_dir on a second drive; analyse continuously
    #                  from that mirror, so analysis I/O never touches the
    #                  acquisition drive while the microscope is writing.
    analysis_mode:        str            = "same_drive"
    analysis_source_dir:  Optional[Path] = None   # mode A: second-drive mirror to analyse from
    n_analysis_workers:   Optional[int]  = None   # FOV process-pool size; None → default_n_workers()

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
        # __post_init__ guarantees analysis_source_dir is set in mirror mode.
        return self.analysis_source_dir if self.analysis_mode == "mirror_drive" else self.data_dir

    @property
    def resolved_n_workers(self) -> int:
        """Number of FOV worker processes to use (>= 1)."""
        if self.n_analysis_workers is not None:
            return max(1, int(self.n_analysis_workers))
        return default_n_workers()

    @classmethod
    def from_sample_dir(cls, sample_dir: Path, **kwargs) -> "ExperimentConfig":
        """
        Config for the standard experiment layout under *sample_dir*:
        ``data/``, ``metadata/``, ``analysis/``, ``settings/`` and
        ``metadata/round_info.csv``. *kwargs* give ``positions_txt`` (required)
        and any other field, and can override these defaults.
        """
        sample_dir = Path(sample_dir)
        defaults = dict(
            data_dir       = sample_dir / "data",
            metadata_dir   = sample_dir / "metadata",
            analysis_dir   = sample_dir / "analysis",
            settings_dir   = sample_dir / "settings",
            round_info_csv = sample_dir / "metadata" / "round_info.csv",
        )
        return cls(**{**defaults, **kwargs})

    # ── Initialisation ─────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        for attr in (
            "data_dir", "metadata_dir", "analysis_dir",
            "round_info_csv", "positions_txt",
        ):
            setattr(self, attr, Path(getattr(self, attr)))

        from ..acquisition.configs import get_camera_frame_size, get_camera_pixel_size_um
        if self.pixel_size_um is None:
            self.pixel_size_um = get_camera_pixel_size_um(self.microscope)
        if self.image_size_px is None:
            self.image_size_px = get_camera_frame_size(self.microscope)[0]

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

        for sub in ("thumbnails", "stats", "histograms", "mosaics", "done", "logs", "ffc"):
            (self.analysis_dir / sub).mkdir(parents=True, exist_ok=True)