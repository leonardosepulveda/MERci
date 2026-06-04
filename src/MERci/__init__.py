# MERci/__init__.py
"""
MERci
=====
End-to-end pipeline for MERFISH experiment setup and analysis.

Acquisition modules: MERci.acquisition
Analysis modules:    MERci.analysis
Shared utilities:    MERci.common
"""
from .common.config   import ExperimentConfig
from .common.metadata import ExperimentMetadata, RoundInfo, SeriesInfo, FOVInfo
from .state           import ExperimentStateMonitor, ExperimentPhase
from .progress        import ProgressTracker
from .scheduler       import FOVScheduler, RoundScheduler, ExperimentScheduler
from .analysis.fov    import (
    create_thumbnail,
    create_thumbnails_for_stack,
    measure_stats,
    get_histogram,
    load_stats,
    load_histogram,
)
from .analysis.round  import create_mosaic, load_thumbnails_for_round

__all__ = [
    # Config & Metadata
    "ExperimentConfig",
    "ExperimentMetadata", "RoundInfo", "SeriesInfo", "FOVInfo",
    # Runtime state
    "ExperimentStateMonitor", "ExperimentPhase",
    "ProgressTracker",
    # Schedulers
    "FOVScheduler", "RoundScheduler", "ExperimentScheduler",
    # FOV-level analysis
    "create_thumbnail", "create_thumbnails_for_stack",
    "measure_stats", "get_histogram", "load_stats", "load_histogram",
    # Round-level analysis
    "create_mosaic", "load_thumbnails_for_round",
]
