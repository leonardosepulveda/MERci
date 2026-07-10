# MERci/acquisition/state.py
"""
Determine whether the microscope is currently imaging or idle, and compute
the time elapsed since the last imaging round ended.

Strategy
--------
We watch the data directory for new image files.  If the most-recently
written file is older than ``config.imaging_idle_threshold`` seconds, the
microscope is considered idle (fluidics / waiting).  The analysis window
opens ``t_min`` seconds after imaging stops and closes at ``t_max`` seconds.

                   Imaging         Fluidics (~60 min)
  ════════════════╪═══════════════════════════════════════════╪═══ ...
                  ▲               ▲               ▲           ▲
               last file       t_min            t_max      next round
                               (start)          (stop)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .common.config import ExperimentConfig

log = logging.getLogger(__name__)


@dataclass
class ExperimentPhase:
    """Snapshot of the experiment state at one point in time."""
    timestamp: float                        # unix time of this snapshot
    is_imaging: bool                        # True while files are being written
    latest_file_mtime: Optional[float]      # mtime of newest image file
    time_since_imaging: Optional[float]     # seconds since last file written
    should_analyze: bool                    # True when inside [t_min, t_max]

    @property
    def phase_str(self) -> str:
        if self.is_imaging:
            return "IMAGING"
        if self.time_since_imaging is None:
            return "WAITING_FOR_DATA"
        return "FLUIDICS"

    def __str__(self) -> str:
        tsi = (
            f"{self.time_since_imaging:.0f} s"
            if self.time_since_imaging is not None
            else "n/a"
        )
        return (
            f"ExperimentPhase(phase={self.phase_str}, "
            f"time_since_imaging={tsi}, should_analyze={self.should_analyze})"
        )


class ExperimentStateMonitor:
    """
    Watches the data directory for new image files and reports the current
    experiment phase.

    The file-system scan result is cached for 10 s to avoid hammering slow
    network storage on every call.
    """

    _CACHE_TTL = 10.0  # seconds

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._cached_mtime: Optional[float] = None
        self._cache_ts: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def snapshot(self) -> ExperimentPhase:
        """Return the current experiment phase without blocking."""
        now   = time.time()
        mtime = self._latest_mtime()

        is_imaging = (
            mtime is not None
            and (now - mtime) < self.config.imaging_idle_threshold
        )

        tsi = (now - mtime) if mtime is not None else None

        in_window = (
            tsi is not None
            and not is_imaging
            and self.config.t_min <= tsi <= self.config.t_max
        )

        return ExperimentPhase(
            timestamp=now,
            is_imaging=is_imaging,
            latest_file_mtime=mtime,
            time_since_imaging=tsi,
            should_analyze=in_window,
        )

    def wait_for_analysis_window(
        self,
        poll_interval: float = 30.0,
        on_tick: Optional[Callable[["ExperimentPhase"], None]] = None,
    ) -> ExperimentPhase:
        """
        Block until ``should_analyze`` is True, then return the phase.

        Parameters
        ----------
        poll_interval : seconds between checks
        on_tick       : optional callback called with the phase on every check
                        (use this in notebooks to display a live status line)
        """
        while True:
            phase = self.snapshot()
            if on_tick is not None:
                on_tick(phase)
            if phase.should_analyze:
                return phase
            log.debug("Not in analysis window: %s", phase)
            time.sleep(poll_interval)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _latest_mtime(self) -> Optional[float]:
        """Return mtime of the most recently modified image file (cached)."""
        now = time.time()
        if now - self._cache_ts < self._CACHE_TTL and self._cached_mtime is not None:
            return self._cached_mtime

        suffix = self.config.image_suffix
        # round_robin_drives: bits-round images live on several physical
        # drives, never all under config.data_dir alone — scan every root
        # referenced in round_info.csv (config.all_data_roots), or phase
        # detection would get stuck reporting WAITING_FOR_DATA the moment
        # the cells round finishes.
        roots = (
            self.config.all_data_roots
            if self.config.analysis_mode == "round_robin_drives"
            else [self.config.data_dir]
        )
        files: list = []
        for root in roots:
            try:
                files.extend(root.rglob(f"*{suffix}"))
            except OSError:
                log.warning("Could not scan %s (disk unreachable?) — skipping.", root)
        if not files:
            return None

        mtime = max(f.stat().st_mtime for f in files)
        self._cached_mtime = mtime
        self._cache_ts     = now
        return mtime