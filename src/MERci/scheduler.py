# MERci/scheduler.py
"""
High-level scheduling logic for the three notebook types.

FOVScheduler
    → continuously processes new image files during fluidics windows

RoundScheduler
    → monitors FOV progress; creates mosaics as soon as each round is complete

ExperimentScheduler
    → blocks until all rounds are done, then runs an experiment-level callback
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

from .common.config   import ExperimentConfig
from .common.metadata import ExperimentMetadata
from .common.io       import read_dax, discover_image_files
from .progress        import ProgressTracker
from .state           import ExperimentStateMonitor, ExperimentPhase
from .analysis.fov    import (
    create_thumbnails_for_stack,
    measure_stats,
    get_histogram,
)
from .analysis.round    import create_mosaic, load_thumbnails_for_round

log = logging.getLogger(__name__)


# ── FOV Scheduler ─────────────────────────────────────────────────────────────

class FOVScheduler:
    """
    Discovers new image files and runs all FOV-level analyses during the
    [t_min, t_max] fluidics window.

    Typical usage (in a notebook cell)::

        scheduler = FOVScheduler(config, meta, tracker, monitor)
        scheduler.run_loop(on_phase_update=display_status)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
        tracker: ProgressTracker,
        monitor: ExperimentStateMonitor,
    ) -> None:
        self.config  = config
        self.meta    = metadata
        self.tracker = tracker
        self.monitor = monitor

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_loop(
        self,
        on_phase_update: Optional[Callable[[ExperimentPhase], None]] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Run the FOV analysis loop indefinitely (or for *max_iterations* ticks).

        On each tick:
        1. Sample the experiment phase.
        2. If ``should_analyze``, scan for pending files and process them.
        3. Sleep for ``config.poll_interval`` seconds.

        Parameters
        ----------
        on_phase_update : called with the current ExperimentPhase every tick;
                          useful for live status output in a Jupyter cell
        max_iterations  : stop after this many ticks (``None`` → run forever)
        """
        iteration = 0
        while True:
            phase = self.monitor.snapshot()

            if on_phase_update is not None:
                on_phase_update(phase)

            if phase.should_analyze:
                n_processed = self._process_pending()
                log.info(
                    "[tick %d] Processed %d FOV file(s). "
                    "Time since imaging: %.0f s.",
                    iteration, n_processed, phase.time_since_imaging or 0,
                )
            else:
                log.debug(
                    "[tick %d] Phase=%s (tsi=%.0f s) – not in window.",
                    iteration, phase.phase_str, phase.time_since_imaging or -1,
                )

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break

            time.sleep(self.config.poll_interval)

    # ── Internal processing ───────────────────────────────────────────────────

    def _process_pending(self) -> int:
        """Discover and analyse all pending FOV files. Returns count processed."""
        all_files = discover_image_files(
            self.config.data_dir, self.config.image_suffix
        )
        pending = self.tracker.pending_fov_files(all_files)

        log.info(
            "Pending: %d of %d image files need FOV analysis.",
            len(pending), len(all_files),
        )

        count = 0
        for fpath in pending:
            try:
                self._analyse_one_file(fpath)
                count += 1
            except Exception:
                log.exception("Error processing %s", fpath.name)
        return count

    def _analyse_one_file(self, fpath: Path) -> None:
        """
        Run thumbnail + stats + histogram for one image file.
        Writes a FOV sentinel when every output file exists.
        """
        if self.meta.series_of_file(fpath) is None:
            log.warning("'%s' is not in imaging_info.csv – processing anyway.",
                        fpath.name)

        log.info("Analysing: %s", fpath.name)
        stem = fpath.stem

        # ── Read image once ───────────────────────────────────────────────────
        stack = read_dax(
            fpath,
            frame_width=self.config.frame_width,
            frame_height=self.config.frame_height,
        )
        n_frames = len(stack)
        frames   = self.config.thumbnail_frames or list(range(n_frames))

        try:
            # ── Thumbnails ────────────────────────────────────────────────────
            create_thumbnails_for_stack(
                stack,
                stem=stem,
                output_dir=self.config.analysis_dir / "thumbnails",
                frame_indices=frames,
                target_size=self.config.thumbnail_size,
                percentile_clip=self.config.thumbnail_percentile_clip,
            )

            # ── Stats ─────────────────────────────────────────────────────────
            stats_out = self.tracker.stats_path(fpath)
            if not stats_out.exists():
                measure_stats(stack, stats_out, source_filename=fpath.name)

            # ── Histogram ─────────────────────────────────────────────────────
            hist_out = self.tracker.histogram_path(fpath)
            if not hist_out.exists():
                get_histogram(
                    stack, hist_out,
                    bins=self.config.histogram_bins,
                    hist_range=self.config.histogram_range,
                )
        finally:
            del stack   # release ~200 MB per file promptly

        # Mark complete (written only after all outputs exist)
        self.tracker.mark_fov_done(fpath)
        log.info("Done: %s", fpath.name)


# ── Round Scheduler ───────────────────────────────────────────────────────────

class RoundScheduler:
    """
    Monitors FOV-level sentinel files and builds a round mosaic as soon as all
    FOVs in a round are done.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
        tracker: ProgressTracker,
        monitor: ExperimentStateMonitor,
    ) -> None:
        self.config  = config
        self.meta    = metadata
        self.tracker = tracker
        self.monitor = monitor

    def run_loop(
        self,
        on_phase_update: Optional[Callable[[ExperimentPhase], None]] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Run the round analysis loop (same tick-sleep structure as FOVScheduler).
        """
        iteration = 0
        while True:
            phase = self.monitor.snapshot()

            if on_phase_update is not None:
                on_phase_update(phase)

            if phase.should_analyze:
                n_processed = self._process_pending_rounds()
                log.info("[tick %d] Built mosaics for %d round(s).",
                         iteration, n_processed)
            else:
                log.debug("[tick %d] Phase=%s – not in window.",
                          iteration, phase.phase_str)

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break

            time.sleep(self.config.poll_interval)

    def _process_pending_rounds(self) -> int:
        pending = self.tracker.pending_rounds(
            self.meta.valid_round_ids(), self.meta
        )
        log.info("Pending rounds: %s", pending)
        count = 0
        for rid in pending:
            try:
                self._analyse_one_round(rid)
                count += 1
            except Exception:
                log.exception("Error building mosaic for round %d", rid)
        return count

    def _analyse_one_round(self, round_id: int) -> None:
        log.info("Building mosaic for round %d …", round_id)
        thumbnails, positions = load_thumbnails_for_round(
            round_id=round_id,
            metadata=self.meta,
            thumbnails_dir=self.config.analysis_dir / "thumbnails",
            frame_idx=self.config.mosaic_frame_idx,
        )
        if not thumbnails:
            log.warning("No thumbnails found for round %d; skipping.", round_id)
            return

        create_mosaic(
            thumbnails=thumbnails,
            positions=positions,
            output_path=self.tracker.mosaic_path(round_id),
            thumbnail_size=self.config.thumbnail_size,
            padding=self.config.mosaic_padding,
            flip_y=self.config.mosaic_flip_y,
        )
        self.tracker.mark_round_done(round_id)


# ── Experiment Scheduler ──────────────────────────────────────────────────────

class ExperimentScheduler:
    """
    Waits until all round-level analyses are complete, then triggers a
    user-supplied experiment-level analysis function.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        metadata: ExperimentMetadata,
        tracker: ProgressTracker,
    ) -> None:
        self.config  = config
        self.meta    = metadata
        self.tracker = tracker

    def all_rounds_complete(self) -> bool:
        return all(
            self.tracker.is_round_done(rid)
            for rid in self.meta.valid_round_ids()
        )

    def wait_and_run(
        self,
        experiment_fn: Callable[["ExperimentConfig", "ExperimentMetadata"], None],
        poll_interval: float = 120.0,
        on_tick: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """
        Block until all rounds are done, then call
        ``experiment_fn(config, metadata)``.

        Parameters
        ----------
        experiment_fn : your experiment-level analysis function
        poll_interval : seconds between progress checks
        on_tick       : called with the current summary dict on every check
        """
        log.info("Experiment scheduler started; polling every %.0f s.", poll_interval)
        while not self.all_rounds_complete():
            summary = self.tracker.summary(self.meta)
            log.info(
                "Waiting: rounds %d/%d done, files %d/%d done.",
                summary["rounds_done"], summary["rounds_total"],
                summary["files_fov_done"], summary["files_total"],
            )
            if on_tick is not None:
                on_tick(summary)
            time.sleep(poll_interval)

        log.info("All rounds complete → running experiment-level analysis.")
        experiment_fn(self.config, self.meta)