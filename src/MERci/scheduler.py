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
from typing import Callable, Dict, List, Optional

import pandas as pd

from .common.config   import ExperimentConfig
from .common.metadata import ExperimentMetadata
from .common.io       import read_image, discover_image_files
from .transfer        import transfer_round
from .progress        import ProgressTracker
from .state           import ExperimentStateMonitor, ExperimentPhase
from .analysis.fov    import (
    create_thumbnails_for_stack,
    measure_stats,
    get_histogram,
)
from .analysis.round  import create_mosaic, load_thumbnails_for_round
from .acquisition.configs import (
    read_hal_flip_vertical,
    find_frame_table_for_hal_config,
    get_color_frame_indices,
)

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

        # Restrict to the requested FOV subset when specified
        if self.config.fov_subset is not None:
            fov_set = set(self.config.fov_subset)
            pending = [
                f for f in pending
                if self.meta.fov_id_of_file(f) in fov_set
            ]

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
        stack = read_image(
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
        self._transfers_in_progress: set = set()   # round_ids currently being transferred

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
                n_processed = self._process_pending_rounds(phase)
                log.info("[tick %d] Built mosaics for %d round(s).",
                         iteration, n_processed)
            else:
                log.debug("[tick %d] Phase=%s – not in window.",
                          iteration, phase.phase_str)

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break

            time.sleep(self.config.poll_interval)

    def _process_pending_rounds(self, phase: ExperimentPhase) -> int:
        pending = self.tracker.pending_rounds(
            self.meta.valid_round_ids(), self.meta, self.config.fov_subset
        )
        log.info("Pending rounds: %s", pending)
        count = 0
        for rid in pending:
            try:
                self._analyse_one_round(rid)
                count += 1
            except Exception:
                log.exception("Error building mosaic for round %d", rid)

        if self.config.transfer_dest is not None:
            self._process_pending_transfers(phase)

        return count

    # ── Transfer helpers ──────────────────────────────────────────────────────

    def _source_dirs_for_round(self, round_id: int) -> List[Path]:
        """Return the unique data directories that hold files for *round_id*.

        Uses the metadata's resolved paths so a cells round that landed in
        ``data/`` rather than ``data/cells`` (or vice-versa) is transferred
        from its real location."""
        dirs = {f.parent for f in self.meta.files_for_round(round_id)}
        return sorted(dirs)

    def _process_pending_transfers(self, phase: ExperimentPhase) -> None:
        """
        Start background transfers for any rounds that are done but not yet
        transferred, provided there is sufficient time left in the fluidics window.
        """
        time_remaining = self.config.t_max - (phase.time_since_imaging or 0)
        if time_remaining < self.config.transfer_min_time:
            log.info(
                "Data transfer skipped — %.0f s remaining < %.0f s threshold.",
                time_remaining, self.config.transfer_min_time,
            )
            return

        for rid in self.meta.valid_round_ids():
            if (
                self.tracker.is_round_done(rid)
                and not self.tracker.is_round_transferred(rid)
                and rid not in self._transfers_in_progress
            ):
                self._start_transfer_for_round(rid, time_remaining)

    def _start_transfer_for_round(self, round_id: int, time_remaining: float) -> None:
        """Launch a background thread to copy round *round_id* to transfer_dest."""
        src_dirs = self._source_dirs_for_round(round_id)
        if not src_dirs:
            log.warning("Round %d: no source dirs found — skipping transfer.", round_id)
            return

        self._transfers_in_progress.add(round_id)
        log.info(
            "Round %d: starting transfer of %d dir(s) to %s  (%.0f s remaining).",
            round_id, len(src_dirs), self.config.transfer_dest, time_remaining,
        )

        def _on_done(success: bool) -> None:
            self._transfers_in_progress.discard(round_id)
            if success:
                self.tracker.mark_round_transferred(round_id)
            else:
                log.error("Round %d: transfer failed — will retry next tick.", round_id)

        transfer_round(src_dirs, self.config.transfer_dest, on_complete=_on_done)

    # ── Round analysis helpers ────────────────────────────────────────────────

    def _resolve_flip_y(self, round_id: int) -> bool:
        """
        Return the flip_y value for *round_id*.

        If ``config.mosaic_flip_y`` is explicitly set (not None), use it.
        Otherwise read ``<flip_vertical>`` from the HAL config associated with
        the first bits-type series in the round.
        """
        if self.config.mosaic_flip_y is not None:
            return self.config.mosaic_flip_y

        if self.config.settings_dir is None:
            return False

        series_list = self.meta.series_for_round(round_id)
        for s in series_list:
            if s.hal_config:
                hal_path = self.config.settings_dir / s.hal_config
                if hal_path.exists():
                    return read_hal_flip_vertical(hal_path)
        return False

    def _color_frame_indices(self, round_id: int) -> Dict[float, int]:
        """
        Return {color_nm: frame_idx} for the middle-z slice of *round_id*.

        Falls back to {0.0: 0} (first frame) when the frame table is not found.
        """
        if self.config.settings_dir is None or self.config.metadata_dir is None:
            return {}

        series_list = self.meta.series_for_round(round_id)
        for s in series_list:
            if s.hal_config:
                hal_path = self.config.settings_dir / s.hal_config
                ft_path  = find_frame_table_for_hal_config(
                    hal_path, self.config.metadata_dir
                )
                if ft_path is not None:
                    ft = pd.read_csv(ft_path, index_col=0)
                    indices = get_color_frame_indices(ft)
                    if indices:
                        return indices
        log.warning(
            "Could not determine color frame indices for round %d; "
            "falling back to frame 0.", round_id
        )
        return {}

    def _analyse_one_round(self, round_id: int) -> None:
        log.info("Building mosaics for round %d …", round_id)

        flip_y        = self._resolve_flip_y(round_id)
        color_indices = self._color_frame_indices(round_id)

        # Fallback: one mosaic with frame 0 when no frame table was found
        if not color_indices:
            color_indices = {None: 0}

        thumbnails_dir = self.config.analysis_dir / "thumbnails"
        any_mosaic_built = False

        for color, frame_idx in color_indices.items():
            thumbnails, positions = load_thumbnails_for_round(
                round_id=round_id,
                metadata=self.meta,
                thumbnails_dir=thumbnails_dir,
                frame_idx=frame_idx,
                fov_subset=self.config.fov_subset,
            )
            if not thumbnails:
                log.warning(
                    "No thumbnails for round %d color %s frame %d; skipping.",
                    round_id, color, frame_idx,
                )
                continue

            out_path = self.tracker.mosaic_path(round_id, color)
            create_mosaic(
                thumbnails=thumbnails,
                positions=positions,
                output_path=out_path,
                thumbnail_size=self.config.thumbnail_size,
                padding=self.config.mosaic_padding,
                flip_y=flip_y,
            )
            any_mosaic_built = True

        if any_mosaic_built:
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