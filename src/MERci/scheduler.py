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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .common.config   import ExperimentConfig
from .common.metadata import ExperimentMetadata
from .common.io       import discover_image_files
from .transfer        import transfer_round, mirror_tree
from .progress        import ProgressTracker
from .state           import ExperimentStateMonitor, ExperimentPhase
from .analysis.fov    import analyze_file
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
    Discovers new image files and runs all FOV-level analyses **continuously**
    (during both acquisition and fluidics), processing FOVs **in parallel**
    across a pool of worker processes.

    Where it reads from is set by ``config.analysis_mode``:
      * ``"same_drive"`` — read from ``config.data_dir`` (the acquisition drive).
      * ``"mirror_drive"`` — during fluidics, incrementally mirror ``data_dir`` to
        ``config.analysis_source_dir`` on a second drive, and read from that
        mirror, so analysis I/O never competes with the microscope's writes.
      * ``"round_robin_drives"`` — read directly from every drive referenced in
        ``round_info.csv`` (``config.all_data_roots``), skipping only whichever
        round HAL is actively writing right now (``meta.actively_writing_round``),
        so already-completed rounds on other drives analyse immediately with no
        mirroring step.

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
        self._pool: Optional[ProcessPoolExecutor] = None
        self._mirror_thread = None

    # ── Worker pool lifecycle ───────────────────────────────────────────────────

    def _get_pool(self) -> ProcessPoolExecutor:
        """Lazily create (and reuse) the FOV worker-process pool."""
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.config.resolved_n_workers)
        return self._pool

    def close(self) -> None:
        """Shut the worker pool down. Safe to call more than once."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

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
        2. In ``mirror_drive`` mode, refresh the second-drive mirror while the
           microscope is idle (fluidics).
        3. Analyse all pending files (continuously, in parallel across FOVs).
        4. Sleep for ``config.poll_interval`` seconds.

        Parameters
        ----------
        on_phase_update : called with the current ExperimentPhase every tick;
                          useful for live status output in a Jupyter cell
        max_iterations  : stop after this many ticks (``None`` → run forever)
        """
        iteration = 0
        try:
            while True:
                phase = self.monitor.snapshot()

                if on_phase_update is not None:
                    on_phase_update(phase)

                # Mode A: mirror the acquisition drive to the second drive while the
                # microscope is idle, so analysis reads never contend with writes.
                if self.config.analysis_mode == "mirror_drive" and not phase.is_imaging:
                    self._maybe_mirror()

                n_processed = self._process_pending()
                log.info(
                    "[tick %d] phase=%s — processed %d FOV file(s).",
                    iteration, phase.phase_str, n_processed,
                )

                iteration += 1
                if max_iterations is not None and iteration >= max_iterations:
                    break

                time.sleep(self.config.poll_interval)
        finally:
            self.close()

    # ── Mirror (mode A) ─────────────────────────────────────────────────────────

    def _maybe_mirror(self) -> None:
        """Start an incremental mirror of data_dir → analysis_source_dir unless one
        is already running."""
        if self._mirror_thread is not None and self._mirror_thread.is_alive():
            return
        src = self.config.data_dir
        dst = self.config.analysis_source_dir
        log.info("Mirror (fluidics): %s → %s", src, dst)
        self._mirror_thread = mirror_tree(src, dst)

    # ── Internal processing ───────────────────────────────────────────────────

    def _build_task(self, fpath: Path) -> Tuple[Path, dict]:
        """Build the (image_path, kwargs) pair passed to ``analyze_file``."""
        kwargs = dict(
            thumbnails_dir            = self.config.analysis_dir / "thumbnails",
            stats_path                = self.tracker.stats_path(fpath),
            histogram_path            = self.tracker.histogram_path(fpath),
            sentinel_path             = self.tracker.fov_sentinel(fpath),
            frame_width               = self.config.frame_width,
            frame_height              = self.config.frame_height,
            thumbnail_frames          = self.config.thumbnail_frames,
            thumbnail_size            = self.config.thumbnail_size,
            thumbnail_percentile_clip = self.config.thumbnail_percentile_clip,
            histogram_bins            = self.config.histogram_bins,
            histogram_range           = self.config.histogram_range,
        )
        return fpath, kwargs

    def _discovery_roots(self) -> List[Path]:
        """
        Directories to search for image files.

        ``round_robin_drives`` mode spreads files across every drive named in
        round_info.csv (``config.all_data_roots``) instead of one root; other
        modes keep today's single-root behavior (``config.analysis_data_dir``).
        """
        if self.config.analysis_mode == "round_robin_drives":
            return self.config.all_data_roots
        return [self.config.analysis_data_dir]

    def _process_pending(self) -> int:
        """Discover and analyse all pending FOV files. Returns count processed.

        Reads from ``config.analysis_data_dir`` (the mirror in mirror mode, or
        every drive in ``round_robin_drives`` mode — see ``_discovery_roots``),
        and dispatches one worker process per FOV file (each reads the file
        once and runs every analysis), bounded by ``config.resolved_n_workers``.
        """
        all_files: List[Path] = []
        for root in self._discovery_roots():
            try:
                all_files.extend(
                    discover_image_files(root, self.config.image_suffix)
                )
            except OSError:
                log.warning("Could not scan %s (disk unreachable?) — skipping.", root)
        all_files = sorted(set(all_files))
        pending = self.tracker.pending_fov_files(all_files)

        # round_robin_drives: never analyse the round HAL is actively writing
        # right now — its disk is contended; everything else proceeds as usual.
        if self.config.analysis_mode == "round_robin_drives":
            active_round = self.meta.actively_writing_round()
            active_drive = (
                self.meta.drive_of_round(active_round) if active_round is not None else None
            )
            if active_drive is not None:
                pending = [f for f in pending if Path(f).drive.lower() != active_drive]

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
        if not pending:
            return 0

        tasks = [self._build_task(f) for f in pending]
        n_workers = self.config.resolved_n_workers

        # Serial path (single worker) — simpler, in-process, easier to debug.
        if n_workers == 1:
            count = 0
            for fpath, kwargs in tasks:
                try:
                    analyze_file(fpath, **kwargs)
                    log.info("Done: %s", fpath.name)
                    count += 1
                except Exception:
                    log.exception("Error processing %s", fpath.name)
            return count

        # Parallel path — one worker process per FOV file.
        pool = self._get_pool()
        futures = {
            pool.submit(analyze_file, fpath, **kwargs): fpath
            for fpath, kwargs in tasks
        }
        count = 0
        for fut in as_completed(futures):
            fpath = futures[fut]
            try:
                fut.result()
                log.info("Done: %s", fpath.name)
                count += 1
            except Exception:
                log.exception("Error processing %s", fpath.name)
        return count


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

        Mosaics are built continuously as soon as a round's FOVs are all done.
        Optional NAS transfers (``transfer_dest``) run only during fluidics
        (when the microscope is idle) in ``same_drive``/``mirror_drive`` modes,
        or continuously — skipping only the round on the currently-active drive
        — in ``round_robin_drives`` mode (see ``_process_pending_transfers``).
        """
        iteration = 0
        while True:
            phase = self.monitor.snapshot()

            if on_phase_update is not None:
                on_phase_update(phase)

            n_processed = self._process_pending_rounds(phase)
            log.info("[tick %d] phase=%s — built mosaics for %d round(s).",
                     iteration, phase.phase_str, n_processed)

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
        transferred.

        ``round_robin_drives`` mode: each round's data lives on its own
        physical drive (see ``acquisition.dave.create_round_info(data_drives=…)``),
        so a completed round on an idle drive can transfer to ``transfer_dest``
        (e.g. a NAS) continuously, even while HAL is still imaging a different
        round on a different drive. Gated the same way FOVScheduler defers
        analysis of the hot round: skip only the round whose drive matches
        ``meta.actively_writing_round()``'s drive.

        Other modes (``same_drive``/``mirror_drive``): unchanged — transfers
        read from the single acquisition drive, so they only run once the
        microscope goes idle (fluidics), and only while there's still enough
        time left in the fluidics window (``transfer_min_time``).
        """
        if self.config.analysis_mode == "round_robin_drives":
            active_round = self.meta.actively_writing_round()
            active_drive = (
                self.meta.drive_of_round(active_round) if active_round is not None else None
            )
            for rid in self.meta.valid_round_ids():
                if (
                    self.tracker.is_round_done(rid)
                    and not self.tracker.is_round_transferred(rid)
                    and rid not in self._transfers_in_progress
                    and (active_drive is None or self.meta.drive_of_round(rid) != active_drive)
                ):
                    self._start_transfer_for_round(rid)
            return

        if phase.is_imaging or phase.time_since_imaging is None:
            return
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

    def _start_transfer_for_round(self, round_id: int, time_remaining: Optional[float] = None) -> None:
        """Launch a background thread to copy round *round_id* to transfer_dest."""
        src_dirs = self._source_dirs_for_round(round_id)
        if not src_dirs:
            log.warning("Round %d: no source dirs found — skipping transfer.", round_id)
            return

        self._transfers_in_progress.add(round_id)
        if time_remaining is not None:
            log.info(
                "Round %d: starting transfer of %d dir(s) to %s  (%.0f s remaining).",
                round_id, len(src_dirs), self.config.transfer_dest, time_remaining,
            )
        else:
            log.info(
                "Round %d: starting transfer of %d dir(s) to %s.",
                round_id, len(src_dirs), self.config.transfer_dest,
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