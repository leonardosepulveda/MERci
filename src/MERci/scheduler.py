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
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .common.config   import ExperimentConfig
from .common.metadata import ExperimentMetadata
from .common.io       import discover_image_files
from .transfer        import transfer_round, mirror_tree, sync_files
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


# ── Shared, dependency-free task builders ────────────────────────────────────
# Pulled out of FOVScheduler/RoundScheduler so a standalone SLURM-array CLI
# script (no scheduler instance, no process pool) can build the exact same
# paths/kwargs and get identical output.

def build_fov_task_kwargs(fpath: Path, config: ExperimentConfig, tracker: ProgressTracker) -> dict:
    """Build the kwargs ``analyze_file(fpath, **kwargs)`` needs for *fpath*."""
    return dict(
        thumbnails_dir            = config.analysis_dir / "thumbnails",
        stats_path                = tracker.stats_path(fpath),
        histogram_path            = tracker.histogram_path(fpath),
        sentinel_path             = tracker.fov_sentinel(fpath),
        frame_width               = config.frame_width,
        frame_height              = config.frame_height,
        thumbnail_frames          = config.thumbnail_frames,
        thumbnail_size            = config.thumbnail_size,
        thumbnail_percentile_clip = config.thumbnail_percentile_clip,
        histogram_bins            = config.histogram_bins,
        histogram_range           = config.histogram_range,
    )


def resolve_round_flip_y(round_id: int, config: ExperimentConfig, metadata: ExperimentMetadata) -> bool:
    """
    Return the flip_y value for *round_id*.

    If ``config.mosaic_flip_y`` is explicitly set (not None), use it.
    Otherwise read ``<flip_vertical>`` from the HAL config associated with
    the first bits-type series in the round.
    """
    if config.mosaic_flip_y is not None:
        return config.mosaic_flip_y

    if config.settings_dir is None:
        return False

    for s in metadata.series_for_round(round_id):
        if s.hal_config:
            hal_path = config.settings_dir / s.hal_config
            if hal_path.exists():
                return read_hal_flip_vertical(hal_path)
    return False


def resolve_round_color_frame_indices(
    round_id: int, config: ExperimentConfig, metadata: ExperimentMetadata,
) -> Dict[float, int]:
    """
    Return {color_nm: frame_idx} for the middle-z slice of *round_id*.

    Falls back to {} (caller substitutes {None: 0}, first frame) when the
    frame table is not found.
    """
    if config.settings_dir is None or config.metadata_dir is None:
        return {}

    for s in metadata.series_for_round(round_id):
        if s.hal_config:
            hal_path = config.settings_dir / s.hal_config
            ft_path  = find_frame_table_for_hal_config(hal_path, config.metadata_dir)
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


def build_round_mosaics(
    round_id: int,
    config:   ExperimentConfig,
    metadata: ExperimentMetadata,
    tracker:  ProgressTracker,
) -> bool:
    """
    Build every per-color mosaic for *round_id* from its FOV thumbnails, and
    mark it done if at least one mosaic was built. Returns whether any mosaic
    was built. Shared by ``RoundScheduler`` and the standalone
    ``cli_build_round_mosaic`` SLURM script.
    """
    log.info("Building mosaics for round %d …", round_id)

    flip_y        = resolve_round_flip_y(round_id, config, metadata)
    color_indices = resolve_round_color_frame_indices(round_id, config, metadata)

    # Fallback: one mosaic with frame 0 when no frame table was found
    if not color_indices:
        color_indices = {None: 0}

    thumbnails_dir = config.analysis_dir / "thumbnails"
    any_mosaic_built = False

    for color, frame_idx in color_indices.items():
        thumbnails, positions = load_thumbnails_for_round(
            round_id=round_id,
            metadata=metadata,
            thumbnails_dir=thumbnails_dir,
            frame_idx=frame_idx,
            fov_subset=config.fov_subset,
        )
        if not thumbnails:
            log.warning(
                "No thumbnails for round %d color %s frame %d; skipping.",
                round_id, color, frame_idx,
            )
            continue

        out_path = tracker.mosaic_path(round_id, color)
        create_mosaic(
            thumbnails=thumbnails,
            positions=positions,
            output_path=out_path,
            thumbnail_size=config.thumbnail_size,
            padding=config.mosaic_padding,
            flip_y=flip_y,
        )
        any_mosaic_built = True

    if any_mosaic_built:
        tracker.mark_round_done(round_id)
    return any_mosaic_built


def source_dirs_for_round(round_id: int, metadata: ExperimentMetadata) -> List[Path]:
    """Return the unique data directories that hold files for *round_id*.

    Uses the metadata's resolved paths so a cells round that landed in
    ``data/`` rather than ``data/cells`` (or vice-versa) is transferred
    from its real location."""
    dirs = {f.parent for f in metadata.files_for_round(round_id)}
    return sorted(dirs)


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
        return fpath, build_fov_task_kwargs(fpath, self.config, self.tracker)

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
        """Return the unique data directories that hold files for *round_id*."""
        return source_dirs_for_round(round_id, self.meta)

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
        """Return the flip_y value for *round_id*."""
        return resolve_round_flip_y(round_id, self.config, self.meta)

    def _color_frame_indices(self, round_id: int) -> Dict[float, int]:
        """Return {color_nm: frame_idx} for the middle-z slice of *round_id*."""
        return resolve_round_color_frame_indices(round_id, self.config, self.meta)

    def _analyse_one_round(self, round_id: int) -> None:
        build_round_mosaics(round_id, self.config, self.meta, self.tracker)


# ── Transfer Scheduler ────────────────────────────────────────────────────────

class TransferScheduler:
    """
    Continuously transfers each round's raw data — plus the small metadata/
    positions/settings files a cluster-side ``ExperimentMetadata.load()``
    needs alongside it — to ``config.transfer_dest`` as soon as the round is
    **fully written**, decoupled from any local QC analysis.

    Unlike ``RoundScheduler``'s ``round_robin_drives`` transfer path (gated
    on ``tracker.is_round_done``, i.e. mosaics already built locally), this
    gates purely on ``metadata.round_fully_written`` (every expected raw
    file for the round exists on disk) — so raw data can leave the
    microscope computer immediately, with QC analysis moved entirely to a
    SLURM cluster reading from wherever it lands (see
    ``06_transfer_to_nas.ipynb`` / ``07_cluster_submit_analysis.ipynb``).

    Requires ``config.analysis_mode == "round_robin_drives"`` (the "which
    drive is hot right now" signal this relies on) and ``config.transfer_dest``
    to be set.

    Call ``transfer_progress()`` at any time (e.g. from an ``on_tick``
    callback) for a display-ready snapshot: FOVs arrived vs. total, the
    measured average seconds/FOV from recently completed round transfers,
    and ETAs for the in-progress round and for everything remaining.

    Typical usage (in a notebook cell)::

        TransferScheduler(config, meta, tracker).run_loop()
    """

    def __init__(
        self,
        config:     ExperimentConfig,
        metadata:   ExperimentMetadata,
        tracker:    ProgressTracker,
        sample_dir: Optional[Path] = None,
    ) -> None:
        if config.transfer_dest is None:
            raise ValueError("TransferScheduler requires config.transfer_dest to be set.")
        self.config     = config
        self.meta       = metadata
        self.tracker    = tracker
        # Auxiliary files (round_info.csv, positions_*.txt, settings/*.xml) are
        # synced preserving their path relative to sample_dir; defaults to
        # data_dir's parent, true for every notebook's ExperimentConfig(data_dir=
        # SAMPLE_DIR/"data", ...) convention, but overridable in case it isn't.
        self.sample_dir = Path(sample_dir) if sample_dir is not None else Path(config.data_dir).parent
        self._transfers_in_progress: set = set()   # round_ids currently being transferred
        self._round_start_info: Dict[int, Tuple[float, int]] = {}   # round_id -> (start_time, n_files)
        # Seconds-per-file samples from the last 20 *completed* round transfers,
        # used to estimate remaining time -- a deque so the average tracks
        # recent network conditions rather than being dragged down by a slow
        # transfer from hours ago.
        self._transfer_rate_samples: "deque[float]" = deque(maxlen=20)

    def run_loop(
        self,
        on_tick: Optional[Callable[[dict], None]] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Run the transfer loop indefinitely (or for *max_iterations* ticks).

        Each tick: sync the small metadata/positions/settings files, then
        start a background transfer for any round that is fully written, not
        yet transferred, not already mid-transfer, and not on the currently
        actively-written (hot) drive.
        """
        iteration = 0
        while True:
            n_started = self._process_pending_transfers()
            if on_tick is not None:
                on_tick({"iteration": iteration, "transfers_started": n_started})
            log.info("[tick %d] started %d round transfer(s).", iteration, n_started)

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break

            time.sleep(self.config.poll_interval)

    def _aux_file_paths(self) -> List[Path]:
        """Small, fast-changing files a cluster-side ``ExperimentMetadata.load()``
        needs alongside the (much larger, background-threaded) round image data."""
        paths: List[Path] = []
        if Path(self.config.round_info_csv).exists():
            paths.append(Path(self.config.round_info_csv))
        if self.config.metadata_dir is not None:
            rbc = Path(self.config.metadata_dir) / "round_bit_color_map.csv"
            if rbc.exists():
                paths.append(rbc)
        if self.config.positions_txt is not None:
            paths.extend(sorted(Path(self.config.positions_txt).parent.glob("positions_*.txt")))
        if self.config.settings_dir is not None and Path(self.config.settings_dir).exists():
            paths.extend(sorted(Path(self.config.settings_dir).glob("*.xml")))
        return paths

    def _process_pending_transfers(self) -> int:
        sync_files(self._aux_file_paths(), self.sample_dir, self.config.transfer_dest)

        active_round = self.meta.actively_writing_round()
        active_drive = (
            self.meta.drive_of_round(active_round) if active_round is not None else None
        )

        count = 0
        for rid in self.meta.valid_round_ids():
            if (
                self.meta.round_fully_written(rid)
                and not self.tracker.is_round_transferred(rid)
                and rid not in self._transfers_in_progress
                and (active_drive is None or self.meta.drive_of_round(rid) != active_drive)
            ):
                self._start_transfer_for_round(rid)
                count += 1
        return count

    def _start_transfer_for_round(self, round_id: int) -> None:
        """Launch a background thread to copy round *round_id* to transfer_dest."""
        src_dirs = source_dirs_for_round(round_id, self.meta)
        if not src_dirs:
            log.warning("Round %d: no source dirs found — skipping transfer.", round_id)
            return

        n_files    = len(self.meta.files_for_round(round_id))
        started_at = time.time()
        self._transfers_in_progress.add(round_id)
        self._round_start_info[round_id] = (started_at, n_files)
        log.info(
            "Round %d: starting transfer of %d dir(s) (%d file(s)) to %s.",
            round_id, len(src_dirs), n_files, self.config.transfer_dest,
        )

        def _on_done(success: bool) -> None:
            self._transfers_in_progress.discard(round_id)
            start_time, n = self._round_start_info.pop(round_id, (started_at, n_files))
            if success:
                self.tracker.mark_round_transferred(round_id)
                if n > 0:
                    self._transfer_rate_samples.append((time.time() - start_time) / n)
            else:
                log.error("Round %d: transfer failed — will retry next tick.", round_id)

        transfer_round(src_dirs, self.config.transfer_dest, on_complete=_on_done)

    # ── Progress / ETA queries ────────────────────────────────────────────────

    @property
    def in_progress_rounds(self) -> List[int]:
        """Round ids currently being copied to transfer_dest (background threads)."""
        return sorted(self._transfers_in_progress)

    def count_arrived_files(self, round_id: int) -> int:
        """
        Approximate count of *round_id*'s expected raw files that already
        exist at ``transfer_dest``.

        Existence-based, same caveat as ``ExperimentMetadata.round_fully_written``:
        it doesn't verify an individual file's copy has *finished* (a zarr
        FOV's destination directory can exist before every one of its
        internal chunk files has arrived), just that robocopy has started
        writing it -- good enough for a progress estimate, not a
        completeness guarantee (that's what ``tracker.is_round_transferred``
        is for). Rounds already marked transferred short-circuit to the full
        count, skipping the filesystem check.
        """
        if self.tracker.is_round_transferred(round_id):
            return len(self.meta.files_for_round(round_id))
        dest_root = Path(self.config.transfer_dest)
        return sum(
            1 for f in self.meta.files_for_round(round_id)
            if (dest_root / f.parent.name / f.name).exists()
        )

    def average_seconds_per_fov(self) -> Optional[float]:
        """Mean seconds-per-file from the most recent completed round
        transfers (up to the last 20), or ``None`` if no round transfer has
        completed yet."""
        if not self._transfer_rate_samples:
            return None
        return sum(self._transfer_rate_samples) / len(self._transfer_rate_samples)

    def transfer_progress(self) -> dict:
        """
        Snapshot of transfer progress across the whole experiment, for
        display in a notebook.

        Returns a dict with:
          fovs_arrived / fovs_total     : files landed at transfer_dest vs.
                                          every expected raw file (existence-
                                          based, see ``count_arrived_files``)
          avg_seconds_per_fov           : see ``average_seconds_per_fov``;
                                          ``None`` until a round has finished
          in_progress_rounds            : round ids currently mid-transfer
          in_progress_fovs_remaining    : files not yet arrived, summed over
                                          ``in_progress_rounds``
          eta_in_progress_s             : estimated seconds to finish
                                          whatever's currently in flight, or
                                          ``None`` without a rate yet
          all_remaining_fovs            : every not-yet-arrived file across
                                          ALL rounds (written or not)
          eta_all_remaining_s           : estimated seconds to finish
                                          transferring everything, ASSUMING
                                          every round were already fully
                                          written right now (i.e. this
                                          estimates transfer time only, not
                                          how long HAL still has to image)
        """
        round_ids  = self.meta.valid_round_ids()
        arrived_by_round = {rid: self.count_arrived_files(rid) for rid in round_ids}
        total_by_round   = {rid: len(self.meta.files_for_round(rid)) for rid in round_ids}

        fovs_arrived = sum(arrived_by_round.values())
        fovs_total   = sum(total_by_round.values())
        avg_rate     = self.average_seconds_per_fov()

        in_progress = self.in_progress_rounds
        in_progress_remaining = sum(
            total_by_round[rid] - arrived_by_round[rid] for rid in in_progress
        )
        eta_in_progress = (
            in_progress_remaining * avg_rate
            if avg_rate is not None and in_progress else None
        )

        all_remaining = fovs_total - fovs_arrived
        eta_all = all_remaining * avg_rate if avg_rate is not None else None

        return {
            "fovs_arrived":               fovs_arrived,
            "fovs_total":                 fovs_total,
            "avg_seconds_per_fov":        avg_rate,
            "in_progress_rounds":         in_progress,
            "in_progress_fovs_remaining": in_progress_remaining,
            "eta_in_progress_s":          eta_in_progress,
            "all_remaining_fovs":         all_remaining,
            "eta_all_remaining_s":        eta_all,
        }


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