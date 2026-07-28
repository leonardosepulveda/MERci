# MERci/progress.py
"""
Lightweight, file-system-based analysis progress tracker.

Design principle
----------------
Progress is tracked purely through the presence of output files and
zero-byte sentinel files.  No shared database is needed, which means
multiple notebooks can run concurrently without coordination.

Sentinel files
--------------
FOV-level done:
    <analysis_dir>/done/<stem>.fov_done

Round-level done:
    <analysis_dir>/done/round_<r:03d>.round_done

Round-level transferred:
    <analysis_dir>/done/round_<r:03d>.round_transferred

SLURM submission bookkeeping (cluster-side, see cli_analyze_fov.py /
cli_build_round_mosaic.py / 07_cluster_submit_analysis.ipynb):
    <analysis_dir>/done/round_<r:03d>.fov_submitted            (FOV array job)
    <analysis_dir>/done/round_<r:03d>.round_mosaic_submitted    (mosaic job)
These hold the submitted SLURM job id as their (small) text content, not just
an empty touch -- so a later run can check ``sacct`` to see whether that job
is still PENDING/RUNNING before deciding whether to resubmit.

Flat-field-correction (FFC) done, per color (computed once per experiment,
not per round -- see analysis/ffc.py):
    <analysis_dir>/done/ffc_<color>nm.ffc_done
    <analysis_dir>/ffc/ffc_<color>nm.npz    (the cached field itself)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_FOV_DONE_SUFFIX               = ".fov_done"
_ROUND_DONE_SUFFIX             = ".round_done"
_ROUND_TRANSFERRED_SUFFIX      = ".round_transferred"
_FOV_SUBMITTED_SUFFIX          = ".fov_submitted"
_ROUND_MOSAIC_SUBMITTED_SUFFIX = ".round_mosaic_submitted"
_FFC_DONE_SUFFIX                = ".ffc_done"


def _read_job_id(sentinel: Path) -> Optional[int]:
    """Read a submitted-job-id sentinel's content as an int, or ``None`` if
    the sentinel doesn't exist or holds something unparseable (e.g. an older,
    empty ``.touch()``-only sentinel from before job ids were recorded)."""
    if not sentinel.exists():
        return None
    try:
        return int(sentinel.read_text().strip())
    except ValueError:
        return None


class ProgressTracker:
    """
    All path-generation methods are deterministic so that independent
    processes reach identical conclusions by inspecting the same directory.
    """

    def __init__(self, analysis_dir: Path) -> None:
        self.analysis_dir   = Path(analysis_dir)
        self.thumbnails_dir = self.analysis_dir / "thumbnails"
        self.stats_dir      = self.analysis_dir / "stats"
        self.histograms_dir = self.analysis_dir / "histograms"
        self.mosaics_dir    = self.analysis_dir / "mosaics"
        self.ffc_dir        = self.analysis_dir / "ffc"
        self.done_dir       = self.analysis_dir / "done"
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.ffc_dir.mkdir(parents=True, exist_ok=True)

    # ── Deterministic path helpers ────────────────────────────────────────────

    def thumbnail_path(self, dax_path: Path, frame_idx: int) -> Path:
        return self.thumbnails_dir / f"{Path(dax_path).stem}_frame{frame_idx:03d}.png"

    def stats_path(self, dax_path: Path) -> Path:
        return self.stats_dir / f"{Path(dax_path).stem}_stats.csv"

    def histogram_path(self, dax_path: Path) -> Path:
        return self.histograms_dir / f"{Path(dax_path).stem}_histograms.npz"

    def mosaic_path(self, round_id: int, color: Optional[float] = None) -> Path:
        if color is None:
            return self.mosaics_dir / f"round_{round_id:03d}_mosaic.png"
        return self.mosaics_dir / f"round_{round_id:03d}_{int(color)}nm_mosaic.png"

    def fov_sentinel(self, dax_path: Path) -> Path:
        return self.done_dir / f"{Path(dax_path).stem}{_FOV_DONE_SUFFIX}"

    def round_sentinel(self, round_id: int) -> Path:
        return self.done_dir / f"round_{round_id:03d}{_ROUND_DONE_SUFFIX}"

    def transfer_sentinel(self, round_id: int) -> Path:
        return self.done_dir / f"round_{round_id:03d}{_ROUND_TRANSFERRED_SUFFIX}"

    def fov_submitted_sentinel(self, round_id: int) -> Path:
        return self.done_dir / f"round_{round_id:03d}{_FOV_SUBMITTED_SUFFIX}"

    def round_mosaic_submitted_sentinel(self, round_id: int) -> Path:
        return self.done_dir / f"round_{round_id:03d}{_ROUND_MOSAIC_SUBMITTED_SUFFIX}"

    def ffc_field_path(self, color: float) -> Path:
        return self.ffc_dir / f"ffc_{int(color)}nm.npz"

    def ffc_done_sentinel(self, color: float) -> Path:
        return self.done_dir / f"ffc_{int(color)}nm{_FFC_DONE_SUFFIX}"

    # ── Status queries ────────────────────────────────────────────────────────

    def is_fov_done(self, dax_path: Path) -> bool:
        """True if the FOV-level sentinel exists for this file."""
        return self.fov_sentinel(dax_path).exists()

    def is_round_done(self, round_id: int) -> bool:
        """True if the round-level sentinel exists."""
        return self.round_sentinel(round_id).exists()

    def is_round_transferred(self, round_id: int) -> bool:
        """True if the round's data transfer has completed."""
        return self.transfer_sentinel(round_id).exists()

    def is_fov_analysis_submitted(self, round_id: int) -> bool:
        """True if a SLURM array job was submitted for this round's pending FOVs."""
        return self.fov_submitted_sentinel(round_id).exists()

    def fov_analysis_submitted_job_id(self, round_id: int) -> Optional[int]:
        """The SLURM job id last submitted for this round's FOV analysis, if any."""
        return _read_job_id(self.fov_submitted_sentinel(round_id))

    def is_round_mosaic_submitted(self, round_id: int) -> bool:
        """True if a SLURM job was submitted to build this round's mosaic(s)."""
        return self.round_mosaic_submitted_sentinel(round_id).exists()

    def round_mosaic_submitted_job_id(self, round_id: int) -> Optional[int]:
        """The SLURM job id last submitted for this round's mosaic build, if any."""
        return _read_job_id(self.round_mosaic_submitted_sentinel(round_id))

    def is_thumbnail_done(self, dax_path: Path, frame_idx: int) -> bool:
        return self.thumbnail_path(dax_path, frame_idx).exists()

    def is_stats_done(self, dax_path: Path) -> bool:
        return self.stats_path(dax_path).exists()

    def is_histogram_done(self, dax_path: Path) -> bool:
        return self.histogram_path(dax_path).exists()

    def is_ffc_done(self, color: float) -> bool:
        """True if the FFC field for this color has already been computed
        and cached (once per experiment, not per round)."""
        return self.ffc_done_sentinel(color).exists()

    def all_fovs_done_for_round(
        self,
        round_id:   int,
        metadata,               # ExperimentMetadata
        fov_subset: Optional[List[int]] = None,
    ) -> bool:
        """
        Return True iff every expected image file in *round_id* (optionally
        filtered to *fov_subset* FOV ids) exists on disk and has a FOV sentinel.
        """
        files = metadata.files_for_round(round_id)
        if not files:
            return False
        if fov_subset is not None:
            fov_set = set(fov_subset)
            files = [
                f for f in files
                if metadata.fov_id_of_file(f) in fov_set
            ]
            if not files:
                return False
        return all(f.exists() and self.is_fov_done(f) for f in files)

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def pending_fov_files(self, candidate_files: List[Path]) -> List[Path]:
        """
        Filter *candidate_files* to those that are stable on disk but lack a
        FOV-level sentinel.  Files still being written are not excluded here;
        that check belongs in ``discover_image_files()``.
        """
        return [
            f for f in candidate_files
            if f.exists() and not self.is_fov_done(f)
        ]

    def pending_rounds(
        self,
        round_ids:  List[int],
        metadata,
        fov_subset: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Return round ids where all FOV analyses are complete (within
        *fov_subset* if given) but no round-level sentinel exists yet.
        """
        return [
            rid for rid in round_ids
            if not self.is_round_done(rid)
            and self.all_fovs_done_for_round(rid, metadata, fov_subset)
        ]

    # ── Status updates ────────────────────────────────────────────────────────

    def mark_fov_done(self, dax_path: Path) -> None:
        """Create the FOV-level sentinel (idempotent)."""
        p = self.fov_sentinel(dax_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        log.debug("FOV marked done: %s", Path(dax_path).name)

    def mark_round_done(self, round_id: int) -> None:
        """Create the round-level sentinel (idempotent)."""
        p = self.round_sentinel(round_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        log.info("Round %d marked done.", round_id)

    def mark_round_transferred(self, round_id: int) -> None:
        """Create the transfer sentinel (idempotent)."""
        p = self.transfer_sentinel(round_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        log.info("Round %d marked transferred.", round_id)

    def mark_fov_analysis_submitted(self, round_id: int, job_id: int) -> None:
        """Record *job_id* as the SLURM array job submitted for this round's
        pending FOVs (overwrites any previous job id — idempotent resubmission)."""
        p = self.fov_submitted_sentinel(round_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(job_id))
        log.info("Round %d: FOV analysis submitted as job %s.", round_id, job_id)

    def mark_round_mosaic_submitted(self, round_id: int, job_id: int) -> None:
        """Record *job_id* as the SLURM job submitted to build this round's mosaic(s)."""
        p = self.round_mosaic_submitted_sentinel(round_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(job_id))
        log.info("Round %d: mosaic build submitted as job %s.", round_id, job_id)

    def mark_ffc_done(self, color: float) -> None:
        """Create the FFC-done sentinel for this color (idempotent)."""
        p = self.ffc_done_sentinel(color)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        log.info("FFC field for %snm marked done.", color)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, metadata) -> dict:
        """Return a progress summary dict suitable for display in a notebook."""
        all_files    = metadata.all_expected_files()
        valid_rounds = metadata.valid_round_ids()

        n_files      = len(all_files)
        n_fov_done   = sum(1 for f in all_files if self.is_fov_done(f))
        n_rounds     = len(valid_rounds)
        n_round_done = sum(1 for r in valid_rounds if self.is_round_done(r))

        return {
            "files_total":    n_files,
            "files_fov_done": n_fov_done,
            "files_pending":  n_files - n_fov_done,
            "rounds_total":   n_rounds,
            "rounds_done":    n_round_done,
            "rounds_pending": n_rounds - n_round_done,
        }