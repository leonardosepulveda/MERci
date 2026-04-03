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
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_FOV_DONE_SUFFIX   = ".fov_done"
_ROUND_DONE_SUFFIX = ".round_done"


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
        self.done_dir       = self.analysis_dir / "done"
        self.done_dir.mkdir(parents=True, exist_ok=True)

    # ── Deterministic path helpers ────────────────────────────────────────────

    def thumbnail_path(self, dax_path: Path, frame_idx: int) -> Path:
        return self.thumbnails_dir / f"{Path(dax_path).stem}_frame{frame_idx:03d}.png"

    def stats_path(self, dax_path: Path) -> Path:
        return self.stats_dir / f"{Path(dax_path).stem}_stats.csv"

    def histogram_path(self, dax_path: Path) -> Path:
        return self.histograms_dir / f"{Path(dax_path).stem}_histograms.npz"

    def mosaic_path(self, round_id: int) -> Path:
        return self.mosaics_dir / f"round_{round_id:03d}_mosaic.png"

    def fov_sentinel(self, dax_path: Path) -> Path:
        return self.done_dir / f"{Path(dax_path).stem}{_FOV_DONE_SUFFIX}"

    def round_sentinel(self, round_id: int) -> Path:
        return self.done_dir / f"round_{round_id:03d}{_ROUND_DONE_SUFFIX}"

    # ── Status queries ────────────────────────────────────────────────────────

    def is_fov_done(self, dax_path: Path) -> bool:
        """True if the FOV-level sentinel exists for this file."""
        return self.fov_sentinel(dax_path).exists()

    def is_round_done(self, round_id: int) -> bool:
        """True if the round-level sentinel exists."""
        return self.round_sentinel(round_id).exists()

    def is_thumbnail_done(self, dax_path: Path, frame_idx: int) -> bool:
        return self.thumbnail_path(dax_path, frame_idx).exists()

    def is_stats_done(self, dax_path: Path) -> bool:
        return self.stats_path(dax_path).exists()

    def is_histogram_done(self, dax_path: Path) -> bool:
        return self.histogram_path(dax_path).exists()

    def all_fovs_done_for_round(
        self,
        round_id: int,
        metadata,           # ExperimentMetadata
    ) -> bool:
        """
        Return True iff *every* expected image file in *round_id* exists on
        disk and has a FOV-level sentinel.
        """
        files = metadata.files_for_round(round_id)
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
        round_ids: List[int],
        metadata,
    ) -> List[int]:
        """
        Return round ids where all FOV analyses are complete but no
        round-level sentinel exists yet.
        """
        return [
            rid for rid in round_ids
            if not self.is_round_done(rid)
            and self.all_fovs_done_for_round(rid, metadata)
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