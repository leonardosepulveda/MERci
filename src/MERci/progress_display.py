# MERci/progress_display.py
"""
Reusable, dependency-free progress/ETA reporting for any long-running
per-item loop (analyzing FOVs, computing histograms, transferring files, ...).

Distinct from ``progress.py``'s ``ProgressTracker``, which tracks completion
via on-disk sentinel files across separate process runs -- this is purely a
live, in-process console/notebook display for a loop that's actively running
right now, with no persistence.

Usage
-----
Wrap an iterable directly (simplest)::

    for item in ProgressReporter(len(items), "Computing histograms").wrap(items):
        ... do work ...

Or drive it manually when the work doesn't fit a plain ``for`` loop::

    reporter = ProgressReporter(len(items), "Computing histograms")
    for item in items:
        ... do work ...
        reporter.update()
    reporter.done()
"""
from __future__ import annotations

import time
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def format_duration(seconds: Optional[float]) -> str:
    """Human-readable duration (e.g. ``"2m 31s"``), or ``"n/a"`` for ``None``."""
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


class ProgressReporter:
    """
    Prints an in-place-updating (``\\r``) status line: n/total, percent, a
    simple text bar, elapsed time, and an ETA extrapolated from the average
    per-item rate seen so far. Works the same in a Jupyter cell or a plain
    terminal -- no IPython/notebook-specific dependency.

    Parameters
    ----------
    total          : total number of items expected. ``0`` degrades to a
                     bare running count with no bar/percentage/ETA (useful
                     when the total isn't known up front).
    label          : short prefix describing what's being processed.
    min_interval_s : minimum seconds between printed updates, so a fast
                     inner loop doesn't spend more time printing than
                     working; the final update always prints regardless.
    """

    def __init__(self, total: int, label: str = "", min_interval_s: float = 0.5) -> None:
        self.total          = total
        self.label           = label
        self.min_interval_s  = min_interval_s
        self.n_done          = 0
        self._start_time     = time.time()
        self._last_print     = 0.0

    def update(self, n: int = 1) -> None:
        """Record *n* more completed items and (rate-limited) reprint the status line."""
        self.n_done += n
        now = time.time()
        if now - self._last_print >= self.min_interval_s or self.n_done >= self.total:
            self._print(now)
            self._last_print = now

    def done(self) -> None:
        """Force a final print and move to a fresh line. Call once, after the loop."""
        self._print(time.time())
        print()

    def wrap(self, iterable: Iterable[T]) -> Iterator[T]:
        """Yield every item from *iterable*, calling ``update()`` once per item,
        and ``done()`` once the iterable is exhausted."""
        for item in iterable:
            yield item
            self.update()
        self.done()

    def _print(self, now: float) -> None:
        elapsed = now - self._start_time
        rate    = self.n_done / elapsed if elapsed > 0 else 0.0

        prefix = f"{self.label}: " if self.label else ""
        if self.total > 0:
            pct       = 100 * self.n_done / self.total
            remaining = self.total - self.n_done
            eta       = remaining / rate if rate > 0 else None
            bar_width = 24
            filled    = int(bar_width * pct / 100)
            bar       = "#" * filled + "-" * (bar_width - filled)
            msg = (
                f"{prefix}[{bar}] {self.n_done}/{self.total} ({pct:5.1f}%)  "
                f"elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}"
            )
        else:
            msg = f"{prefix}{self.n_done} done  elapsed {format_duration(elapsed)}"

        # Trailing spaces overwrite any leftover tail from a longer previous line.
        print("\r" + msg + " " * 10, end="", flush=True)
