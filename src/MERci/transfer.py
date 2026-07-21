# MERci/transfer.py
"""
Background data-transfer utilities.

Copies one or more source directories to a destination root (e.g. a network
share) in a daemon thread so that the scheduler loop is not blocked.

On Windows, ``robocopy`` is used (/Z restartable mode, retries on failure).
On other platforms, ``shutil.copytree`` is used as a fallback.

Exit codes
----------
robocopy returns 0–7 for informational outcomes (0 = no change, 1 = files
copied, …).  Codes ≥ 8 indicate errors.  We treat 0–7 as success.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger(__name__)


def relative_to_data_root(path: Path) -> Path:
    """
    Return *path*'s sub-path starting at its ``data`` directory component,
    e.g. ``D:/Leonardo/sample/merfish/data/hybs/H01`` -> ``data/hybs/H01``.

    Strips whatever absolute prefix precedes ``data/...`` so a round lands
    at the same logical location under the destination root regardless of
    which local path it was acquired to. Falls back to just *path*'s own
    name (the previous, flatter behaviour) if no ``data`` component is
    found, so a transfer still proceeds either way.
    """
    parts = Path(path).parts
    lower = [p.lower() for p in parts]
    if "data" in lower:
        return Path(*parts[lower.index("data"):])
    log.warning(
        "No 'data' directory component found in %s -- transferring to "
        "dest_root/%s instead of a nested data/... path.",
        path, Path(path).name,
    )
    return Path(Path(path).name)


def _copy_robocopy(src: Path, dst: Path) -> bool:
    """Copy *src* directory to *dst* using robocopy. Returns True on success."""
    dst.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "robocopy", str(src), str(dst),
            "/E",       # include subdirectories (needed for zarr stores)
            "/Z",       # restartable mode — safe to interrupt and resume
            "/R:3",     # retry failed files 3 times
            "/W:5",     # wait 5 s between retries
            "/NP",      # suppress per-file progress output
            "/NDL",     # suppress directory listing
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode >= 8:
        log.error(
            "robocopy failed (exit %d) for %s → %s:\n%s",
            result.returncode, src, dst, result.stdout,
        )
        return False
    log.debug("robocopy exit %d for %s", result.returncode, src.name)
    return True


def _copy_shutil(src: Path, dst: Path) -> bool:
    """Copy *src* directory to *dst* using shutil. Returns True on success."""
    try:
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        return True
    except Exception as exc:
        log.error("Transfer failed %s → %s: %s", src, dst, exc)
        return False


def mirror_dir_sync(src: Path, dst: Path) -> bool:
    """
    Mirror directory *src* into *dst* synchronously (blocks the calling
    thread until done) — for one-off, run-once-and-watch-it-finish syncs
    (e.g. ``data/mosaic10x``, or the static ``MERci``/``merlin``/``fishtank``
    folders) where a notebook cell wants the result before moving on, unlike
    :func:`mirror_tree`/:func:`transfer_round`'s background-threaded copies
    meant to not block a polling tick loop. Safe to re-run — additive/
    incremental via the same ``robocopy /E /Z`` (or ``shutil.copytree``)
    used everywhere else in this module.
    """
    use_robocopy = platform.system() == "Windows"
    copy_fn = _copy_robocopy if use_robocopy else _copy_shutil
    return copy_fn(Path(src), Path(dst))


def mirror_tree(
    src:         Path,
    dst:         Path,
    on_complete: Optional[Callable[[bool], None]] = None,
) -> threading.Thread:
    """
    Incrementally mirror the *contents* of directory ``src`` into ``dst`` in a
    background daemon thread (additive — never deletes files already in ``dst``).

    Used by the "mirror_drive" analysis mode: during fluidics, the acquisition
    drive is copied to a second drive so analysis can read from the copy without
    contending with the microscope's writes.  ``robocopy /E /Z`` skips files that
    are already up to date, so repeated calls only copy what is new.

    Parameters
    ----------
    src         : source directory (the acquisition ``data_dir``)
    dst         : destination directory (the second-drive mirror)
    on_complete : called with ``True`` on success, ``False`` on any failure

    Returns
    -------
    The started :class:`threading.Thread` (daemon=True).
    """
    use_robocopy = platform.system() == "Windows"
    copy_fn = _copy_robocopy if use_robocopy else _copy_shutil

    def _run() -> None:
        ok = copy_fn(Path(src), Path(dst))
        if on_complete is not None:
            on_complete(ok)

    t = threading.Thread(target=_run, daemon=True, name=f"mirror-{Path(src).name}")
    t.start()
    return t


def transfer_round(
    source_dirs:  List[Path],
    dest_root:    Path,
    on_complete:  Optional[Callable[[bool], None]] = None,
) -> threading.Thread:
    """
    Copy each directory in *source_dirs* to ``dest_root / relative_to_data_root(dir)``
    (e.g. ``dest_root/data/hybs/H01``) in a background daemon thread.

    Parameters
    ----------
    source_dirs : source directories to copy (typically the per-round data dir)
    dest_root   : destination root, e.g. ``Path(r"\\\\NAS\\experiments")``
    on_complete : called with ``True`` on full success, ``False`` on any failure

    Returns
    -------
    The started :class:`threading.Thread` (daemon=True).
    """
    use_robocopy = platform.system() == "Windows"
    copy_fn = _copy_robocopy if use_robocopy else _copy_shutil

    label = source_dirs[0].name if source_dirs else "empty"

    def _run() -> None:
        overall_ok = True
        for src in source_dirs:
            dst = dest_root / relative_to_data_root(src)
            log.info("Transfer: %s  →  %s", src, dst)
            ok = copy_fn(src, dst)
            if not ok:
                overall_ok = False
        if on_complete is not None:
            on_complete(overall_ok)

    t = threading.Thread(target=_run, daemon=True, name=f"transfer-{label}")
    t.start()
    return t
