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

import hashlib
import logging
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


def relative_to_data_root(path: Path) -> Path:
    """
    Return *path*'s sub-path starting at its ``data`` directory component,
    e.g. ``D:/Leonardo/sample/merfish/data/hybs/H01`` -> ``data/hybs/H01``.

    Round-robin-drives rounds live under a per-round drive (``D:``, ``E:``,
    ``F:``, …), each still nested under its own ``.../data/...`` — this
    strips the drive-specific prefix so the round lands at the SAME logical
    location once consolidated onto one destination root, whether it came
    from a round-robin drive or straight from ``SAMPLE_DIR/data/...``. Falls
    back to just *path*'s own name (the previous, flatter behaviour) if no
    ``data`` component is found, so a transfer still proceeds either way.
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


def rewrite_round_info_dirs(round_info_csv: Path, dest_path: Path) -> None:
    """
    Copy *round_info_csv* to *dest_path*, rewriting its ``dir``/``data_dir``
    column (if present) from each round's original absolute, drive-specific
    acquisition path to the same canonical ``data/...`` sub-path
    :func:`transfer_round` copies that round's files to (see
    :func:`relative_to_data_root`) -- so a cluster-side
    ``ExperimentMetadata.load()`` reading this copy resolves paths under
    its OWN ``data_dir`` instead of a microscope-local drive that doesn't
    exist there. The original file -- the microscope's live
    ``round_info.csv``, which must keep absolute per-drive paths for the
    running acquisition -- is left untouched.
    """
    df = pd.read_csv(round_info_csv)
    col = "dir" if "dir" in df.columns else ("data_dir" if "data_dir" in df.columns else None)
    if col is not None:
        df[col] = df[col].apply(
            lambda v: str(relative_to_data_root(Path(v))) if pd.notna(v) else v
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_path, index=False)


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


# ── Per-FOV verified transfer ────────────────────────────────────────────────
#
# Unlike transfer_round() above (one robocopy /E per whole round directory,
# no post-copy verification), these copy and bit-for-bit verify one FOV's
# full associated file set at a time -- the image store itself (.zarr/.dax/
# .tiff) plus every same-stem sidecar (.inf/.off/.power/.xml/...), whatever
# HAL happens to write alongside it, discovered by glob rather than a fixed
# list. Used by TransferScheduler so that (a) a round's "transferred" state
# is only ever true once every one of its FOVs has been read back and
# confirmed identical, byte for byte, to the source -- the bar that has to
# be cleared before it's safe to delete the only copy of raw acquisition
# data, and (b) per-FOV completions and their own durations are available
# immediately (average_seconds_per_fov/ETA no longer need to wait for an
# entire round to finish before showing anything).

def fov_associated_paths(image_path: Path) -> List[Path]:
    """
    Every file/directory in *image_path*'s directory sharing its stem --
    the image store itself (``.zarr``/``.dax``/``.tiff``) plus whatever
    same-stem sidecars HAL wrote alongside it (``.inf``, ``.off``,
    ``.power``, ``.xml``, ...). Discovered by glob rather than a hardcoded
    extension list, so it doesn't need updating if a HAL version adds or
    drops a sidecar type.
    """
    image_path = Path(image_path)
    return sorted(image_path.parent.glob(f"{image_path.stem}.*"))


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of one file's bytes, read in chunks (memory-safe
    for large zarr chunk files)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_copy(src: Path, dst: Path, method: str = "hash") -> bool:
    """
    Verify that *dst* is a copy of *src*. *src* may be a single file or a
    directory (e.g. a zarr store, which is really many chunk files) -- for a
    directory, every file within it, recursively, must exist at *dst* at the
    same relative path (no missing/extra file); which comparison is applied
    per file depends on *method*:

    - ``"hash"`` (default): full SHA-256 comparison -- catches silent
      corruption, not just truncation, but reads every byte of every file
      TWICE (once at the source, once at the destination). Over a network
      share this doubles the slowest leg's I/O and can dominate total
      transfer time for large files -- measure ``transfer_fov``'s
      ``verify_seconds`` before assuming this is affordable at your data rate.
    - ``"size"``: only compares file size (no content read at all) -- much
      faster, but won't catch a corrupted file that happens to keep the
      same size (rare for a truncated/interrupted copy, the realistic
      failure mode for a LAN transfer, but not a byte-flip).
    """
    if method not in ("hash", "size"):
        raise ValueError(f"Unknown verify method: {method!r} (expected 'hash' or 'size')")

    src, dst = Path(src), Path(dst)
    if not dst.exists():
        return False

    if src.is_dir():
        if not dst.is_dir():
            return False
        src_rel = sorted(p.relative_to(src) for p in src.rglob("*") if p.is_file())
        dst_rel = sorted(p.relative_to(dst) for p in dst.rglob("*") if p.is_file())
        if src_rel != dst_rel:
            return False
        for rel in src_rel:
            s, d = src / rel, dst / rel
            if s.stat().st_size != d.stat().st_size:
                return False
            if method == "hash" and _hash_file(s) != _hash_file(d):
                return False
        return True

    if not dst.is_file():
        return False
    if src.stat().st_size != dst.stat().st_size:
        return False
    if method == "hash":
        return _hash_file(src) == _hash_file(dst)
    return True


def copy_fov(image_path: Path, dest_root: Path) -> List[Tuple[Path, Path]]:
    """
    Copy every file/directory associated with one FOV (see
    ``fov_associated_paths``) to *dest_root*, preserving the ``data/...``
    structure (see ``relative_to_data_root``). Returns the ``(src, dst)``
    pairs copied, for the caller to pass to ``verify_copy``.
    """
    dest_root = Path(dest_root)
    use_robocopy = platform.system() == "Windows"
    copy_fn = _copy_robocopy if use_robocopy else _copy_shutil

    pairs: List[Tuple[Path, Path]] = []
    for src in fov_associated_paths(image_path):
        dst = dest_root / relative_to_data_root(src)
        if src.is_dir():
            copy_fn(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
        pairs.append((src, dst))
    return pairs


def transfer_fov(image_path: Path, dest_root: Path, verify_method: str = "hash") -> dict:
    """
    Copy one FOV's full associated file set to *dest_root* and verify it
    (see ``verify_copy``'s *method* parameter for the ``"hash"``/``"size"``
    trade-off). Does not delete the source -- deletion (see
    ``delete_source_tree``) is a round-level decision, only safe once EVERY
    FOV in the round has verified.

    Returns a dict: ``verified`` (bool, True iff every associated file
    verified), ``copy_seconds``/``verify_seconds`` (measured separately, so
    a slow FOV can be diagnosed as copy-bound (e.g. network bandwidth) vs.
    verify-bound (e.g. ``"hash"`` reading everything twice) rather than only
    ever seeing one combined number).
    """
    t0 = time.time()
    pairs = copy_fov(image_path, dest_root)
    t1 = time.time()
    if not pairs:
        log.warning("No associated files found for %s -- nothing to transfer.", image_path)
        return {"verified": False, "copy_seconds": t1 - t0, "verify_seconds": 0.0}
    verified = all(verify_copy(src, dst, method=verify_method) for src, dst in pairs)
    t2 = time.time()
    return {"verified": verified, "copy_seconds": t1 - t0, "verify_seconds": t2 - t1}


def delete_source_tree(path: Path) -> None:
    """
    Delete *path* (file or directory tree) from the source drive.

    Only ever call this after every FOV under *path* has an independently
    verified (``verify_copy``) copy at the destination -- this function
    itself does no verification; the caller is the safety boundary.
    """
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    log.info("Deleted source: %s", path)


def sync_files(
    paths:      List[Path],
    sample_dir: Path,
    dest_root:  Path,
) -> bool:
    """
    Copy each file in *paths* to ``dest_root / path.relative_to(sample_dir)``,
    creating parent directories as needed. Runs synchronously in the calling
    thread (unlike :func:`transfer_round`/:func:`mirror_tree`) -- meant for
    the small, fast-changing metadata files (``round_info.csv``,
    ``round_bit_color_map.csv``, ``positions_*.txt``, ``settings/*.xml``) a
    cluster-side ``ExperimentMetadata.load()`` needs alongside the (much
    larger, background-threaded) round image data, so callers can simply
    re-run this every scheduler tick without spawning a thread per file.

    Parameters
    ----------
    paths      : absolute file paths, each somewhere under *sample_dir*
    sample_dir : experiment root *paths* are relative to
    dest_root  : destination root (e.g. the NAS ``transfer_dest``); the
                 relative path of each file under *sample_dir* is preserved

    Returns
    -------
    True if every file copied successfully (missing source files are
    skipped with a warning, not treated as failures -- some may not exist
    yet, e.g. ``round_bit_color_map.csv`` before notebook 03 has run).
    """
    sample_dir = Path(sample_dir)
    dest_root  = Path(dest_root)
    overall_ok = True
    for path in paths:
        path = Path(path)
        if not path.exists():
            log.warning("sync_files: %s does not exist yet — skipping.", path)
            continue
        try:
            rel = path.relative_to(sample_dir)
        except ValueError:
            log.error("sync_files: %s is not under sample_dir %s — skipping.", path, sample_dir)
            overall_ok = False
            continue
        dst = dest_root / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(dst))
        except Exception as exc:
            log.error("sync_files: failed to copy %s → %s: %s", path, dst, exc)
            overall_ok = False
    return overall_ok
