"""
Disk-usage auditing for shared microscope-computer drives.

Scans a configurable list of drive/folder roots that each hold a
``{root}/{lab_member}/{sample_dir}`` layout, measures the on-disk size and file
timestamps of every ``sample_dir``, and returns one row per sample as a
DataFrame -- the input to ``notebooks/misc/audit_disk_usage.ipynb``, which
sorts it by age and by size to help decide whose data to ask to be deleted.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple, Union

import pandas as pd


def _iter_files_recursive(path: Path) -> Iterator[os.DirEntry]:
    """
    Yield every regular file under `path`, recursing into subdirectories.

    Symlinks/junctions are not followed, to avoid infinite loops on reparse
    points (common on Windows network-attached storage). A subdirectory that
    raises a permission/OS error is skipped rather than aborting the whole
    walk, since a shared drive commonly has some folders another user has
    locked down.
    """
    try:
        with os.scandir(path) as it:
            entries = list(it)
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                yield from _iter_files_recursive(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield entry
        except OSError:
            continue


def measure_folder(path: Path) -> dict:
    """
    Recursively measure a folder's total size and file timestamp range.

    Parameters
    ----------
    path : Path
        Folder to measure.

    Returns
    -------
    dict with keys:
        size_bytes : int
            Sum of every file's size under `path`, in bytes.
        n_files : int
            Number of files counted.
        created : datetime or None
            The folder's own creation time (``st_ctime``, which on Windows is
            genuinely creation time, not the Linux "metadata changed" meaning).
            None if `path` itself is inaccessible.
        earliest_file_modified, latest_file_modified : datetime or None
            Oldest/newest file `mtime` found in the tree -- computed for free
            in the same walk used for size, and a useful cross-check on
            `created` (e.g. if the top folder was touched/renamed after the
            data was copied in).
    """
    size_bytes = 0
    n_files = 0
    earliest = None
    latest = None
    for entry in _iter_files_recursive(path):
        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        size_bytes += stat.st_size
        n_files += 1
        mtime = stat.st_mtime
        if earliest is None or mtime < earliest:
            earliest = mtime
        if latest is None or mtime > latest:
            latest = mtime

    try:
        created = datetime.fromtimestamp(Path(path).stat().st_ctime)
    except OSError:
        created = None

    return {
        "size_bytes": size_bytes,
        "n_files": n_files,
        "created": created,
        "earliest_file_modified": datetime.fromtimestamp(earliest) if earliest else None,
        "latest_file_modified": datetime.fromtimestamp(latest) if latest else None,
    }


# Windows-reserved folders present on every drive (e.g. the recycle bin, the
# volume-shadow-copy store) that even an administrator normally cannot list --
# not real lab-member data, so skip them by name rather than letting the
# PermissionError they raise abort the whole scan.
_IGNORED_DIR_NAMES = {"system volume information", "$recycle.bin", "recycler"}


def _safe_subdirs(path: Path) -> List[Path]:
    """
    List the subdirectories of `path`, skipping anything inaccessible.

    Every Windows drive carries OS-reserved folders that even the drive's
    regular users cannot list, and any lab member's own folder could in
    principle have a restrictive ACL. Skip either case with a printed warning
    instead of raising, so one locked-down folder doesn't abort the scan for
    every other drive/lab member.
    """
    try:
        entries = list(path.iterdir())
    except OSError as e:
        print(f"WARNING: cannot list {path}: {e}")
        return []
    result = []
    for p in entries:
        if p.name.lower() in _IGNORED_DIR_NAMES:
            continue
        try:
            if p.is_dir():
                result.append(p)
        except OSError as e:
            print(f"WARNING: cannot stat {p}: {e}")
    return result


def discover_sample_dirs(roots: List[Union[str, Path]]) -> List[dict]:
    """
    Find every {lab_member}/{sample_dir} folder under each root.

    Parameters
    ----------
    roots : list of str or Path
        Drive letters or folder paths, each expected to directly contain one
        subfolder per lab member, which in turn contains one subfolder per
        experiment sample: `{root}/{lab_member}/{sample_dir}`.

    Returns
    -------
    list of dict, each with keys `root`, `lab_member`, `sample_dir`, `path`.
    A root that doesn't exist (e.g. a disconnected drive) is skipped with a
    printed warning rather than raising, as is any subfolder that can't be
    listed (Windows-reserved folders, or another user's locked-down folder).
    """
    found = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            print(f"WARNING: root not found, skipping: {root}")
            continue
        for lab_member_dir in sorted(_safe_subdirs(root)):
            for sample_dir in sorted(_safe_subdirs(lab_member_dir)):
                found.append({
                    "root": str(root),
                    "lab_member": lab_member_dir.name,
                    "sample_dir": sample_dir.name,
                    "path": sample_dir,
                })
    return found


def audit_disk_usage(roots: List[Union[str, Path]], verbose: bool = True) -> pd.DataFrame:
    """
    Measure size and age for every {lab_member}/{sample_dir} folder under `roots`.

    Parameters
    ----------
    roots : list of str or Path
        See `discover_sample_dirs`.
    verbose : bool
        Print progress (one line per sample folder) as each is measured -- a
        full recursive size scan of a multi-terabyte experiment folder on a
        network drive can take minutes, so silent operation would look hung.

    Returns
    -------
    pd.DataFrame with one row per sample folder: `root`, `lab_member`,
    `sample_dir`, `path`, `size_bytes`, `size_gb`, `n_files`, `created`,
    `earliest_file_modified`, `latest_file_modified`, `days_since_created`.
    """
    samples = discover_sample_dirs(roots)
    rows = []
    for i, s in enumerate(samples, 1):
        if verbose:
            print(f"[{i}/{len(samples)}] measuring {s['root']}/{s['lab_member']}/{s['sample_dir']} ...")
        stats = measure_folder(s["path"])
        rows.append({**s, **stats})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["size_gb"] = df["size_bytes"] / 1e9
        df["days_since_created"] = (
            (pd.Timestamp.now() - pd.to_datetime(df["created"])).dt.days
        )
    return df


def group_by_lab_member(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """
    Group an audit DataFrame by lab member, case-insensitively.

    The same person's folder can appear under slightly different casing --
    across drives, or just because Windows Explorer lets people rename a
    folder without warning it already exists differently-cased (e.g. "Didar"
    on one drive, "didar" on another) -- which a plain `df.groupby("lab_member")`
    would wrongly treat as two different people. Folders under the same name
    on different roots (e.g. `D:/Data/Aaron` and `E:/Aaron`) already merge
    correctly with a plain groupby, since grouping is on the folder name
    alone; this only additionally folds together casing variants of that name.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a `lab_member` column (e.g. from `audit_disk_usage`).

    Returns
    -------
    list of (display_name, group_df) tuples, sorted by display_name
    case-insensitively. `display_name` is the single casing found, or every
    distinct casing seen joined with " / " (e.g. "Didar / didar") so a casing
    merge is visible rather than silently picking one spelling.
    """
    if df.empty:
        return []
    groups = []
    for _, group in df.groupby(df["lab_member"].str.lower()):
        names = sorted(group["lab_member"].unique())
        display_name = names[0] if len(names) == 1 else " / ".join(names)
        groups.append((display_name, group))
    groups.sort(key=lambda item: item[0].lower())
    return groups
