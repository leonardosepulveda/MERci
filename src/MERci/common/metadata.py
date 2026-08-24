# MERci/common/metadata.py
"""
Parse round_info.csv and positions.txt; build the unified look-up structures
used by both acquisition-planning and online-analysis modules.

Data model
----------
SeriesInfo        – one row of round_info.csv
FOVInfo           – per-FOV container: position + expected file paths
RoundInfo         – per-round container: series list + file paths by FOV
ExperimentMetadata – top-level object; build with ExperimentMetadata.load()
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


def _path_exists_safe(p: Path) -> bool:
    """
    Like ``Path.exists()``, but treats an OS-level access failure (e.g. an
    unmapped/differently-credentialed network drive letter recorded in
    round_info.csv's ``dir`` column on a different machine -- e.g.
    ``OSError: [WinError 1326] The user name or password is incorrect`` for
    a real absolute path that isn't reachable from this machine) the same as
    "doesn't exist" rather than letting it propagate. ``Path.exists()``
    itself only swallows the specific errno values that mean "not found";
    other OSErrors (permission/auth/unreachable) would otherwise crash the
    whole candidate-path resolution instead of just skipping to the next
    candidate.
    """
    try:
        return p.exists()
    except OSError:
        return False


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class SeriesInfo:
    """
    One row from ``round_info.csv``.

    Attributes
    ----------
    name          : series pattern, e.g. ``hal-mf3_{fov:03d}_00``
    round_id      : explicit imaging round number (``imaging_round`` column)
    imaging_type  : optional label such as ``"bits"`` or ``"cells"``
    hal_config    : optional HAL config filename for this series
    shutter_file  : optional shutter XML filename for this series
    data_dir      : optional per-round data directory; when set, image paths
                    are resolved relative to this directory instead of the
                    top-level ``data_dir``
    extra_meta    : all remaining CSV columns as a plain dict
    """

    name:          str
    round_id:      int
    imaging_type:  Optional[str]  = None
    hal_config:    Optional[str]  = None
    shutter_file:  Optional[str]  = None
    data_dir:      Optional[Path] = None
    extra_meta:    Dict           = field(default_factory=dict)
    candidate_dirs: List[Path]    = field(default_factory=list)

    _regex: "re.Pattern" = field(init=False, repr=False)
    # Cache of {directory: {fov_id: path}}, lazily built the first time
    # resolve_path() needs to fall back to a directory scan (see below) --
    # scanning is O(entries in dir), so this makes repeated per-FOV lookups
    # against the same directory (the common case: one round's 1000+ FOVs)
    # cost one scan total, not one per FOV.
    _dir_scan_cache: Dict[Path, Dict[int, Path]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._regex = _pattern_to_regex(self.name)

    def fov_from_stem(self, stem: str) -> Optional[int]:
        """Return the FOV id if *stem* matches this series, else ``None``."""
        m = self._regex.match(stem)
        if m is None:
            return None
        groups = m.groupdict()
        return int(groups["fov"]) if "fov" in groups else None

    def build_filename(self, fov_id: int, image_suffix: str = ".dax") -> str:
        """Construct the expected filename for *fov_id* from this series pattern."""
        return self.name.format(fov=fov_id) + image_suffix

    def candidate_paths(self, fov_id: int, image_suffix: str = ".dax") -> List[Path]:
        """
        Return every directory where *fov_id*'s image for this series might
        live, in search order.  Most series have a single candidate; the cells
        round is allowed to live in either ``data/`` or ``data/cells`` (see
        :func:`_build_metadata`), so it can have several.
        """
        fname = self.build_filename(fov_id, image_suffix)
        dirs  = self.candidate_dirs or ([self.data_dir] if self.data_dir else [Path(".")])
        return [Path(d) / fname for d in dirs]

    def _scan_dir_for_fov(self, dir_path: Path, fov_id: int, image_suffix: str) -> Optional[Path]:
        """
        Width-agnostic fallback: scan *dir_path* once, mapping every entry
        that matches this series' pattern (``fov_from_stem`` -- built from
        ``\\d+``, so it never assumed a specific zero-pad width to begin
        with) to its FOV id, then look up *fov_id* in that map.

        Exists because ``round_info.csv``'s ``series`` pattern encodes a
        SPECIFIC zero-pad width (e.g. ``{fov:03d}``) that can go stale --
        e.g. positions.txt regenerated with more FOVs (needing more digits)
        after round_info.csv was already written -- in which case
        ``build_filename``'s exact-width guess silently never matches any
        real file, even though the file is right there under a different
        width. Cached per directory so 1000+ per-FOV lookups against the
        same round only pay for one scan.
        """
        if dir_path not in self._dir_scan_cache:
            found: Dict[int, Path] = {}
            if _path_exists_safe(dir_path):
                suffix_lower = image_suffix.lower()
                try:
                    entries = list(dir_path.iterdir())
                except OSError:
                    entries = []
                for entry in entries:
                    if entry.suffix.lower() != suffix_lower:
                        continue
                    entry_fov = self.fov_from_stem(entry.stem)
                    if entry_fov is not None:
                        found[entry_fov] = entry
            self._dir_scan_cache[dir_path] = found
        return self._dir_scan_cache[dir_path].get(fov_id)

    def resolve_path(self, fov_id: int, image_suffix: str = ".dax") -> Path:
        """
        Return the first candidate path that exists on disk, falling back to
        a width-agnostic directory scan (see ``_scan_dir_for_fov``) if the
        exact-width guess doesn't match anything, and finally to the primary
        candidate when the FOV genuinely doesn't exist yet (e.g. before
        acquisition).
        """
        paths = self.candidate_paths(fov_id, image_suffix)
        for p in paths:
            if _path_exists_safe(p):
                return p
        dirs = self.candidate_dirs or ([self.data_dir] if self.data_dir else [Path(".")])
        for d in dirs:
            found = self._scan_dir_for_fov(Path(d), fov_id, image_suffix)
            if found is not None:
                return found
        return paths[0]


@dataclass
class FOVInfo:
    fov_id:    int
    position:  Tuple[float, float]
    files:     Dict[str, Path] = field(default_factory=dict)   # series_name → Path
    round_ids: List[int]        = field(default_factory=list)


@dataclass
class RoundInfo:
    round_id:  int
    series:    List[SeriesInfo]          = field(default_factory=list)
    fov_files: Dict[int, List[Path]]     = field(default_factory=dict)
    # fov_id → [path_series_A, path_series_B, …]


@dataclass
class ExperimentMetadata:
    """
    Top-level look-up object.  Build it once with
    ``ExperimentMetadata.load()``.
    """

    data_dir:    Path
    all_series:  List[SeriesInfo]
    fovs:        Dict[int, FOVInfo]    # fov_id  → FOVInfo
    rounds:      Dict[int, RoundInfo]  # round_id → RoundInfo
    n_fovs:      int
    n_rounds:    int
    image_suffix: str = ".dax"

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        round_info_csv: Path,
        positions_txt:  Path,
        data_dir:       Path,
        image_suffix:   str = ".dax",
    ) -> "ExperimentMetadata":
        """
        Parse both metadata files and assemble the ExperimentMetadata object.

        Parameters
        ----------
        round_info_csv : CSV with at minimum ``round_id`` and ``series`` columns
        positions_txt  : comma-separated ``x,y`` file (one FOV per line)
        data_dir       : root directory where image files will be found
        image_suffix   : file extension (default ``.dax``)
        """
        df        = _read_round_info(Path(round_info_csv))
        positions = _read_positions(Path(positions_txt))
        n_fovs    = len(positions)
        return _build_metadata(df, positions, Path(data_dir), n_fovs, image_suffix)

    # ── Convenience accessors ──────────────────────────────────────────────────

    def all_expected_files(self) -> List[Path]:
        """
        Every image file expected across the whole experiment.

        Paths are resolved against each series' candidate directories at call
        time, so a cells round that landed in ``data/`` rather than
        ``data/cells`` (or vice-versa) is still reported at its real location.
        """
        return [
            s.resolve_path(fov_id, self.image_suffix)
            for s in self.all_series
            for fov_id in sorted(self.fovs)
        ]

    def files_for_round(self, round_id: int) -> List[Path]:
        """
        All expected image files for the given round (all FOVs, all series),
        resolved to their real on-disk location when present.
        """
        r = self.rounds.get(round_id)
        if r is None:
            return []
        return [
            s.resolve_path(fov_id, self.image_suffix)
            for s in r.series
            for fov_id in sorted(self.fovs)
        ]

    def files_for_fov(self, fov_id: int) -> List[Path]:
        return list(self.fovs[fov_id].files.values())

    def round_id_of_file(self, path: Path) -> Optional[int]:
        """Return the round id of *path*, or ``None`` if unrecognised."""
        stem = Path(path).stem
        for s in self.all_series:
            if s.fov_from_stem(stem) is not None:
                return s.round_id
        return None

    def fov_id_of_file(self, path: Path) -> Optional[int]:
        """Return the FOV id of *path*, or ``None`` if unrecognised."""
        stem = Path(path).stem
        for s in self.all_series:
            fov = s.fov_from_stem(stem)
            if fov is not None:
                return fov
        return None

    def series_of_file(self, path: Path) -> Optional[SeriesInfo]:
        """Return the SeriesInfo matching *path*, or ``None``."""
        stem = Path(path).stem
        for s in self.all_series:
            if s.fov_from_stem(stem) is not None:
                return s
        return None

    def series_for_round(
        self,
        round_id:     int,
        imaging_type: Optional[str] = None,
    ) -> List[SeriesInfo]:
        """
        Return all SeriesInfo objects for *round_id*, optionally filtered
        by *imaging_type* (e.g. ``"bits"`` or ``"cells"``).
        """
        r = self.rounds.get(round_id)
        if r is None:
            return []
        if imaging_type is None:
            return r.series
        return [s for s in r.series if s.imaging_type == imaging_type]

    def valid_round_ids(self) -> List[int]:
        """All round ids, sorted."""
        return sorted(self.rounds)

    def round_fully_written(self, round_id: int) -> bool:
        """
        True iff every expected raw image file for *round_id* already exists
        on disk -- purely "has HAL finished writing this round", with no
        dependency on any analysis sentinel (unlike
        :func:`MERci.progress.ProgressTracker.all_fovs_done_for_round`, which
        additionally requires a ``.fov_done`` sentinel per file).
        """
        files = self.files_for_round(round_id)
        return bool(files) and all(_path_exists_safe(f) for f in files)


# ── Internal helpers ────────────────────────────────────────────────────────

def _is_cells_series(s: SeriesInfo) -> bool:
    """A series is the cells round if its imaging_type is ``cells`` (or, absent
    that, if ``cells`` appears in its series name)."""
    if s.imaging_type is not None:
        return s.imaging_type.strip().lower() == "cells"
    return "cells" in s.name.lower()


def _resolve_series_dir(dir_str: str, data_dir: Path) -> Path:
    """
    Resolve one ``round_info.csv`` ``dir``/``data_dir`` cell to a real path
    on THIS machine.

    ``dir`` is written by whichever machine generated ``round_info.csv``
    (normally the microscope's own Windows PC, via ``before_imaging``'s
    generators) and is an absolute path there -- but a Windows-style
    absolute path (``V:\\Leonardo\\...\\data\\focus_test`` or
    ``D:/Leonardo/.../data/hybs/H01``, either slash direction) is NEVER
    absolute once read back with ``pathlib.Path`` on POSIX (Linux/mac):
    ``Path.is_absolute()`` requires a leading ``/``, which a drive letter
    never has. This silently fell into the "relative -- resolve under this
    machine's own SAMPLE_DIR" branch, but a backslash-separated string
    parses as ONE opaque path component under POSIX `Path` (backslash isn't
    a separator there), so joining it onto SAMPLE_DIR just produces one
    bogus, nonexistent nested directory -- every round can resolve 0 imaged
    FOVs even though every real file is present. A forward-slash Windows
    path decomposes into real path parts under POSIX `Path` but is still
    wrong (the drive + every
    directory above ``data/`` gets appended onto SAMPLE_DIR verbatim,
    still never a real path) -- so this isn't just a backslash bug.

    Fixed by parsing *dir_str* with :class:`PureWindowsPath` (recognizes
    both slash directions and drive letters regardless of the OS actually
    running this code), and -- whenever that reveals a genuine Windows
    drive letter, meaning the original ``is_absolute()``/join logic could
    never have been right on this machine -- keeping only the path's tail
    from its last literal ``"data"`` segment onward and re-rooting that
    under THIS machine's own *data_dir* (``SAMPLE_DIR/data``). Every real
    ``dir`` value in this repo's own convention points somewhere under
    ``.../data/...`` (``data/cells``, ``data/hybs/H01``, ``data/
    focus_test``, ``data/tissue_1/hybs/H01``, ...), so that tail is exactly
    the sub-path this machine's own ``data_dir`` needs.

    A genuinely POSIX-absolute *dir_str* (this machine's own convention, or
    an experiment that has always lived on Linux) is returned unchanged, as
    before. A real relative path (no drive letter) still resolves against
    ``data_dir.parent`` (SAMPLE_DIR), also as before.
    """
    p = Path(dir_str)
    if p.is_absolute():
        return p

    wp = PureWindowsPath(dir_str)
    if wp.drive:
        parts = wp.parts
        if "data" in parts:
            tail = parts[parts.index("data") + 1:]
            return data_dir.joinpath(*tail)
        log.warning(
            "round_info.csv dir %r looks like a Windows absolute path but "
            "has no 'data' segment to re-root under this machine's own "
            "data_dir (%s) -- falling back to the (likely still wrong) "
            "raw join.", dir_str, data_dir,
        )

    return data_dir.parent / p


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    """Return *paths* with duplicates removed, preserving first-seen order."""
    seen: set = set()
    out:  List[Path] = []
    for p in paths:
        key = Path(p)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _pattern_to_regex(pattern: str) -> "re.Pattern":
    """
    Convert a Python-format-string series pattern into a compiled regex.

    ``{fov:03d}``  → named group ``(?P<fov>\\d+)``

    Examples
    --------
    ``hal-mf3_{fov:03d}_00``  →  ``^hal\\-mf3_(?P<fov>\\d+)_00$``
    ``hal-mf3-cells_{fov:03d}`` → ``^hal\\-mf3\\-cells_(?P<fov>\\d+)$``
    """
    parts: List[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "{":
            j    = pattern.index("}", i)
            name = pattern[i + 1 : j].split(":")[0]
            parts.append(f"(?P<{name}>\\d+)")
            i = j + 1
        else:
            j = pattern.find("{", i)
            if j == -1:
                j = len(pattern)
            parts.append(re.escape(pattern[i:j]))
            i = j
    return re.compile("^" + "".join(parts) + "$")


def _read_round_info(csv_path: Path) -> pd.DataFrame:
    """
    Load ``round_info.csv``.

    Required columns: ``imaging_round`` (or legacy ``round_id``), ``series``
    Optional columns: ``imaging_type``, ``hal_config``, ``shutter_file``, ``dir``, others
    """
    df = pd.read_csv(csv_path)
    # Accept 'imaging_round' (new) or 'round_id' (legacy)
    if "imaging_round" in df.columns and "round_id" not in df.columns:
        df = df.rename(columns={"imaging_round": "round_id"})
    for col in ("round_id", "series"):
        if col not in df.columns:
            raise ValueError(
                f"round_info.csv must contain a '{col}' column "
                f"(found columns: {list(df.columns)})"
            )
    df["series"]   = df["series"].astype(str).str.strip()
    df["round_id"] = df["round_id"].astype(int)
    return df


def _read_positions(pos_path: Path) -> Dict[int, Tuple[float, float]]:
    """
    Parse per-FOV stage positions from a comma-separated text file.

    One line per FOV: ``x,y``.  Lines beginning with ``#`` or blank
    lines are ignored.

    Returns
    -------
    {fov_id: (x, y)} — zero-indexed.
    """
    positions: Dict[int, Tuple[float, float]] = {}
    fov_id = 0

    with pos_path.open() as fh:
        for raw in fh:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                log.warning("Short line at FOV %d: %r – skipping.", fov_id, raw)
                continue
            try:
                positions[fov_id] = (float(parts[0]), float(parts[1]))
                fov_id += 1
            except ValueError as exc:
                log.warning("Bad position at FOV %d: %s", fov_id, exc)

    return positions


def _parse_series_row(row: pd.Series) -> SeriesInfo:
    reserved = {"round_id", "imaging_round", "series", "imaging_type",
                "hal_config", "shutter_file", "data_dir", "dir"}
    extra    = {k: v for k, v in row.items() if k not in reserved}

    def _opt(key: str) -> Optional[str]:
        val = row.get(key)
        return str(val).strip() if pd.notna(val) else None

    # Accept data_dir (new) or dir (legacy)
    dir_str = _opt("data_dir") or _opt("dir")
    return SeriesInfo(
        name         = str(row["series"]),
        round_id     = int(row["round_id"]),
        imaging_type = _opt("imaging_type"),
        hal_config   = _opt("hal_config"),
        shutter_file = _opt("shutter_file"),
        data_dir     = Path(dir_str) if dir_str else None,
        extra_meta   = extra,
    )


def _build_metadata(
    df:           pd.DataFrame,
    positions:    Dict[int, Tuple[float, float]],
    data_dir:     Path,
    n_fovs:       int,
    image_suffix: str,
) -> ExperimentMetadata:

    all_series = [_parse_series_row(row) for _, row in df.iterrows()]

    # ── FOV objects ────────────────────────────────────────────────────────
    fovs: Dict[int, FOVInfo] = {
        fov_id: FOVInfo(fov_id=fov_id, position=pos)
        for fov_id, pos in positions.items()
    }

    # ── Round objects ──────────────────────────────────────────────────────
    round_ids = sorted({s.round_id for s in all_series})
    rounds: Dict[int, RoundInfo] = {rid: RoundInfo(round_id=rid) for rid in round_ids}

    # ── Assign files ───────────────────────────────────────────────────────
    fov_ids = sorted(fovs.keys())
    for s in all_series:
        rid = s.round_id
        r   = rounds[rid]
        r.series.append(s)

        # Candidate directories, in search order.  The cells round is allowed
        # to live in either the top-level ``data/`` or a ``data/cells`` subfolder,
        # so it gets both as fallbacks regardless of what round_info.csv recorded.
        #
        # See _resolve_series_dir's own docstring for why this isn't just
        # "absolute -> use as-is, else resolve under SAMPLE_DIR": a Windows
        # ``dir`` value (this repo's normal case -- written by the
        # microscope's own Windows PC) needs its own handling to resolve
        # correctly once read back on POSIX.
        if s.data_dir is not None:
            primary = _resolve_series_dir(str(s.data_dir), data_dir)
        else:
            primary = data_dir
        if _is_cells_series(s):
            s.candidate_dirs = _dedupe_paths([primary, data_dir, data_dir / "cells"])
        else:
            s.candidate_dirs = [primary]

        for fov_id in fov_ids:
            try:
                fname = s.build_filename(fov_id, image_suffix)
            except (KeyError, TypeError) as exc:
                log.warning(
                    "Cannot build filename for series '%s' fov %d: %s",
                    s.name, fov_id, exc,
                )
                continue

            # Store the resolved (existing-or-primary) path; stems are identical
            # across candidates, so thumbnail/sentinel lookups are unaffected.
            fpath = s.resolve_path(fov_id, image_suffix)
            fovs[fov_id].files[s.name] = fpath
            if rid not in fovs[fov_id].round_ids:
                fovs[fov_id].round_ids.append(rid)
            r.fov_files.setdefault(fov_id, []).append(fpath)

    return ExperimentMetadata(
        data_dir     = data_dir,
        all_series   = all_series,
        fovs         = fovs,
        rounds       = rounds,
        n_fovs       = n_fovs,
        n_rounds     = len(round_ids),
        image_suffix = image_suffix,
    )