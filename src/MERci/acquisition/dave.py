# MERci/acquisition/dave.py
"""
Generate Dave experiment configuration files.

Dave is the experiment orchestration program that calls HAL (imaging) and
Kilroy (fluidics) to execute a full MERFISH acquisition.  A Dave config
(``<recipe>``) lists imaging loops and fluidics steps in the order they run.

This module produces **explicit-block** configs: every imaging round and every
fluidics step is written as a separate, named element — no loops over variables.
This makes the file easy to inspect and edit before starting an experiment.

Experiment structure
--------------------
Round 1 (imaging): "Cells Imaging" -- cells acquisition for all FOVs (no preceding fluidics)
"Hyb 01 Fluidics": Hybridize/Adaptor 1 → [Readouts] → Buffer   ← NO cleave (first hyb)
Round 2 (imaging): "Hyb 01 Imaging" -- bits #1 acquisition for all FOVs
"Hyb 02 Fluidics": Cleave → Hybridize/Adaptor 2 → [Readouts] → Buffer
…
Round N+1 (imaging): "Hyb N Imaging" -- bits #N acquisition
[Optional] "Fluidics Final": Cleave only

Loops are named "Cells Imaging" (round 1), "Hyb NN Imaging"/"Hyb NN Fluidics"
(bits rounds, NN = bit/hyb index), or "Fluidics Final" (the optional closing
cleave) -- never the raw imaging_round number, so a leading cells round never
shifts what the label means. A fluidics loop is named by the hyb index of the
imaging round it PRECEDES (e.g. "Hyb 01 Fluidics" precedes "Hyb 01 Imaging").
The hyb-protocol number tracks this same bit/hyb index (1…N), and the first
hyb omits the cleave step (see ``create_dave_config(first_hyb_no_cleave=...)``).

The concrete Kilroy protocol names written into the recipe are resolved from the
Kilroy config passed as ``create_dave_config(kilroy_config=...)`` (see
``acquisition/kilroy.py``), so every protocol referenced is guaranteed to exist
in the Kilroy file that runs the experiment.

The ``round_info.csv`` drives everything:
- rows with the same ``imaging_round`` are acquired in the same imaging loop
- the order within a round follows the CSV row order
- ``hal_config`` names the HAL config file (with or without ``.xml``)
- ``series`` encodes the base movie name: strip ``_{fov:…}`` suffix to get the
  dave ``<name>`` element (e.g. ``hal-mf3_01_{fov:03d}`` → ``hal-mf3_01``)
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path, PurePath, PureWindowsPath
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from xml.dom import minidom

import pandas as pd

from ..common.io import load_positions
from .configs import get_camera_frame_size, read_hal_exposure_time, read_hal_frame_count
from .kilroy import (
    KilroyProtocolResolver,
    load_kilroy_protocols,
    load_protocol_durations,
    protocol_last_flowed_valve,
)
from .positions import group_boundaries_by_path_mode

log = logging.getLogger(__name__)


# ── Public helpers ─────────────────────────────────────────────────────────────

def series_to_movie_name(series: str) -> str:
    """
    Strip the ``_{fov:…}`` format-string suffix from a series pattern to get
    the Dave movie base name.

    Examples
    --------
    ``hal-mf3_01_{fov:03d}``    → ``hal-mf3_01``
    ``hal-mf3-cells_{fov:03d}`` → ``hal-mf3-cells``
    """
    return re.sub(r"_\{[^}]+\}$", "", series)


def dave_config_filename(microscope: str, n_hybs: int, sample_name: str) -> str:
    """
    Dave recipe filename: ``dave-{mic}-{n_hybs}hybs-{sample_name}.xml``.

    A single source of truth for this name, shared by the notebook that writes
    the recipe and the one that later re-opens it to annotate it with bit
    info -- constructing the exact expected filename rather than globbing
    ``settings/dave-*.xml`` and guessing which match is "the" recipe. That
    guess breaks as soon as more than one dave-*.xml exists in the same
    settings/ folder (e.g. two acquisitions sharing one sample folder before
    it's split into per-acquisition subfolders): sorting alphabetically picks
    "dave-{mic}-13hybs-…" before "dave-{mic}-9hybs-…" (string comparison, not
    numeric), so the wrong file's annotated silently.
    """
    return f"dave-{microscope.lower()}-{n_hybs}hybs-{sample_name}.xml"


def dave_cells_config_filename(microscope: str, sample_name: str) -> str:
    """Dave recipe filename for the cells-only recipe: ``dave-{mic}-cells-{sample_name}.xml``.

    Same "single source of truth" rationale as :func:`dave_config_filename`.
    """
    return f"dave-{microscope.lower()}-cells-{sample_name}.xml"


def dave_focustest_config_filename(microscope: str, sample_name: str) -> str:
    """Dave recipe filename for the focus-lock test recipe: ``dave-{mic}-focustest-{sample_name}.xml``.

    Same "single source of truth" rationale as :func:`dave_config_filename`.
    """
    return f"dave-{microscope.lower()}-focustest-{sample_name}.xml"


def _infer_microscope(round_info: pd.DataFrame) -> Optional[str]:
    """
    Best-effort microscope id from the ``series`` names in *round_info*.

    MERci series follow ``hal-{mic}…`` (e.g. ``hal-mf3-cells_{fov:03d}``),
    so the token after ``hal-`` is the microscope. Returns it upper-cased (e.g.
    ``"MF3"``), or ``None`` if no series matches the pattern.
    """
    if "series" not in round_info.columns:
        return None
    for s in round_info["series"]:
        m = re.match(r"hal-([A-Za-z0-9]+)-", str(s))
        if m:
            return m.group(1).upper()
    return None


def get_hal_frame_count(hal_config_path: Path) -> int:
    """Return the ``<frames>`` value from a HAL config XML file."""
    n = read_hal_frame_count(hal_config_path)
    if n is None:
        raise ValueError(f"No <frames> element found in {hal_config_path}")
    return n


def resolve_hal_config_path(settings_dir: Path, hal_stem: str) -> Path:
    """
    Resolve a hal_config's path, checking *settings_dir* first (the usual
    single-hal_config-per-round convention) and falling back to the sibling
    ``multi_z/`` folder (where a variable-z-per-FOV round's tier hal_configs
    -- and their co-located shutter files, kept together so HAL's own
    ``<shutters>`` same-directory resolution keeps working -- are written
    instead of flat ``settings/``, see notebook 05's own docstring).

    Returns the first candidate that exists; the *settings_dir* candidate if
    neither does (so a subsequent ``open()`` raises a normal, clear
    ``FileNotFoundError`` rather than this function inventing one).
    """
    settings_dir = Path(settings_dir)
    direct = settings_dir / (hal_stem + ".xml")
    if direct.exists():
        return direct
    multi_z = settings_dir.parent / "multi_z" / (hal_stem + ".xml")
    if multi_z.exists():
        return multi_z
    return direct


def count_positions(positions_path: Path) -> int:
    """
    Count the FOV positions in a ``positions_*.txt`` file.

    One FOV per non-blank line (``x,y``); ``#`` comments and blank lines are
    ignored, matching :func:`MERci.common.io.load_positions`.  This equals
    the number of iterations Dave runs for a ``<loop>`` bound to this file, which
    is what the per-segment ``start`` offsets (see
    :func:`create_round_info_multitissue`) are built from.

    Parameters
    ----------
    positions_path : path to the comma-separated positions file

    Returns
    -------
    int : number of valid ``x,y`` FOV lines
    """
    return len(load_positions(positions_path))


def fov_pad_width(total_fovs: int) -> int:
    """
    Zero-pad width for FOV indices ``0 … total_fovs-1`` (150 FOVs -> 3 digits,
    1036 -> 4), matching how HAL names files. Always derive it from the real
    FOV count: a fixed width stops matching once the count crosses a digit
    boundary. No minimum width.
    """
    return len(str(max(total_fovs - 1, 0)))


# ── Multi-drive group assignment ────────────────────────────────────────────────

def normalize_drive_root(drive: Union[str, Path]) -> PurePath:
    """
    Normalize a drive-letter/root string into an absolute path anchor.

    A bare drive letter like ``"Z:"`` is a *drive-relative* path under
    ``pathlib``/Windows: joining onto it (``Path("Z:") / "data"``) depends on
    that drive's process-wide "current directory", an unpredictable legacy
    mechanism, and would NOT resolve to ``Z:\\data``. Appending a trailing
    separator turns it into the drive's root (``"Z:\\"``) so every join below
    is unambiguously absolute.
    """
    s = str(drive)
    if not s.endswith(("\\", "/")):
        s += "\\"
    # A drive letter is a Windows path on any OS (round_info.csv is read by
    # the Windows acquisition PC, even when generated on Linux).
    root = PureWindowsPath(s) if PureWindowsPath(s).drive else Path(s)
    if not root.is_absolute():
        raise ValueError(
            f"drive {drive!r} does not resolve to an absolute path "
            f"(got {root!r}) — use a drive letter (e.g. \"Z:\") or an absolute path."
        )
    return root


def rebase_on_drive(sample_dir: Union[str, Path], drive_root: PurePath) -> PurePath:
    """
    Re-root ``sample_dir`` onto a different drive, preserving its subpath.

    A round can land on a physical drive other than the one ``SAMPLE_DIR``
    itself lives on, but its folder layout on that drive should mirror
    ``SAMPLE_DIR``'s own structure rather than dumping straight into the
    drive's root — e.g. ``SAMPLE_DIR = Z:\\Leonardo\\LT058_sample_07\\lineage``
    rebased onto ``"Y:"`` gives ``Y:\\Leonardo\\LT058_sample_07\\lineage``, not
    just ``Y:\\``. ``drive_root`` is expected already normalized (see
    :func:`normalize_drive_root`).
    """
    sample_dir = (PureWindowsPath(sample_dir) if PureWindowsPath(str(sample_dir)).drive
                  else Path(sample_dir))
    return drive_root / sample_dir.relative_to(sample_dir.anchor)


def _expand_hyb_drive_groups(
    n_bits: int,
    groups: Sequence[Tuple[int, Union[str, Path]]],
) -> Dict[int, Union[str, Path]]:
    """
    Expand ``(count, drive)`` consecutive-block groups into a ``{bit_idx: drive}``
    mapping over bit/hyb indices ``1…n_bits``, in order.

    E.g. ``n_bits=25``, ``groups=[(6, "Y:"), (6, "Z:"), (6, "Y:"), (7, "Z:")]``
    assigns bits 1-6 to ``"Y:"``, 7-12 to ``"Z:"``, 13-18 to ``"Y:"``, 19-25 to
    ``"Z:"``. The counts must sum to exactly ``n_bits`` -- a mismatch almost
    always means a typo in the group sizes, and silently imaging some rounds
    to the wrong (or no) drive is far more expensive to notice than a
    same-day ``ValueError``.
    """
    assignment: Dict[int, Union[str, Path]] = {}
    bit_idx = 1
    for count, drive in groups:
        for _ in range(count):
            assignment[bit_idx] = drive
            bit_idx += 1
    covered = bit_idx - 1
    if covered != n_bits:
        raise ValueError(
            f"hyb_drive_groups covers {covered} rounds but n_bits={n_bits} "
            f"-- group sizes must sum to n_bits exactly."
        )
    return assignment


# ── round_info builder ─────────────────────────────────────────────────────────

def create_round_info(
    microscope:       str,
    n_bits:           int,
    bits_hal_config:  str,
    cells_hal_config: str,
    sample_dir:       Path,
    positions_txt:    Optional[Path] = None,
    hyb_drive_groups: Optional[Sequence[Tuple[int, Union[str, Path]]]] = None,
    cells_drive:      Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Build the ``round_info.csv`` dataframe for a standard MERFISH experiment.

    Imaging round 1 is the **cells** acquisition only.  Imaging rounds 2…N+1 are
    the bits acquisitions (bit/hyb #1…#N).  The bits movie *series* number tracks
    the bit/hyb index (``_01``…``_0N``), not the imaging-round number, so the
    Kilroy hyb-protocol numbers stay stable regardless of the cells offset.

    Parameters
    ----------
    microscope        : microscope identifier in lowercase, e.g. ``"mf3"``
    n_bits            : number of bits (hybridisation) rounds
    bits_hal_config   : HAL config filename for bits rounds (with ``.xml``)
    cells_hal_config  : HAL config filename for the cells round (with ``.xml``)
    sample_dir        : experiment root directory; used to build ``data_dir`` paths
    positions_txt     : the experiment's positions file, used to count real
                        FOVs and derive the ``series`` pattern's zero-pad
                        width (see :func:`fov_pad_width`) -- e.g. 150 FOVs ->
                        ``{fov:03d}``, 1036 FOVs -> ``{fov:04d}``. ``None``
                        (default) falls back to a fixed 3-digit width (the
                        previous, hardcoded behaviour) with a warning, since
                        the true FOV count isn't known without it -- pass
                        this whenever the positions file is available, which
                        it should be by the time ``round_info.csv`` is built.
    hyb_drive_groups  : when given, consecutive blocks of hyb rounds are
                        assigned to alternating drives instead of all staying
                        on ``sample_dir``'s own drive -- e.g.
                        ``[(6, "Y:"), (6, "Z:"), (6, "Y:"), (7, "Z:")]`` for a
                        25-hyb experiment split into four groups. Each
                        group's ``data_dir`` is rooted at that drive, rebased
                        onto ``sample_dir``'s own subpath (see
                        :func:`rebase_on_drive`) -- so e.g.
                        ``Z:\\Leonardo\\LT066_sample_02\\lineage`` groups onto
                        ``"Y:"`` become
                        ``Y:\\Leonardo\\LT066_sample_02\\lineage``, not just
                        ``Y:\\``. The group counts must sum to exactly
                        ``n_bits``. ``create_dave_config`` emits a matching
                        ``<change_directory>`` before each round's imaging
                        loop. ``None`` (default) preserves today's
                        single-drive layout.
    cells_drive       : when given, the cells round's ``data_dir`` is rooted
                        at this drive the same way (e.g. so cells lands on
                        the same drive as the first/last hyb group instead of
                        always ``sample_dir``'s own drive). ``None`` (default)
                        keeps the cells round on ``sample_dir``'s own drive.

    Returns
    -------
    pd.DataFrame with columns ``imaging_round``, ``imaging_type``, ``series``,
    ``hal_config``, ``data_dir``
    """
    mic  = microscope.lower()
    rows: List[dict] = []

    if positions_txt is not None:
        pad = fov_pad_width(count_positions(positions_txt))
    else:
        log.warning(
            "create_round_info: no positions_txt given -- falling back to a "
            "fixed 3-digit FOV zero-pad width. Pass positions_txt so the "
            "width is derived from the real FOV count instead (an "
            "experiment with >=1000 FOVs needs 4+ digits, and a fixed width "
            "silently stops matching real files once FOV count crosses a "
            "digit boundary)."
        )
        pad = 3

    drive_for_bit = _expand_hyb_drive_groups(n_bits, hyb_drive_groups) if hyb_drive_groups else {}

    # Imaging Round 1: CELLS ONLY (no fluidics precedes it).
    cells_root = (
        rebase_on_drive(sample_dir, normalize_drive_root(cells_drive))
        if cells_drive else Path(sample_dir)
    )
    rows.append({
        "imaging_round": 1,
        "imaging_type":  "cells",
        "series":        f"hal-{mic}-cells_{{fov:0{pad}d}}",
        "hal_config":    cells_hal_config,
        "data_dir":      str(cells_root / "data" / "cells"),
    })

    # Imaging Rounds 2 … N+1: bits #1 … #N.  The series number tracks the
    # bit/hyb index (1…N); the imaging_round is bit_idx + 1.  Each bits round
    # writes into its own subfolder ``data/hybs/H{NN}`` (NN = bit/hyb index), so
    # the rounds are spread across folders instead of piling into one ``data/``.
    # When ``hyb_drive_groups`` is given, that subfolder is additionally rooted
    # at the group-assigned drive instead of always ``sample_dir`` (see
    # ``rebase_on_drive``).
    for bit_idx in range(1, n_bits + 1):
        drive     = drive_for_bit.get(bit_idx)
        bits_root = (
            rebase_on_drive(sample_dir, normalize_drive_root(drive))
            if drive else Path(sample_dir)
        )
        rows.append({
            "imaging_round": bit_idx + 1,
            "imaging_type":  "bits",
            "series":        f"hal-{mic}_{bit_idx:02d}_{{fov:0{pad}d}}",
            "hal_config":    bits_hal_config,
            "data_dir":      str(bits_root / "data" / "hybs" / f"H{bit_idx:02d}"),
        })

    return pd.DataFrame(
        rows,
        columns=["imaging_round", "imaging_type", "series", "hal_config", "data_dir"],
    )


def create_round_info_multitissue(
    microscope:         str,
    n_bits:             int,
    bits_hal_config:    str,
    cells_hal_config:   str,
    transit_hal_config: str,
    sample_dir:         Path,
    boundaries:         Sequence,
    mode:               str,
    sample_name:        str,
    tissue_path_mode:   Callable[[int], str] = lambda tissue: "legacy",
) -> pd.DataFrame:
    """
    Build a **segment-aware** ``round_info`` for a multi-boundary experiment.

    Each imaging round visits the acquisition-order segments built by
    :func:`MERci.acquisition.positions.group_boundaries_by_path_mode` (the
    same function ``notebooks/before_imaging/02`` uses to decide what it
    actually writes to ``positions/``): a tissue's own consecutive boundaries
    are merged into ONE segment when ``tissue_path_mode(tissue)`` is
    ``"legacy"`` or ``"union"`` (no transit within that tissue), otherwise
    each boundary keeps its own segment; a transit segment always bridges
    consecutive top-level segments
    (wrapping the last back to the first) whenever there is more than one.
    This produces **one row per (round, segment)** — so each round has
    several movies: a boundary movie (cells/bits HAL config) per segment and
    a transit movie (transit HAL config, blank frames) per transit bridge.

    Round 1 is the cells acquisition; rounds 2…N+1 are bits #1…#N. Every row also
    carries the ``positions_file`` (basename in ``positions/``) and the per-tissue
    ``data_dir`` subfolder the segment writes to.

    **Consolidated movie names + continuous FOV index.** Within a round, all
    boundary movies share ONE movie name (e.g. ``hal-mf3-cells`` /
    ``hal-mf3_01``) and all transit movies share one name
    (``hal-mf3-transit_rNN``) — the per-segment label is dropped from the movie
    name. To keep the per-loop indices from colliding, each row carries an
    ``fov_start`` offset (running FOV count of the preceding segments of the same
    group, in traversal order) and a fixed ``fov_pad`` width; these become the
    ``start``/``pad`` attributes on the Dave ``<name>`` (see
    :func:`create_dave_config`). Boundary FOVs therefore number continuously
    ``0…(ΣboundaryFOVs−1)`` and transit FOVs ``0…(ΣtransitFOVs−1)`` across segments,
    while the loops stay separate so the boundary→transit interleaving is preserved.
    The positions files (written by notebook 02) must already exist — they are read
    to count FOVs. **The recipe this produces requires the patched Dave**
    (``dave_fov_offset_patch``); stock Dave ignores ``start``/``pad`` and the shared
    names would overwrite each other.

    Parameters
    ----------
    microscope         : microscope id, e.g. ``"MF3"``
    n_bits             : number of bits (hyb) rounds
    bits_hal_config    : HAL config filename for boundary movies in bits rounds
    cells_hal_config   : HAL config filename for boundary movies in the cells round
    transit_hal_config : HAL config filename for transit movies (blank frames)
    sample_dir         : experiment root; used to build ``data_dir`` paths
    boundaries         : ordered ``BoundarySpec`` list from
                         :func:`MERci.acquisition.positions.discover_boundary_files`
    mode               : ``"multi"``, ``"single"`` or ``"legacy"`` (from the same
                         discovery call); selects the data-folder layout
    sample_name        : experiment name used in the positions filenames
    tissue_path_mode   : tissue index -> ``"legacy"``, ``"transit"`` or
                         ``"union"`` (see :func:`MERci.acquisition.positions.
                         group_boundaries_by_path_mode`) -- "legacy" and
                         "union" group identically here (this function only
                         needs the resulting segment/label structure, not
                         how notebook 02 actually built each segment's FOV
                         coordinates).
                         MUST match whatever notebook 02 actually used to write
                         ``positions/`` (same convention as this notebook already
                         has to agree with notebook 02 on ``BOUNDARY_SOURCE``),
                         or this will reference positions files that don't
                         exist. Defaults to ``"legacy"`` for every tissue, notebook
                         02's own default.

    Returns
    -------
    pd.DataFrame with columns ``imaging_round``, ``imaging_type`` (``cells`` /
    ``bits`` / ``transit``), ``series``, ``hal_config``, ``data_dir``,
    ``positions_file``, ``tissue``, ``segment``, ``fov_start``, ``fov_pad``.
    """
    mic     = microscope.lower()
    data    = Path(sample_dir) / "data"
    pos_dir = Path(sample_dir) / "positions"

    # Same grouping notebook 02 used to decide what it wrote to positions/ --
    # a tissue's own consecutive boundaries collapse into one segment under
    # "legacy" mode, otherwise each stays its own segment. A transit bridges
    # every consecutive pair of the resulting top-level segments (wrapping
    # the last back to the first) whenever there is more than one.
    groups     = group_boundaries_by_path_mode(boundaries, mode, tissue_path_mode)
    n_groups   = len(groups)
    n_transits = n_groups if n_groups > 1 else 0

    def _seg_dir(tissue: int, kind: str, is_cells: bool, hyb_idx: Optional[int] = None) -> str:
        base = data / f"tissue_{tissue}" if mode == "multi" else data
        if kind == "transit":
            return str(base / "transit")
        if is_cells:
            return str(base / "cells")
        # bits: separate each hyb round into its own subfolder ``hybs/H{NN}``
        # (NN = bit/hyb index) so rounds are spread across folders.
        return str(base / "hybs" / f"H{hyb_idx:02d}")

    def _posfile(label: str) -> str:
        # Matches notebook 02's own convention: a merged/legacy segment with
        # an empty label is the plain aggregate positions_{sample}.txt.
        return f"positions_{sample_name}_{label}.txt" if label else f"positions_{sample_name}.txt"

    # Ordered segment templates for one round's traversal (round-independent):
    # (kind, tissue, label, positions_file).
    seg_templates: List[tuple] = []
    for k, g in enumerate(groups):
        seg_templates.append(("boundary", g.tissue, g.label, _posfile(g.label)))
        if n_transits:
            seg_templates.append(
                ("transit", g.tissue, f"transit_{k + 1}",
                 f"positions_{sample_name}_transit_{k + 1}.txt")
            )

    # ── Continuous FOV numbering across segments ────────────────────────────────
    # We want every boundary movie in a round to share ONE movie name (e.g.
    # ``hal-mf3-cells``) with a single running FOV index 0…(ΣtBoundaryFOVs−1),
    # and likewise every transit movie to share one name — while KEEPING the
    # per-segment loops so the boundary→transit interleaving is preserved.
    #
    # Dave numbers each loop 0…n−1 independently, so shared names would collide.
    # The patched ``v2Generator`` accepts a per-movie ``start`` offset and fixed
    # ``pad`` (see ``dave_fov_offset_patch``); we compute them here from the FOV
    # counts of the positions files that Dave will iterate. ``start`` for a segment
    # is the number of FOVs in the preceding segments of the SAME group (boundary
    # vs transit), in traversal order; ``pad`` is a fixed zero-pad width wide enough
    # for the whole group (≥3 to keep the conventional 3-digit index).
    #
    # NOTE: because the movie names are now shared, the generated recipe REQUIRES
    # the patched Dave. Under stock Dave the ``start``/``pad`` attributes are
    # ignored and the shared names would overwrite each other.
    counts = [count_positions(pos_dir / posfile) for (_, _, _, posfile) in seg_templates]

    boundary_total = sum(c for (t, c) in zip(seg_templates, counts) if t[0] == "boundary")
    transit_total  = sum(c for (t, c) in zip(seg_templates, counts) if t[0] == "transit")
    boundary_pad   = fov_pad_width(boundary_total)
    transit_pad    = fov_pad_width(transit_total)

    # Enrich each template with its running start offset and group pad.
    enriched: List[dict] = []
    b_off = t_off = 0
    for (kind, tissue, label, posfile), cnt in zip(seg_templates, counts):
        if kind == "boundary":
            start, pad = b_off, boundary_pad
            b_off += cnt
        else:
            start, pad = t_off, transit_pad
            t_off += cnt
        enriched.append({"kind": kind, "tissue": tissue, "label": label,
                         "posfile": posfile, "start": start, "pad": pad})

    rows: List[dict] = []

    def _emit(rnd: int, is_cells: bool, movie_prefix: str, hal_boundary: str,
              hyb_idx: Optional[int] = None) -> None:
        for seg in enriched:
            tissue, pad = seg["tissue"], seg["pad"]
            if seg["kind"] == "boundary":
                # Shared movie name (no per-segment label): the continuous index
                # comes from start/pad, not from the name.
                imaging_type = "cells" if is_cells else "bits"
                series = f"{movie_prefix}_{{fov:0{pad}d}}"
                hal_config = hal_boundary
                data_dir = _seg_dir(tissue, "boundary", is_cells, hyb_idx)
            else:
                imaging_type = "transit"
                series = f"hal-{mic}-transit_r{rnd:02d}_{{fov:0{pad}d}}"
                hal_config = transit_hal_config
                data_dir = _seg_dir(tissue, "transit", is_cells)
            rows.append({
                "imaging_round":  rnd,
                "imaging_type":   imaging_type,
                "series":         series,
                "hal_config":     hal_config,
                "data_dir":       data_dir,
                "positions_file": seg["posfile"],
                "tissue":         tissue,
                "segment":        seg["label"],
                "fov_start":      seg["start"],
                "fov_pad":        pad,
            })

    # Round 1: cells.
    _emit(1, is_cells=True, movie_prefix=f"hal-{mic}-cells", hal_boundary=cells_hal_config)
    # Rounds 2…N+1: bits #1…#N (movie series number tracks the bit/hyb index).
    for bit_idx in range(1, n_bits + 1):
        _emit(bit_idx + 1, is_cells=False,
              movie_prefix=f"hal-{mic}_{bit_idx:02d}", hal_boundary=bits_hal_config,
              hyb_idx=bit_idx)

    return pd.DataFrame(
        rows,
        columns=["imaging_round", "imaging_type", "series", "hal_config",
                 "data_dir", "positions_file", "tissue", "segment",
                 "fov_start", "fov_pad"],
    )


# ── Dave config builder ────────────────────────────────────────────────────────

# ── create_dave_config helpers ───────────────────────────────────────────────

# The Kilroy configs in data/configs/kilroy/ only define "Hybridize"/
# "Hybridize Adaptors" protocols for hyb indices 1-24 (one physical
# fluidics port per protocol) -- a >24-round protocol (e.g. the 25-round
# lineage_tracing_lineage pipeline) needs the operator to physically
# reload an already-used port with fresh reagent for the extra round(s),
# so the KILROY PROTOCOL for hyb index 25 onward reuses port 2's name
# (26 -> port 3, ...), never port 1. Only the protocol called changes --
# the loop label and data folder for that round still use its true hyb
# index, so nothing collides with hyb 2's own folder/label.
MAX_KILROY_HYB = 24

# Leading comment for a recipe with any "multi" (variable-z-per-FOV) round:
# nothing else in the recipe hints that its positions files need a 3rd column.
_VARIABLE_Z_COMMENT = (
    "VARIABLE-Z-PER-FOV EXPERIMENT.\n"
    "The positions file(s) referenced below must carry a 3rd column\n"
    "per line naming the HAL parameters set (hal_config filename, no\n"
    ".xml extension) to use for that specific FOV, e.g.:\n"
    "  1234.5,987.6,hal-config-st2-bits-shallow-750f10_650f10_560f10\n"
    "A plain \"x,y\" line falls back to whatever HAL parameters are\n"
    "currently active. Requires the patched Dave described in\n"
    "misc/dave_multi_z/README.md (replaces storm_control/dave/\n"
    "xml_generators/v2Generator.py) -- a stock Dave will silently\n"
    "ignore the 3rd column and reuse whichever parameters were last set."
)


def _kilroy_hyb_idx(hyb_idx: int) -> int:
    """Kilroy hybridize-protocol index for *hyb_idx* (wraps past ``MAX_KILROY_HYB``)."""
    if hyb_idx <= MAX_KILROY_HYB:
        return hyb_idx
    return hyb_idx - (MAX_KILROY_HYB - 1)


def _round_has_bits(round_info: pd.DataFrame, rid: int) -> bool:
    """True if round *rid* images bits (not just cells / transit)."""
    rrows = round_info[round_info["imaging_round"] == rid]
    if "imaging_type" in round_info.columns:
        types = {str(t).strip().lower() for t in rrows["imaging_type"].dropna()}
        if types:
            return "bits" in types
    return any("cells" not in str(s) for s in rrows["series"])


def _hyb_idx(round_id: int, first_bits_round: Optional[int]) -> int:
    """Bit/hyb index (1-based) of imaging round *round_id* -- offset so a
    leading cells round never shifts it (round *first_bits_round* -> 1)."""
    if first_bits_round is not None and round_id >= first_bits_round:
        return round_id - first_bits_round + 1
    return round_id


def _add_movie(parent_loop: ET.Element, row: pd.Series, variable_name: str,
               settings_dir: Path, num_focus_checks: int) -> None:
    """Append one <movie> (resolving its HAL frame count) to *parent_loop*."""
    movie_name   = series_to_movie_name(str(row["series"]))
    hal_stem     = Path(str(row["hal_config"])).stem
    hal_path     = resolve_hal_config_path(settings_dir, hal_stem)
    try:
        n_frames = get_hal_frame_count(hal_path)
    except (FileNotFoundError, ValueError) as exc:
        n_frames = 0
        # <length>0</length> is a zero-frame movie in the real recipe, not just
        # a gap in the estimate, so say so.
        print(f"[create_dave_config] WARNING: could not read <frames> from "
              f"{hal_path} ({exc}) -- movie {movie_name!r} written with "
              f"<length>0</length>. Check that round_info.csv's hal_config "
              f"column ({hal_stem!r}) matches a real file in {settings_dir}.")

    movie   = ET.SubElement(parent_loop, "movie")
    name_el = ET.SubElement(movie, "name")
    name_el.set("increment", "Yes")
    name_el.text = movie_name
    # Patched Dave's <name start=… pad=…>: one running FOV index across
    # per-segment loops (see create_dave_config).
    if "fov_start" in row.index and pd.notna(row.get("fov_start")):
        name_el.set("start", str(int(row["fov_start"])))
    if "fov_pad" in row.index and pd.notna(row.get("fov_pad")):
        name_el.set("pad", str(int(row["fov_pad"])))
    # No static <length>/<parameters> for a "multi" (variable-z) round: Dave
    # reads each tag with ElementTree.find() (first match wins), so a static
    # value here would shadow the per-FOV one from the positions file.
    if row.get("tissue_thickness") != "multi":
        ET.SubElement(movie, "length").text     = str(n_frames)
        ET.SubElement(movie, "parameters").text = hal_stem
    cf = ET.SubElement(movie, "check_focus")
    ET.SubElement(cf, "num_focus_checks").text = str(num_focus_checks)
    ET.SubElement(cf, "focus_scan")
    ET.SubElement(movie, "overwrite").text = "False"
    ve = ET.SubElement(movie, "variable_entry")
    ve.set("name", variable_name)


def _between_round_protocols(
    hyb_idx:            int,
    skip_cleave:        bool,
    use_adaptors:       bool,
    fluidics_protocols: Optional[Sequence[str]],
    resolver:           Optional[KilroyProtocolResolver],
    kilroy_config:      Optional[Path],
) -> List[str]:
    """Kilroy protocol names for the fluidics block before hyb *hyb_idx*."""
    if fluidics_protocols is not None:
        fl_protocols = list(fluidics_protocols)
        if resolver is not None:
            resolver.validate(fl_protocols)
        return fl_protocols
    if resolver is not None:
        # Names taken from the Kilroy config (see kilroy_config).
        cleave = [] if skip_cleave else [resolver.cleave(adaptors=use_adaptors)]
        kilroy_hyb_idx = _kilroy_hyb_idx(hyb_idx)
        if use_adaptors:
            steps = [resolver.hybridize(kilroy_hyb_idx, adaptors=True), resolver.readouts()]
        else:
            steps = [resolver.hybridize(kilroy_hyb_idx, adaptors=False)]
        # Skip the image-buffer protocol when the preceding step already ends
        # by flowing the same valve (e.g. "Hybridize N"), or it flows twice.
        # Compare the last valve that actually FLOWED, not the last <valve>
        # element: a protocol can end with a bare valve move (see
        # protocol_last_flowed_valve).
        image_buffer = resolver.image_buffer()
        preceding_last_flow = protocol_last_flowed_valve(kilroy_config, steps[-1])
        buffer_last_flow    = protocol_last_flowed_valve(kilroy_config, image_buffer)
        already_flowed = bool(preceding_last_flow and buffer_last_flow
                              and preceding_last_flow.lower() == buffer_last_flow.lower())
        if not already_flowed:
            steps.append(image_buffer)
        return cleave + steps
    if use_adaptors:
        # Legacy hard-coded names (no Kilroy cross-check).
        return ([] if skip_cleave else ["Cleave adaptors"]) + [
            f"Hyb adaptors {hyb_idx}",
            "Hyb readouts",
            "Flow Image Buffer",
        ]
    return ([] if skip_cleave else ["Cleave direct"]) + [
        f"Hybridize {hyb_idx}",
        "Wash and Imaging Buffers",
    ]


def _fluidics_after(
    round_id:             int,
    is_last:              bool,
    first_bits_round:     Optional[int],
    first_hyb_no_cleave:  bool,
    include_final_cleave: bool,
    use_adaptors:         bool,
    fluidics_protocols:   Optional[Sequence[str]],
    resolver:             Optional[KilroyProtocolResolver],
    kilroy_config:        Optional[Path],
) -> Optional[Tuple[str, List[str]]]:
    """``(loop name, protocols)`` of the fluidics block that FOLLOWS *round_id*,
    or None if there is none.

    *round_id* need not itself exist in ``round_info`` -- only
    ``round_id + 1`` (the round this fluidics precedes) matters, so passing
    ``round_ids[0] - 1`` gives a LEADING fluidics block before a round_info
    slice's first round (see ``create_dave_config``'s ``leading_fluidics``).
    """
    if not is_last:
        next_round = round_id + 1
        # Hyb number tracks the bit/hyb index of the NEXT imaging round (not
        # the raw imaging_round number), so a leading cells round shifts
        # neither the Kilroy protocol numbers nor the loop's own name.
        hyb_idx = _hyb_idx(next_round, first_bits_round)
        # The fluidics that precedes the FIRST bits round omits the cleave.
        is_first_hyb = (first_bits_round is not None and next_round == first_bits_round)
        skip_cleave  = is_first_hyb and first_hyb_no_cleave
        return (f"Hyb {hyb_idx:02d} Fluidics",
                _between_round_protocols(hyb_idx, skip_cleave, use_adaptors,
                                         fluidics_protocols, resolver, kilroy_config))
    if include_final_cleave:
        if resolver is not None:
            return "Fluidics Final", [resolver.cleave(adaptors=use_adaptors)]
        return "Fluidics Final", ["Cleave adaptors" if use_adaptors else "Cleave direct"]
    return None


class _RecipeWriter:
    """One Dave recipe being built: its command sequence, loop variables and
    the currently active save directory."""

    def __init__(self, create_data_dirs: bool):
        self.root = ET.Element("recipe")
        self.seq  = ET.SubElement(self.root, "command_sequence")
        self.create_data_dirs = create_data_dirs
        self.imaging_loop_vars:  List[Tuple[str, str]]       = []
        self.fluidics_loop_vars: List[Tuple[str, List[str]]] = []
        self._created_dirs: set = set()
        self._current_dir:  Optional[str] = None   # last <change_directory> emitted

    def change_directory(self, dir_value) -> None:
        """
        Emit a ``<change_directory>`` (sets HAL's save dir for the FOLLOWING loop)
        from *dir_value*, and — when ``create_data_dirs`` — create that folder.

        HAL rejects a directory that does not exist, and nothing in Dave/HAL makes
        it, so the directory is created here. De-duplicated: emitting the directory
        that is already active is a no-op. No-op when *dir_value* is
        missing/blank/NaN (e.g. round_info has no ``data_dir`` column).
        """
        if not pd.notna(dir_value):
            return
        dpath = str(dir_value).strip()
        if not dpath or dpath == self._current_dir:
            return
        ET.SubElement(self.seq, "change_directory").text = dpath
        self._current_dir = dpath
        if self.create_data_dirs and dpath not in self._created_dirs:
            self._created_dirs.add(dpath)
            try:
                Path(dpath).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                # Non-fatal: recipe still written. Warn so the user creates the
                # folder on the acquisition machine (HAL requires it to exist).
                print(f"[create_dave_config] WARNING: could not create data dir "
                      f"{dpath!r}: {exc}. Create it on the acquisition computer "
                      f"before running Dave.")

    def imaging_loop(self, name: str, positions_path: str) -> ET.Element:
        """Append an imaging ``<loop>`` and its own same-named positions loop_variable."""
        loop = ET.SubElement(self.seq, "loop")
        loop.set("name", name)
        self.imaging_loop_vars.append((name, positions_path))
        return loop

    def fluidics_loop(self, block: Optional[Tuple[str, List[str]]]) -> None:
        """Append a fluidics ``<loop>`` from ``_fluidics_after``'s result (None: nothing)."""
        if block is None:
            return
        fl_name, fl_protocols = block
        fl_loop = ET.SubElement(self.seq, "loop")
        fl_loop.set("name", fl_name)
        ET.SubElement(fl_loop, "variable_entry").set("name", fl_name)
        self.fluidics_loop_vars.append((fl_name, fl_protocols))

    def write(self, output_path: Path, leading_comment: Optional[str]) -> None:
        """Append the loop variables and write the recipe XML."""
        # Grouped under two labeled comments so the two kinds of loop_variable
        # (position files vs. fluidics protocol lists) are easy to tell apart when
        # reading the raw XML.
        if self.imaging_loop_vars:
            self.root.append(ET.Comment(" POSITION VARIABLES "))
        for lname, pos_path in self.imaging_loop_vars:
            lv = ET.SubElement(self.root, "loop_variable")
            lv.set("name", lname)
            ET.SubElement(lv, "file_path").text = pos_path
        if self.fluidics_loop_vars:
            self.root.append(ET.Comment(" FLUIDICS VARIABLES "))
        for lname, protocols in self.fluidics_loop_vars:
            lv = ET.SubElement(self.root, "loop_variable")
            lv.set("name", lname)
            val = ET.SubElement(lv, "value")
            for protocol in protocols:
                ET.SubElement(val, "valve_protocol").text = protocol
        _write_dave_xml(self.root, Path(output_path), leading_comment=leading_comment)


def create_dave_config(
    round_info:           pd.DataFrame,
    positions_file:       Path,
    settings_dir:         Path,
    output_path:          Path,
    use_adaptors:         bool = False,
    include_final_cleave: bool = False,
    first_hyb_no_cleave:  bool = True,
    leading_fluidics:     bool = False,
    num_focus_checks:     int  = 50,
    fluidics_protocols:   Optional[Sequence[str]] = None,
    kilroy_config:        Optional[Path] = None,
    positions_dir:        Optional[Path] = None,
    create_data_dirs:     bool = True,
    print_estimate:       bool = True,
    per_round:            bool = True,
    microscope:           Optional[str] = None,
    estimate_frame_shape: Optional[Sequence[int]] = None,
    estimate_bytes_per_pixel: int = 2,
    start_time:           Optional[datetime] = None,
) -> Optional["ExperimentEstimate"]:
    """
    Write an explicit-block Dave recipe XML from ``round_info``.

    **Positions.** By default every movie iterates the one *positions_file*
    and each imaging round is one ``<loop>``. When ``round_info`` has a
    ``positions_file`` column and *positions_dir* is given (the multi-boundary
    layout from :func:`create_round_info_multitissue`), each segment
    (boundary or transit) gets its own ``<loop>``, named
    ``"<Cells Imaging|Hyb NN Imaging> - <segment>"``, in ``round_info`` row
    order. Fluidics loops go after a round's last segment loop.

    **One ``<loop_variable>`` per loop, always.** Dave's ``v2Generator``
    looks each ``<loop>`` up by its own name in the loop variables, so every
    loop needs a same-named ``<loop_variable>``, even when rounds share a
    positions file. (A shared variable raises ``ValueError: 'Hyb NN Imaging'
    is not in list`` in real Dave.)

    Rows with ``fov_start``/``fov_pad`` (from
    :func:`create_round_info_multitissue`) get ``start``/``pad`` attributes
    on the movie ``<name>``, so all boundary movies (and all transit movies)
    share one name with one running FOV index. This needs the patched Dave
    ``v2Generator`` (``dave_fov_offset_patch``); stock Dave ignores the
    attributes and the names would collide.

    **Fluidics.** Each fluidics loop is named by the hyb index of the NEXT
    imaging round ("Hyb 01 Fluidics" precedes "Hyb 01 Imaging"). The hyb index
    counts bits rounds, so a leading cells round shifts neither loop names nor
    Kilroy protocol numbers. Past ``MAX_KILROY_HYB`` (24 ports) the Kilroy
    protocol wraps to port 2 (25 -> 2, 26 -> 3, ...; the port is reloaded
    with fresh reagent), while loop labels and data folders keep the true
    index. The last round has no fluidics unless *include_final_cleave*.

    **Save directory.** With a ``data_dir`` column, a ``<change_directory>``
    goes right before each round's imaging loop (and before each extra
    segment directory in the multi-boundary layout), skipped when unchanged.
    HAL errors if the directory doesn't exist and nothing else creates it, so
    *create_data_dirs* creates them. (``change_directory`` is HAL's deprecated
    but working "Set Directory".)

    Parameters
    ----------
    round_info            : columns ``imaging_round``, ``series``,
                            ``hal_config``, optionally ``imaging_type``,
                            ``data_dir``, ``positions_file``/``segment``,
                            ``fov_start``/``fov_pad``, and:

                            * ``tissue_thickness`` (``"single"``, default, or
                              ``"multi"``): a ``"multi"`` row's movie omits
                              ``<length>``/``<parameters>``, which the positions
                              file then supplies per FOV via the patched Dave in
                              ``misc/dave_multi_z/`` (see ``_add_movie``). Its
                              ``hal_config`` must still name a real file. Any
                              ``"multi"`` row adds a leading XML comment stating
                              the positions-file requirement.
                            * ``z_lengths``: informational only, not read here
                              (JSON list of the round's frame counts).
    positions_file        : ``positions_*.txt`` written into each
                            ``<loop_variable>/<file_path>``
    settings_dir          : directory with the HAL config XMLs (for ``<frames>``)
    output_path           : where to write the recipe XML
    use_adaptors          : True = adaptor fluidics (cleave adaptors, hyb
                            adaptors N, readouts, image buffer); False = direct
                            (cleave direct, hybridize N, wash and imaging buffers)
    include_final_cleave  : append a "Fluidics Final" block with one cleave step
                            after the last round
    first_hyb_no_cleave   : omit the cleave before the FIRST bits round (a cells
                            round was imaged first on a fresh sample). Ignored
                            with *fluidics_protocols*.
    leading_fluidics      : also emit a fluidics block before the first round,
                            as if round ``round_ids[0] - 1`` had just been
                            imaged. Makes a hybs-only recipe runnable on its own.
                            No-op for an empty ``round_info``.
    num_focus_checks      : value for ``<num_focus_checks>``
    fluidics_protocols    : fixed Kilroy protocol list for every between-round
                            block; overrides *use_adaptors*
    kilroy_config         : Kilroy config XML that will run the experiment. When
                            given, every protocol name is taken from it and must
                            exist there (``ValueError`` otherwise). ``None`` uses
                            hard-coded names without checking.
    positions_dir         : folder holding the per-segment positions files
    create_data_dirs      : create every ``data_dir`` folder (default True). Set
                            False off the acquisition computer.
    print_estimate        : print run time and raw storage
                            (:func:`estimate_dave_experiment`); the fluidics part
                            needs *kilroy_config*
    per_round             : include the "Per round:" breakdown in that print
                            (turn off when showing :func:`experiment_estimate_table`)
    microscope            : scope id, for the frame size in the storage estimate;
                            ``None`` infers it from ``series`` names
    estimate_frame_shape  : explicit ``(width, height)`` for the estimate;
                            overrides *microscope*
    estimate_bytes_per_pixel : bytes per pixel for the estimate (2 = uint16)
    start_time            : expected recipe start; adds per-round start/end
                            dates to the printed estimate

    Returns
    -------
    ExperimentEstimate or None
        The estimate when *print_estimate* is True, else None.
    """
    round_ids = sorted(round_info["imaging_round"].unique())

    # Per-segment layout: active when round_info carries a positions_file column
    # AND a positions_dir is given (the multi-boundary layout). Otherwise every
    # movie in a round shares the single positions_file.
    segment_mode  = ("positions_file" in round_info.columns and positions_dir is not None)
    positions_dir = Path(positions_dir) if positions_dir is not None else None

    # Resolve fluidic protocol names against the Kilroy config that will run this
    # experiment, so every protocol written here exists as a Kilroy <protocol>.
    resolver = (
        KilroyProtocolResolver(load_kilroy_protocols(kilroy_config))
        if kilroy_config is not None else None
    )
    bits_round_ids   = [rid for rid in round_ids if _round_has_bits(round_info, rid)]
    first_bits_round = bits_round_ids[0] if bits_round_ids else None
    fluidics_kw = dict(
        first_bits_round=first_bits_round, first_hyb_no_cleave=first_hyb_no_cleave,
        include_final_cleave=include_final_cleave, use_adaptors=use_adaptors,
        fluidics_protocols=fluidics_protocols, resolver=resolver, kilroy_config=kilroy_config,
    )
    writer = _RecipeWriter(create_data_dirs)

    if leading_fluidics and round_ids:
        writer.fluidics_loop(_fluidics_after(round_ids[0] - 1, False, **fluidics_kw))

    for idx, round_id in enumerate(round_ids):
        is_last = (idx == len(round_ids) - 1)
        rows    = round_info[round_info["imaging_round"] == round_id]
        # Fixed "Cells Imaging" for the (single) non-bits round.
        label   = (f"Hyb {_hyb_idx(round_id, first_bits_round):02d} Imaging"
                   if round_id in bits_round_ids else "Cells Imaging")

        if segment_mode:
            # One loop per (round, segment) -- a Dave loop iterates a single
            # positions file. Each loop gets its OWN same-named loop_variable
            # (see "One <loop_variable> per loop" in the docstring), and each
            # segment sets its own save directory just before its loop.
            for _, row in rows.iterrows():
                seg   = str(row.get("segment", "")).strip() or series_to_movie_name(str(row["series"]))
                lname = f"{label} - {seg}"
                writer.change_directory(row.get("data_dir"))
                loop = writer.imaging_loop(lname, str(positions_dir / str(row["positions_file"])))
                _add_movie(loop, row, lname, settings_dir, num_focus_checks)
        else:
            # Single loop for the round; all movies share positions_file and one
            # save directory (from the round's first row's data_dir).
            writer.change_directory(rows.iloc[0].get("data_dir"))
            loop = writer.imaging_loop(label, str(positions_file))
            for _, row in rows.iterrows():
                _add_movie(loop, row, label, settings_dir, num_focus_checks)

        writer.fluidics_loop(_fluidics_after(round_id, is_last, **fluidics_kw))

    is_variable_z = ("tissue_thickness" in round_info.columns
                     and (round_info["tissue_thickness"] == "multi").any())
    writer.write(output_path, _VARIABLE_Z_COMMENT if is_variable_z else None)

    if print_estimate:
        return _print_estimate(output_path, kilroy_config, settings_dir,
                               microscope or _infer_microscope(round_info), estimate_frame_shape,
                               estimate_bytes_per_pixel, per_round, start_time)
    return None


def _print_estimate(output_path, kilroy_config, settings_dir, microscope, frame_shape,
                    bytes_per_pixel, per_round, start_time) -> "ExperimentEstimate":
    """Estimate the recipe just written and print the report. An explicit
    *frame_shape* wins over *microscope*'s camera size."""
    if frame_shape is not None:
        frame_w, frame_h = int(frame_shape[0]), int(frame_shape[1])
    else:
        frame_w, frame_h = get_camera_frame_size(microscope)
    est = estimate_dave_experiment(
        Path(output_path),
        kilroy_config   = kilroy_config,
        settings_dir    = settings_dir,
        frame_width     = frame_w,
        frame_height    = frame_h,
        bytes_per_pixel = bytes_per_pixel,
    )
    print(format_experiment_estimate(est, per_round=per_round, start_time=start_time))
    return est


# ── Focus-lock test recipe ───────────────────────────────────────────────────

def create_focus_test_dave_config(
    positions_file:   Path,
    output_path:      Path,
    num_focus_checks: int = 50,
    focus_scan:       bool = True,
    n_test_frames:    int = 0,
    hal_config:       Optional[str] = None,
    settings_dir:     Optional[Path] = None,
    data_dir:         Optional[Path] = None,
    create_data_dir:  bool = True,
    movie_name:       str = "focustest",
    print_estimate:   bool = True,
    per_round:        bool = True,
    kilroy_config:    Optional[Path] = None,
    microscope:       Optional[str] = None,
    estimate_frame_shape: Optional[Sequence[int]] = None,
    estimate_bytes_per_pixel: int = 2,
    start_time:       Optional[datetime] = None,
) -> Tuple[int, Optional["ExperimentEstimate"]]:
    """
    Write a Dave recipe that visits every FOV in *positions_file* and checks
    focus lock only (no fluidics), to catch a bad lock across the coverslip
    before the full acquisition.

    **Check-only by default (``n_test_frames=0``).** Each ``<movie>`` has only
    ``<name>``/``<check_focus>``. Dave's ``v2Generator`` skips any action
    missing its fields (``DASetParameters`` needs ``parameters``,
    ``DATakeMovie`` needs ``length > 0``), so each FOV is just move stage +
    check focus: no image, no HAL parameter change. Works with stock Dave.

    This mode leaves no per-FOV record on disk: Dave shows focus failures only
    in its in-memory GUI warnings, and HAL's ``.off`` sidecar (``good-offset``,
    read by :mod:`MERci.analysis.stage_z`) is only written once movie frames
    arrive.

    **``n_test_frames > 0`` gives a per-FOV record.** Every movie gets
    ``<length>`` (the real frame count HAL takes), ``<parameters>`` (from
    *hal_config*, required) and ``<overwrite>True</overwrite>``. Each FOV then
    writes a short movie and its ``.off`` sidecar, readable with
    :func:`MERci.analysis.stage_z.focus_lock_summary_for_fov`. Costs a little
    time and disk, and the data directory must exist.

    Parameters
    ----------
    positions_file   : positions_*.txt to visit
    output_path      : where to write the recipe XML
    num_focus_checks : ``<num_focus_checks>`` per FOV
    focus_scan       : include ``<focus_scan/>`` (scan if not locked); False
                       only checks the current lock
    n_test_frames    : 0 = check-only; > 0 = also a movie of this many frames
    hal_config       : HAL config filename (with or without ``.xml``) for the
                       movies; required when ``n_test_frames > 0``, else ignored
    settings_dir     : with ``n_test_frames > 0``, check that *hal_config*
                       exists (same lookup as :func:`create_dave_config`,
                       including the ``multi_z/`` sibling)
    data_dir         : with ``n_test_frames > 0``, emit a ``<change_directory>``
                       to it before the loop
    create_data_dir  : create *data_dir* when given (HAL requires it)
    movie_name       : base movie name (e.g. ``"hal-mf3-focustest"``), numbered
                       per FOV (``increment="Yes"``)
    print_estimate   : print a run-time/storage estimate
                       (:func:`estimate_dave_experiment`). In check-only mode
                       the time is only the per-FOV move overhead.
    per_round        : as in :func:`create_dave_config`
    kilroy_config    : unused (no fluidics); accepted so callers can pass the
                       same arguments as to :func:`create_dave_config`
    microscope, estimate_frame_shape, estimate_bytes_per_pixel, start_time :
                       as in :func:`create_dave_config`

    Returns
    -------
    (int, ExperimentEstimate or None)
        Number of FOVs visited, and the estimate when *print_estimate* is True.
    """
    if n_test_frames > 0 and not hal_config:
        raise ValueError(
            "hal_config is required when n_test_frames > 0 -- HAL needs real "
            "parameters to take a movie, even a 1-frame one."
        )

    hal_stem = None
    if n_test_frames > 0:
        hal_stem = Path(hal_config).stem
        if settings_dir is not None:
            hal_path = resolve_hal_config_path(settings_dir, hal_stem)
            if not hal_path.exists():
                raise FileNotFoundError(
                    f"hal_config {hal_stem!r} not found in {settings_dir} "
                    f"(or its multi_z/ sibling) -- check the filename."
                )

    n_fovs = count_positions(positions_file)
    if n_fovs == 0:
        raise ValueError(f"No FOVs found in {positions_file}.")

    root = ET.Element("recipe")
    seq  = ET.SubElement(root, "command_sequence")

    if n_test_frames > 0 and data_dir is not None:
        ET.SubElement(seq, "change_directory").text = str(data_dir)
        if create_data_dir:
            Path(data_dir).mkdir(parents=True, exist_ok=True)

    loop = ET.SubElement(seq, "loop")
    loop.set("name", "Focus Test")

    movie   = ET.SubElement(loop, "movie")
    name_el = ET.SubElement(movie, "name")
    name_el.set("increment", "Yes")
    name_el.text = movie_name
    if n_test_frames > 0:
        ET.SubElement(movie, "length").text     = str(n_test_frames)
        ET.SubElement(movie, "parameters").text = hal_stem
    cf = ET.SubElement(movie, "check_focus")
    ET.SubElement(cf, "num_focus_checks").text = str(num_focus_checks)
    if focus_scan:
        ET.SubElement(cf, "focus_scan")
    if n_test_frames > 0:
        ET.SubElement(movie, "overwrite").text = "True"
    ve = ET.SubElement(movie, "variable_entry")
    ve.set("name", "Focus Test")

    lv = ET.SubElement(root, "loop_variable")
    lv.set("name", "Focus Test")
    ET.SubElement(lv, "file_path").text = str(positions_file)

    _write_dave_xml(root, Path(output_path))

    est = None
    if print_estimate:
        est = _print_estimate(output_path, kilroy_config, settings_dir, microscope,
                              estimate_frame_shape, estimate_bytes_per_pixel, per_round, start_time)
    return n_fovs, est


# ── Dave annotation ────────────────────────────────────────────────────────────

def annotate_dave_with_round_info(
    dave_path:       Path,
    round_bit_color: list[tuple],
) -> None:
    """
    Insert XML comments into an existing Dave recipe XML describing which bits
    are imaged in each round.

    For round 1: comment is placed before the ``<loop name="Cells Imaging">``
    block.  For rounds 2+: comment is placed before the corresponding
    ``<loop name="Hyb NN Fluidics">`` block (which precedes that imaging
    round; hyb index NN = imaging-round index − 1).  A blank line is inserted
    before each comment for readability.

    In the default cells-first layout, imaging round 1 is the cells acquisition
    (no bits), so it normally has no entry here; the bits comments attach to the
    ``Hyb NN Fluidics`` loops for rounds 2…N+1.  The ``round_1indexed`` values
    passed in must therefore be **imaging-round** indices (bits start at 2), not
    bit/hyb indices — see ``notebooks/before_imaging/04``.

    Parameters
    ----------
    dave_path       : path to the Dave XML file to annotate (modified in-place)
    round_bit_color : list of ``(round_1indexed, bit_number, color_nm)`` tuples
    """
    # Group bits by round and build comment strings
    bits_by_round: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for round_1idx, bit, color_nm in round_bit_color:
        bits_by_round[round_1idx].append((bit, color_nm))

    round_comments: dict[int, list[str]] = {}
    for round_1idx, bit_colors in sorted(bits_by_round.items()):
        round_comments[round_1idx] = [
            f"Bit {bit} ({color} nm)"
            for bit, color in sorted(bit_colors, key=lambda x: x[1], reverse=True)
        ]

    # Read file preserving raw CRLF so split("\r\n") works correctly
    with open(dave_path, "r", encoding="ISO-8859-1", newline="") as fh:
        content = fh.read()

    lines = content.split("\r\n")
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        indent   = line[: len(line) - len(line.lstrip())]

        def _append_comment(n: int) -> None:
            if n not in round_comments:
                return
            # n is an imaging-round index (see docstring); relabel to match
            # the loop naming it sits next to -- "Cells" for round 1, else
            # the hyb index (n - 1, since round 1 is always cells).
            label = "Cells" if n == 1 else f"Hyb {n - 1:02d}"
            new_lines.append("")
            new_lines.append(f"{indent}<!-- {label}:")
            for s in round_comments[n]:
                new_lines.append(f"{indent}        {s}")
            new_lines.append(f"{indent}-->")

        # Round 1: insert before "Cells Imaging" loop
        if stripped == '<loop name="Cells Imaging">':
            _append_comment(1)
        else:
            # Rounds 2+: insert before the corresponding "Hyb NN Fluidics" loop
            # -- hyb index NN corresponds to imaging-round index NN + 1, since
            # round 1 is always cells.
            m = re.match(r'^<loop name="Hyb (\d+) Fluidics">', stripped)
            if m:
                _append_comment(int(m.group(1)) + 1)

        new_lines.append(line)

    content = "\r\n".join(new_lines)
    with open(dave_path, "w", encoding="ISO-8859-1", newline="") as fh:
        fh.write(content)


# ── Experiment time / storage estimate ──────────────────────────────────────────

@dataclass
class ExperimentEstimate:
    """
    Estimated run time and storage for a Dave recipe.

    Attributes
    ----------
    total_time_s    : imaging_time_s + fluidics_time_s (seconds)
    imaging_time_s  : total time acquiring movies (Σ frames × frame_time over every
                      FOV-movie)
    fluidics_time_s : total time in between-round fluidics (Σ Kilroy protocol
                      durations)
    total_bytes     : total raw image size (Σ frames × bytes_per_frame over every
                      FOV-movie)
    n_fov_movies    : number of per-FOV movies acquired across the whole experiment
    per_fov_time_s  : total imaging time spent on ONE FOV across the whole experiment
                      (Σ, over every round, of that round's own per-FOV movie time --
                      i.e. what one FOV alone would take, not multiplied by FOV count)
    per_round       : list of ``{"hyb", "label", "imaging_s", "fluidics_s",
                      "bytes", "movies", "series", "per_fov_imaging_s"}`` dicts --
                      "Cells Imaging" first, then hyb-numbered rows (``"hyb"`` =
                      bit/hyb index) in order, then any other non-numbered row (e.g.
                      the closing "Fluidics Final" step, labeled by its own
                      loop name rather than a fabricated hyb number).
                      ``series`` is the list of distinct movie names (e.g.
                      ``"hal-st2_01"``) imaged in that round.
                      ``per_fov_imaging_s`` is that round's own contribution to
                      ``per_fov_time_s`` above.
    assumptions     : human-readable list of the numbers assumed (frame size, frame
                      time source, …)
    warnings        : anything that made the estimate approximate (missing positions
                      file, unknown protocol, …)
    """
    total_time_s:    float
    imaging_time_s:  float
    fluidics_time_s: float
    total_bytes:     int
    n_fov_movies:    int
    per_fov_time_s:  float = 0.0
    per_round:       List[dict] = field(default_factory=list)
    assumptions:     List[str]  = field(default_factory=list)
    warnings:        List[str]  = field(default_factory=list)


def estimate_dave_experiment(
    dave_recipe:          Path,
    *,
    frame_width:          int,
    frame_height:         int,
    kilroy_config:        Optional[Path] = None,
    settings_dir:         Optional[Path] = None,
    bytes_per_pixel:      int   = 2,
    frame_time_s:         Optional[float] = None,
    readout_overhead_s:   float = 0.0,
    per_movie_overhead_s: float = 5.0,
) -> ExperimentEstimate:
    """
    Estimate total run time and raw storage for a written Dave recipe.

    Reproduces the estimate Dave itself shows (which it obtains from HAL/Kilroy
    test-mode responses):

    * **movie duration** = ``frames / fps`` — here ``frames × frame_time`` where
      ``frame_time`` is the HAL config's ``exposure_time`` (+ ``readout_overhead_s``)
      or the explicit ``frame_time_s`` (HAL: ``tcpControl.calculateMovieStats``);
    * **movie size** = ``frames × frame_width × frame_height × bytes_per_pixel``
      (HAL: ``bytes_per_frame × frames``);
    * **fluidics duration** = Σ of the named protocol's step durations
      (Kilroy: ``KilroyProtocols.requiredTime``).

    Each imaging loop is multiplied by the FOV count of its positions file (read
    from the recipe's ``<loop_variable>/<file_path>``).

    Camera frame geometry is not stored in the MERci HAL config, so
    ``frame_width``/``frame_height`` are parameters (take them from
    ``configs.get_camera_frame_size``), as is ``bytes_per_pixel`` (default
    2 = uint16).

    Parameters
    ----------
    dave_recipe          : path to the recipe XML written by :func:`create_dave_config`
    kilroy_config        : Kilroy config XML; source of fluidic protocol durations.
                           When None, fluidics time is not estimated (reported 0).
    settings_dir         : directory with the HAL config XMLs; used to read each
                           movie's ``exposure_time`` when ``frame_time_s`` is None
    frame_width          : camera frame width in pixels
    frame_height         : camera frame height in pixels
    bytes_per_pixel      : bytes per pixel (2 for uint16)
    frame_time_s         : fixed per-frame time (s); overrides the HAL exposure read
    readout_overhead_s   : added to each HAL ``exposure_time`` (camera readout etc.)
    per_movie_overhead_s : fixed seconds added per FOV-movie (stage move, focus, …);
                           Dave's own estimate omits this -- default 5 s assumes the
                           time to move to a new FOV and start acquiring

    Returns
    -------
    ExperimentEstimate
    """
    recipe = Path(dave_recipe)
    root   = ET.fromstring(recipe.read_text(encoding="ISO-8859-1"))
    seq    = root.find("command_sequence")
    if seq is None:
        raise ValueError(f"No <command_sequence> in {recipe}")

    # Map each loop_variable name → its expansion: a positions file (imaging) or a
    # list of Kilroy protocol names (fluidics).
    lv_kind:  Dict[str, str]        = {}
    lv_value: Dict[str, object]     = {}
    for lv in root.findall("loop_variable"):
        name = lv.get("name", "")
        fp   = lv.find("file_path")
        if fp is not None:
            lv_kind[name], lv_value[name] = "file", (fp.text or "").strip()
        else:
            prots = [vp.text.strip() for vp in lv.iter("valve_protocol")
                     if vp.text and vp.text.strip()]
            lv_kind[name], lv_value[name] = "fluidics", prots

    proto_dur   = load_protocol_durations(kilroy_config) if kilroy_config else {}
    frame_bytes = frame_width * frame_height * bytes_per_pixel
    warnings_list: List[str] = []

    exposure_cache: Dict[str, Optional[float]] = {}
    def _frame_time(hal_stem: str) -> float:
        if frame_time_s is not None:
            return frame_time_s
        if hal_stem not in exposure_cache:
            exp = read_hal_exposure_time(resolve_hal_config_path(settings_dir, hal_stem)) \
                  if settings_dir is not None else None
            exposure_cache[hal_stem] = exp
        exp = exposure_cache[hal_stem]
        if exp is None:
            exp = 0.25   # fallback matching create_hal_config's default exposure
        return exp + readout_overhead_s

    fov_cache: Dict[str, int] = {}
    def _n_fovs(path_str: str) -> int:
        if path_str not in fov_cache:
            p = Path(path_str)
            if p.exists():
                fov_cache[path_str] = count_positions(p)
            else:
                warnings_list.append(f"positions file not found, FOV count taken as 0: {path_str}")
                fov_cache[path_str] = 0
        return fov_cache[path_str]

    imaging_time = fluidics_time = 0.0
    total_bytes  = 0
    n_movies     = 0
    per_round: Dict[object, dict] = {}

    def _hyb_no(lname: str) -> Optional[int]:
        """Hyb index from a loop name like "Hyb 01 Imaging"/"Hyb 01 Fluidics"
        (ignoring any trailing " - <segment>" suffix), or None for "Cells
        Imaging"/"Fluidics Final" (and their segment-mode variants), which
        have no hyb number to group by."""
        m = re.search(r"Hyb (\d+)", lname)
        return int(m.group(1)) if m else None

    for loop in seq.findall("loop"):
        lname  = loop.get("name", "")
        hno    = _hyb_no(lname)
        # Loops without a hyb number in their name ("Cells Imaging", "Fluidics
        # Final", and their segment-mode variants) get their own row keyed and
        # labeled by their real name, instead of being silently folded into a
        # fabricated "Hyb 00" -- which not only mislabeled the step but could
        # sort it out of its true position in the report.
        key    = hno if hno is not None else lname
        rec    = per_round.setdefault(
            key, {"hyb": hno, "label": (f"Hyb {hno:02d}" if hno is not None else lname),
                  "imaging_s": 0.0, "fluidics_s": 0.0, "bytes": 0, "movies": 0, "series": [],
                  "per_fov_imaging_s": 0.0})
        movies = loop.findall("movie")
        if movies:                                   # imaging loop
            # Read the positions variable from the movie's own
            # <variable_entry> (every movie in a loop uses the same one).
            ve_el    = movies[0].find("variable_entry")
            var_name = ve_el.get("name", "") if ve_el is not None else lname
            path = lv_value.get(var_name, "")
            n_fovs = _n_fovs(path) if lv_kind.get(var_name) == "file" else 0
            loop_time = 0.0
            loop_bytes = 0
            for mv in movies:
                length_el = mv.find("length")
                frames    = int(length_el.text) if (length_el is not None and length_el.text) else 0
                par_el    = mv.find("parameters")
                hal_stem  = (par_el.text or "").strip() if par_el is not None else ""
                loop_time  += frames * _frame_time(hal_stem) + per_movie_overhead_s
                loop_bytes += frames * frame_bytes
                name_el     = mv.find("name")
                series_name = (name_el.text or "").strip() if name_el is not None else ""
                if series_name and series_name not in rec["series"]:
                    rec["series"].append(series_name)
            imaging_time += n_fovs * loop_time
            total_bytes  += n_fovs * loop_bytes
            n_movies     += n_fovs * len(movies)
            rec["imaging_s"] += n_fovs * loop_time
            rec["bytes"]     += n_fovs * loop_bytes
            rec["movies"]    += n_fovs * len(movies)
            # Time to image ONE FOV in this round -- not multiplied by n_fovs.
            # Segment mode can fold several same-round loops (one per tissue
            # segment) into this same rec; they all share the same movie
            # template, so this is set (not summed) each time.
            rec["per_fov_imaging_s"] = loop_time
        else:                                        # fluidics loop
            for prot in lv_value.get(lname, []):
                if prot in proto_dur:
                    fluidics_time    += proto_dur[prot]
                    rec["fluidics_s"] += proto_dur[prot]
                elif kilroy_config is not None:
                    warnings_list.append(f"protocol not in Kilroy config, 0 s assumed: {prot!r}")

    if kilroy_config is None:
        warnings_list.append("no Kilroy config given: fluidics time not estimated (reported as 0)")

    assumptions = [
        f"frame {frame_width}×{frame_height} × {bytes_per_pixel} B = "
        f"{frame_bytes / 2**20:.1f} MiB/frame",
        ("frame time = fixed %.3f s" % frame_time_s) if frame_time_s is not None
        else "frame time = HAL exposure_time"
             + (f" + {readout_overhead_s:.3f} s readout" if readout_overhead_s else "")
             + " (fallback 0.25 s)",
    ]
    if per_movie_overhead_s:
        assumptions.append(f"per-movie overhead = {per_movie_overhead_s:.2f} s")

    return ExperimentEstimate(
        total_time_s    = imaging_time + fluidics_time,
        imaging_time_s  = imaging_time,
        fluidics_time_s = fluidics_time,
        total_bytes     = total_bytes,
        n_fov_movies    = n_movies,
        per_fov_time_s  = sum(r["per_fov_imaging_s"] for r in per_round.values()),
        # "Cells Imaging" (and its segment-mode "Cells Imaging - <segment>"
        # variants) first (round 1 always runs first), then hyb-numbered rows
        # ascending, then any other non-numbered row (e.g. "Fluidics Final")
        # last, in the order it was first seen -- `sorted` is stable, so ties
        # within a tier keep dict insertion order.
        per_round       = [per_round[k] for k in
                           sorted(per_round, key=lambda k: (
                               (1, k) if isinstance(k, int) else
                               (0, 0) if str(k).startswith("Cells") else
                               (2, 0)
                           ))],
        assumptions     = assumptions,
        warnings        = warnings_list,
    )


def _fmt_duration(seconds: float) -> str:
    """Format seconds as ``Dd, Hh, Mm`` (e.g. ``"0d, 11h, 17m"``), truncating to
    the minute. Falls back to ``"Ns"`` under a minute -- otherwise a short
    duration like "Time per FOV" would always round down to "0d, 0h, 0m"."""
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    total //= 60
    d, rem = divmod(total, 1440)
    h, m   = divmod(rem, 60)
    return f"{d}d, {h}h, {m}m"


def _fmt_bytes(n: int) -> str:
    """Format a byte count in binary units (KiB/MiB/GiB/TiB)."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TiB"


def _fmt_dt(dt: datetime) -> str:
    """Format a datetime as ``Month D YYYY, HH:MM`` (e.g. ``"September 5 2026,
    13:05"``). Built without ``%-d``/``%#d`` (no-leading-zero day) since
    those platform-specific strftime extensions raise ``ValueError`` on
    Windows -- ``dt.day`` gives the same result portably."""
    return f"{dt.strftime('%B')} {dt.day} {dt.year}, {dt.strftime('%H:%M')}"


def format_experiment_estimate(
    est:        ExperimentEstimate,
    per_round:  bool = False,
    start_time: Optional[datetime] = None,
) -> str:
    """
    Render an :class:`ExperimentEstimate` as a readable multi-line report.

    Parameters
    ----------
    per_round  : if True, also list each round's own fluidics/imaging breakdown
    start_time : when given, the recipe's assumed real start time -- adds an
                 overall Start/End and, per round, a start/end date for its
                 fluidics and imaging phases, computed by walking the rounds
                 in run order and accumulating each phase's duration (fluidics
                 before imaging within a round, matching the recipe's own
                 acquisition order -- see this module's docstring)
    """
    lines = [
        "Estimated experiment cost",
        f"  FOV-movies:    {est.n_fov_movies}",
        f"  Imaging time:  {_fmt_duration(est.imaging_time_s)}",
        f"  Fluidics time: {_fmt_duration(est.fluidics_time_s)}",
        f"  Total time:    {_fmt_duration(est.total_time_s)}",
        f"  Time per FOV:  {_fmt_duration(est.per_fov_time_s)}",
        f"  Storage:       {_fmt_bytes(est.total_bytes)}",
    ]
    if start_time is not None:
        lines.append(f"  Start:         {_fmt_dt(start_time)}")
        lines.append(f"  End:           {_fmt_dt(start_time + timedelta(seconds=est.total_time_s))}")
    if per_round and est.per_round:
        lines.append("  Per round:")
        for r, phases in _round_phases(est, start_time):
            series_str = ", ".join(r.get("series", [])) or "-"
            lines.append(f"    {r['label']} [{series_str}]:")
            for kind, dur, start, end in phases:
                lines.append(f"      {kind.capitalize()}")
                lines.append(f"        total time: {_fmt_duration(dur)}")
                if start is not None:
                    lines.append(f"        start date: {_fmt_dt(start)}")
                    lines.append(f"        end date:   {_fmt_dt(end)}")
            lines.append(f"      Total movies: {r['movies']}")
            lines.append(f"      Total size:   {_fmt_bytes(r['bytes'])}")
    if est.assumptions:
        lines.append("  Assumptions:")
        for a in est.assumptions:
            lines.append(f"    - {a}")
    if est.warnings:
        lines.append("  Warnings:")
        for w in dict.fromkeys(est.warnings):     # de-duplicate, keep order
            lines.append(f"    - {w}")
    return "\n".join(lines)


def experiment_estimate_table(est: ExperimentEstimate, start_time: datetime) -> pd.DataFrame:
    """
    Per-round fluidics/imaging breakdown as a dataframe, one row per phase,
    chained off ``start_time`` in run order -- the tabular counterpart of
    :func:`format_experiment_estimate`'s own "Per round:" text section
    (``dave_timing_accuracy.ipynb``'s block-by-block table uses this same
    style). Columns: ``block``, ``kind`` ("fluidics"/"imaging"), ``start``,
    ``end``, ``duration`` (the latter three pre-formatted via ``_fmt_dt``/
    ``_fmt_duration``, ready to ``display()`` as-is).
    """
    rows: List[dict] = []
    for r, phases in _round_phases(est, start_time):
        # A "Hyb NN" label (r["hyb"] is not None) is a bare number needing a
        # " Fluidics"/" Imaging" suffix; a non-hyb label ("Cells Imaging",
        # "Fluidics Final", ...) is the raw loop name and already names its
        # one phase -- see estimate_dave_experiment's per_round docstring.
        if r["hyb"] is not None:
            fluidics_label, imaging_label = f"{r['label']} Fluidics", f"{r['label']} Imaging"
        else:
            fluidics_label = imaging_label = r["label"]
        for kind, dur, start, end in phases:
            rows.append({"block": fluidics_label if kind == "fluidics" else imaging_label,
                         "kind": kind, "start": _fmt_dt(start), "end": _fmt_dt(end),
                         "duration": _fmt_duration(dur)})
    return pd.DataFrame(rows, columns=["block", "kind", "start", "end", "duration"])


def _round_phases(est: ExperimentEstimate, start_time: Optional[datetime]):
    """
    Yield ``(round, [(kind, seconds, start, end), ...])`` per round in run
    order: fluidics (if any) then imaging (if any movies), chained off
    *start_time* (start/end are None without one).
    """
    cursor = start_time
    for r in est.per_round:
        phases = []
        for kind, dur, present in (("fluidics", r["fluidics_s"], r["fluidics_s"] > 0),
                                   ("imaging", r["imaging_s"], r["movies"] > 0)):
            if not present:
                continue
            end = cursor + timedelta(seconds=dur) if cursor is not None else None
            phases.append((kind, dur, cursor, end))
            cursor = end
        yield r, phases


# ── XML writer ─────────────────────────────────────────────────────────────────

def _write_dave_xml(root: ET.Element, output_path: Path, leading_comment: Optional[str] = None) -> None:
    """
    Serialize the recipe with indentation and CRLF line endings.

    Parameters
    ----------
    leading_comment : optional text inserted as an XML comment immediately
        after the ``<?xml ... ?>`` declaration, before ``<recipe>`` -- each
        line wrapped in its own ``<!-- ... -->`` so it reads cleanly even in
        a plain text editor. ``None`` (default) writes the file exactly as
        before.
    """
    raw  = ET.tostring(root, encoding="utf-8")
    dom  = minidom.parseString(raw)
    text = dom.toprettyxml(indent="    ", encoding="ISO-8859-1").decode("ISO-8859-1")

    # Remove the extra blank line toprettyxml adds before every element
    text = re.sub(r"\n[ \t]*\n", "\n", text)

    # Restore one blank line after the POSITION/FLUIDICS VARIABLES section
    # comments (create_dave_config), for readability -- the strip above would
    # otherwise flatten them along with every other element.
    text = re.sub(r"(<!-- (?:POSITION|FLUIDICS) VARIABLES -->\n)", r"\1\n", text)

    # Two blank lines BEFORE each of those same section comments too, so the
    # loop_variable declarations read as a clearly separated block from the
    # recipe body above them, not just visually attached to it. The comment
    # is indented (toprettyxml), so the newline and the tag are not adjacent.
    text = re.sub(r"\n([ \t]*<!-- (?:POSITION|FLUIDICS) VARIABLES -->)", r"\n\n\n\1", text)

    if leading_comment:
        decl, _, rest = text.partition("\n")
        # XML forbids "--" anywhere inside a comment's content (only the closing
        # "-->" may contain it) -- this codebase's own prose convention uses "--"
        # for asides (as leading_comment's own text does), so every line is
        # sanitized here rather than trusting callers to avoid it. Collapsing
        # any run of 2+ hyphens to one is a purely cosmetic change to the
        # comment text, not a semantic one.
        comment_lines = (re.sub(r"-{2,}", "-", line) for line in leading_comment.splitlines())
        comment_block = "\n".join(f"<!-- {line} -->" for line in comment_lines)
        text = f"{decl}\n{comment_block}\n{rest}"

    text = text.replace("\n", "\r\n")

    with open(output_path, "w", encoding="ISO-8859-1", newline="") as fh:
        fh.write(text)
