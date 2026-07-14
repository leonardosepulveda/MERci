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
Round 1 (imaging): cells acquisition for all FOVs (no preceding fluidics)
Fluidics Round 02: Hybridize/Adaptor 1 → [Readouts] → Buffer   ← NO cleave (first hyb)
Round 2 (imaging): bits #1 acquisition for all FOVs
Fluidics Round 03: Cleave → Hybridize/Adaptor 2 → [Readouts] → Buffer
…
Round N+1 (imaging): bits #N acquisition
[Optional] Fluidics Final: Cleave only

Fluidics loops are named by the NEXT imaging round (e.g. "Fluidics Round 02"
precedes "Imaging Round 02").  The hyb-protocol number tracks the bit/hyb index
(1…N), not the imaging-round number, and the first hyb omits the cleave step
(see ``create_dave_config(first_hyb_no_cleave=...)``).

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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
from xml.dom import minidom

import pandas as pd

from .configs import get_camera_frame_size
from .kilroy import (
    KilroyProtocolResolver,
    load_kilroy_protocols,
    load_protocol_durations,
    protocol_valve_commands,
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
    with open(hal_config_path, "rb") as fh:
        text = fh.read().decode("ISO-8859-1").replace("\r\n", "\n")
    root = ET.fromstring(text)
    el = root.find(".//frames")
    if el is None:
        raise ValueError(f"No <frames> element found in {hal_config_path}")
    return int(el.text.strip())


def count_positions(positions_path: Path) -> int:
    """
    Count the FOV positions in a ``positions_*.txt`` file.

    One FOV per non-blank line (``x,y``); ``#`` comments and blank lines are
    ignored, matching :func:`MERci.common.metadata._read_positions`.  This equals
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
    n = 0
    with Path(positions_path).open() as fh:
        for raw in fh:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if len(line.split(",")) >= 2:
                n += 1
    return n


def fov_pad_width(total_fovs: int) -> int:
    """
    Zero-pad width wide enough to represent every FOV index ``0 … total_fovs-1``
    (e.g. 150 FOVs -> 3 digits, ``"000".."149"``; 1036 FOVs -> 4 digits,
    ``"0000".."1035"``) -- derived from the actual FOV count read from the
    positions file, never a fixed literal. HAL's own file-naming width scales
    the same way with FOV count, so a hardcoded width (e.g. always 3 digits)
    silently stops matching real files the moment an experiment's FOV count
    crosses a digit boundary the hardcoded value didn't anticipate -- exactly
    what happened for a 1036-FOV experiment whose round_info.csv was written
    with a stale/assumed 3-digit width. No artificial floor: an experiment
    with only a handful of FOVs genuinely needs only that many digits.
    """
    return len(str(max(total_fovs - 1, 0)))


# ── Multi-drive round-robin helpers ─────────────────────────────────────────────

def normalize_drive_root(drive: Union[str, Path]) -> Path:
    """
    Normalize a drive-letter/root string into an absolute path anchor.

    A bare drive letter like ``"D:"`` is a *drive-relative* path under
    ``pathlib``/Windows: joining onto it (``Path("D:") / "data"``) depends on
    that drive's process-wide "current directory", an unpredictable legacy
    mechanism, and would NOT resolve to ``D:\\data``. Appending a trailing
    separator turns it into the drive's root (``"D:\\"``) so every join below
    is unambiguously absolute.
    """
    s = str(drive)
    if not s.endswith(("\\", "/")):
        s += "\\"
    root = Path(s)
    if not root.is_absolute():
        raise ValueError(
            f"data_drives entry {drive!r} does not resolve to an absolute path "
            f"(got {root!r}) — use a drive letter (e.g. \"D:\") or an absolute path."
        )
    return root


def _drive_for_bit(bit_idx: int, data_drives: Sequence[Union[str, Path]]) -> Path:
    """Round-robin the bit/hyb index (1-based) across *data_drives*."""
    return normalize_drive_root(data_drives[(bit_idx - 1) % len(data_drives)])


def rebase_on_drive(sample_dir: Union[str, Path], drive_root: Path) -> Path:
    """
    Re-root ``sample_dir`` onto a different drive, preserving its subpath.

    A round-robin hyb round can land on a physical drive other than the one
    ``SAMPLE_DIR`` itself lives on, but its folder layout on that drive should
    mirror ``SAMPLE_DIR``'s own structure rather than dumping straight into
    the drive's root — e.g. ``SAMPLE_DIR = G:\\Leonardo\\LT058_sample_07\\merfish``
    rebased onto ``"D:"`` gives ``D:\\Leonardo\\LT058_sample_07\\merfish``, not
    just ``D:\\``. ``drive_root`` is expected already normalized (see
    :func:`normalize_drive_root`).
    """
    sample_dir = Path(sample_dir)
    return drive_root / sample_dir.relative_to(sample_dir.anchor)


def create_data_drive_skeleton(
    sample_dir:  Path,
    n_bits:      int,
    data_drives: Sequence[Union[str, Path]],
    mode:        str = "legacy",
    boundaries:  Optional[Sequence] = None,
) -> None:
    """
    Pre-create the ``data/[tissue_{t}/]hybs/H##`` folder skeleton, creating
    each hyb's ``H##`` folder ONLY on the drive it is actually round-robin-
    assigned to (via :func:`_drive_for_bit` — the same assignment
    ``create_round_info``/``create_round_info_multitissue`` use for
    ``round_info.csv``), not on every configured drive. This way a drive's
    folder listing alone tells you which hybs it holds, and can never disagree
    with what ``round_info.csv`` says.

    This is a convenience layered on top of ``create_dave_config``'s
    ``create_data_dirs=True`` (default), which already creates only the one
    directory each round strictly needs for HAL's existence check — calling
    this function is optional; skipping it would not break round-robin
    writing, it would just leave each drive's H-folders absent until their
    round comes up.

    Parameters
    ----------
    sample_dir  : experiment root; each round-robin drive's skeleton is
                  rebased onto this path's own subfolder structure (see
                  :func:`rebase_on_drive`), not dumped at the drive's root
    n_bits      : number of bits (hybridisation) rounds
    data_drives : drive letters / absolute roots to spread hyb rounds across
    mode        : ``"multi"``, ``"single"`` or ``"legacy"`` — selects whether
                  the skeleton is nested under ``tissue_{t}/`` per boundary
                  (multi) or flat (single/legacy), matching
                  :func:`create_round_info_multitissue`'s ``_seg_dir`` layout
    boundaries  : ordered ``BoundarySpec`` list (required when ``mode == "multi"``)
    """
    tissues = sorted({b.tissue for b in boundaries}) if (mode == "multi" and boundaries) else [None]
    for bit_idx in range(1, n_bits + 1):
        root = rebase_on_drive(sample_dir, _drive_for_bit(bit_idx, data_drives))
        for tissue in tissues:
            base = (root / "data" / f"tissue_{tissue}") if tissue is not None else (root / "data")
            (base / "hybs" / f"H{bit_idx:02d}").mkdir(parents=True, exist_ok=True)


# ── round_info builder ─────────────────────────────────────────────────────────

def create_round_info(
    microscope:       str,
    n_bits:           int,
    bits_hal_config:  str,
    cells_hal_config: str,
    sample_dir:       Path,
    data_drives:      Optional[Sequence[Union[str, Path]]] = None,
    positions_txt:    Optional[Path] = None,
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
    data_drives       : when given, bits rounds are spread round-robin across
                        these drives (``data_dir`` becomes
                        ``<drive>/data/hybs/H{NN}`` instead of
                        ``sample_dir/data/hybs/H{NN}``); the cells round is
                        unaffected (always ``sample_dir/data/cells``). ``None``
                        (default) preserves today's single-drive layout.
    positions_txt     : the experiment's positions file, used to count real
                        FOVs and derive the ``series`` pattern's zero-pad
                        width (see :func:`fov_pad_width`) -- e.g. 150 FOVs ->
                        ``{fov:03d}``, 1036 FOVs -> ``{fov:04d}``. ``None``
                        (default) falls back to a fixed 3-digit width (the
                        previous, hardcoded behaviour) with a warning, since
                        the true FOV count isn't known without it -- pass
                        this whenever the positions file is available, which
                        it should be by the time ``round_info.csv`` is built.

    Returns
    -------
    pd.DataFrame with columns ``imaging_round``, ``imaging_type``, ``series``,
    ``hal_config``, ``data_dir``
    """
    mic  = microscope.lower()
    data = Path(sample_dir) / "data"
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

    # Imaging Round 1: CELLS ONLY (no fluidics precedes it).
    rows.append({
        "imaging_round": 1,
        "imaging_type":  "cells",
        "series":        f"hal-{mic}-cells_{{fov:0{pad}d}}",
        "hal_config":    cells_hal_config,
        "data_dir":      str(data / "cells"),
    })

    # Imaging Rounds 2 … N+1: bits #1 … #N.  The series number tracks the
    # bit/hyb index (1…N); the imaging_round is bit_idx + 1.  Each bits round
    # writes into its own subfolder ``data/hybs/H{NN}`` (NN = bit/hyb index), so
    # the rounds are spread across folders instead of piling into one ``data/``.
    # When ``data_drives`` is given, that subfolder is additionally rooted at a
    # round-robin-assigned drive instead of always ``sample_dir`` (so successive
    # hyb rounds land on different physical disks) -- rebased onto
    # ``sample_dir``'s own subpath (see ``rebase_on_drive``) so e.g.
    # ``G:\Leonardo\LT058_sample_07\merfish`` round-robins to
    # ``D:\Leonardo\LT058_sample_07\merfish``, not just ``D:\``.
    # ``create_dave_config`` emits a matching ``<change_directory>`` before
    # each round's imaging loop.
    for bit_idx in range(1, n_bits + 1):
        bits_root = (
            rebase_on_drive(sample_dir, _drive_for_bit(bit_idx, data_drives))
            if data_drives else data.parent
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
    data_drives:        Optional[Sequence[Union[str, Path]]] = None,
    tissue_path_mode:   Callable[[int], str] = lambda tissue: "legacy",
) -> pd.DataFrame:
    """
    Build a **segment-aware** ``round_info`` for a multi-boundary experiment.

    Each imaging round visits the acquisition-order segments built by
    :func:`MERci.acquisition.positions.group_boundaries_by_path_mode` (the
    same function ``notebooks/prepare_imaging/02`` uses to decide what it
    actually writes to ``positions/``): a tissue's own consecutive boundaries
    are merged into ONE segment when ``tissue_path_mode(tissue) == "legacy"``
    (no transit within that tissue), otherwise each boundary keeps its own
    segment; a transit segment always bridges consecutive top-level segments
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
    data_drives        : when given, bits rounds are spread round-robin across
                         these drives (each bits ``data_dir`` is rooted at the
                         assigned drive instead of ``sample_dir``); cells and
                         transit segments are unaffected. ``None`` (default)
                         preserves today's single-drive layout.
    tissue_path_mode   : tissue index -> ``"legacy"`` or ``"transit"`` (see
                         :func:`MERci.acquisition.positions.group_boundaries_by_path_mode`).
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
        # bits: round-robin the drive (if configured) BEFORE appending the
        # tissue/hybs/H{NN} subtree, so rounds are spread across both folders
        # and physical disks; cells/transit always stay rooted at sample_dir.
        # The round-robin drive is rebased onto sample_dir's own subpath (see
        # rebase_on_drive) rather than dumped at the drive's root.
        if kind != "transit" and not is_cells and data_drives:
            root = rebase_on_drive(sample_dir, _drive_for_bit(hyb_idx, data_drives)) / "data"
        else:
            root = data
        base = root / f"tissue_{tissue}" if mode == "multi" else root
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
            tissue, label, posfile = seg["tissue"], seg["label"], seg["posfile"]
            start, pad = seg["start"], seg["pad"]
            if seg["kind"] == "boundary":
                # Shared movie name (no per-segment label): the continuous index
                # comes from start/pad, not from the name.
                rows.append({
                    "imaging_round":  rnd,
                    "imaging_type":   "cells" if is_cells else "bits",
                    "series":         f"{movie_prefix}_{{fov:0{pad}d}}",
                    "hal_config":     hal_boundary,
                    "data_dir":       _seg_dir(tissue, "boundary", is_cells, hyb_idx),
                    "positions_file": posfile,
                    "tissue":         tissue,
                    "segment":        label,
                    "fov_start":      start,
                    "fov_pad":        pad,
                })
            else:
                rows.append({
                    "imaging_round":  rnd,
                    "imaging_type":   "transit",
                    "series":         f"hal-{mic}-transit_r{rnd:02d}_{{fov:0{pad}d}}",
                    "hal_config":     transit_hal_config,
                    "data_dir":       _seg_dir(tissue, "transit", is_cells),
                    "positions_file": posfile,
                    "tissue":         tissue,
                    "segment":        label,
                    "fov_start":      start,
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

def create_dave_config(
    round_info:           pd.DataFrame,
    positions_file:       Path,
    settings_dir:         Path,
    output_path:          Path,
    use_adaptors:         bool = False,
    include_final_cleave: bool = False,
    first_hyb_no_cleave:  bool = True,
    num_focus_checks:     int  = 50,
    fluidics_protocols:   Optional[Sequence[str]] = None,
    kilroy_config:        Optional[Path] = None,
    positions_dir:        Optional[Path] = None,
    create_data_dirs:     bool = True,
    print_estimate:       bool = True,
    microscope:           Optional[str] = None,
    estimate_frame_shape: Optional[Sequence[int]] = None,
    estimate_bytes_per_pixel: int = 2,
) -> Optional["ExperimentEstimate"]:
    """
    Write an explicit-block Dave recipe XML from ``round_info``.

    **Positions model.** Two layouts are supported:

    * *single-positions* (default) — every movie in a round iterates the one
      ``positions_file``; each imaging round is a single ``<loop>``.
    * *per-segment* — used when ``round_info`` has a ``positions_file`` column and
      ``positions_dir`` is given (the multi-boundary layout from
      ``create_round_info_multitissue``). Because a Dave loop iterates exactly one
      positions file, each segment (boundary or transit) becomes its **own**
      ``<loop>`` — named ``"Imaging Round NN - <segment>"`` — with its own movie
      and HAL config, in ``round_info`` row order. Fluidics loops still sit
      between rounds (after a round's last segment loop).

    **One ``<loop_variable>`` per loop, always — even when several rounds
    point at the same positions file.** Dave's real ``v2Generator``
    (``handleLoop``) resolves a ``<loop>`` by looking up
    ``self.loop_variable_names.index(loop.attrib["name"])`` — every loop MUST
    have a ``<loop_variable>`` of the exact same name; there is no mechanism
    for a ``<movie>``'s ``<variable_entry>`` to reference a *different*,
    shared loop_variable declared under another name (verified directly
    against the storm_control source; an earlier attempt to declare one
    shared loop_variable per segment/positions-file, referenced by name from
    each round's movie, loaded fine in MERci's own reader but made real Dave
    raise ``ValueError: 'Imaging Round NN' is not in list``). So every round
    (and, in per-segment mode, every round×segment) still gets its own
    identically-named ``<loop_variable>``, duplicating the same ``file_path``
    across as many declarations as there are rounds/segments that use it.

      When the rows carry ``fov_start``/``fov_pad`` (produced by
      ``create_round_info_multitissue``), each movie ``<name>`` is emitted with
      ``start``/``pad`` attributes so all boundary movies share one name with a
      single running FOV index and all transit movies likewise — see that function.
      This makes the recipe depend on the patched Dave ``v2Generator``
      (``dave_fov_offset_patch``); stock Dave ignores the attributes and the shared
      names would collide.

    Fluidics loops are named by the NEXT imaging round (e.g. "Fluidics Round 02"
    precedes "Imaging Round 02").  The hyb-protocol number tracks the bit/hyb
    index (the count of bits rounds reached so far), not the imaging-round
    number, so a leading cells round does not shift the Kilroy protocol names.
    The last imaging round has no trailing fluidics unless
    ``include_final_cleave=True``.

    **Save location per round.** When ``round_info`` has a ``data_dir`` column, a
    ``<change_directory>`` element sets HAL's save directory (from ``data_dir``) to
    the upcoming imaging round's folder. It is emitted **before that round's fluidics
    loop** — so a ``<change_directory>`` precedes every fluidics block — and before
    the first imaging round (which has no preceding fluidics). Emission is
    de-duplicated: the directory is not re-set before the round's imaging loop when
    the fluidics already set it. In the multi-boundary layout a round spans several
    directories, so the extra per-segment directories are still set before their own
    segment loops. This spreads rounds across folders (e.g. ``data/hybs/H01``,
    ``H02``, …). HAL **requires the directory to exist** (it errors otherwise), and
    neither Dave nor HAL creates it, so with ``create_data_dirs=True`` (default) this
    function creates every referenced directory. (``change_directory`` maps to HAL's
    "Set Directory" message, which is deprecated but still functional — it only emits
    a warning.)

    Parameters
    ----------
    round_info            : DataFrame with columns ``imaging_round``,
                            ``series``, ``hal_config``
    positions_file        : path to ``positions_*.txt``; written into each
                            ``<loop_variable>/<file_path>``
    settings_dir          : directory containing the HAL config XML files
                            (used to read ``<frames>`` counts)
    output_path           : where to write the recipe XML
    use_adaptors          : if True, generate adaptor-based fluidics
                            (``Cleave adaptors`` / ``Hyb adaptors N`` /
                            ``Hyb readouts`` / ``Flow Image Buffer``);
                            if False, use direct readout protocols
                            (``Cleave direct`` / ``Hybridize N`` /
                            ``Wash and Imaging Buffers``)
    include_final_cleave  : if True, append a "Fluidics Final" block after the
                            last imaging round containing only a single cleave
                            step (``Cleave adaptors`` or ``Cleave direct``)
    first_hyb_no_cleave   : if True (default), the fluidics block that precedes
                            the FIRST bits imaging round omits the cleave step
                            (used when a cells round is imaged first, so the
                            first hybridisation flows onto a freshly prepared
                            sample); all later fluidics blocks keep the cleave.
                            Ignored when ``fluidics_protocols`` is given.
    num_focus_checks      : value for ``<num_focus_checks>``
    fluidics_protocols    : if provided, use this fixed list of Kilroy protocol
                            names for every between-round fluidics block,
                            overriding ``use_adaptors``
    kilroy_config         : path to the Kilroy config XML that will run this
                            experiment.  When given, every fluidic protocol
                            written into the recipe is resolved to (and required
                            to exist as) a real ``<protocol>`` in that Kilroy
                            config — the cleave / hybridize / readouts / image-
                            buffer step names are taken from the Kilroy file
                            rather than hard-coded, and a ``ValueError`` is
                            raised if any required step has no matching protocol.
                            When ``None`` (legacy), hard-coded protocol names are
                            used and no Kilroy cross-check is performed.
    create_data_dirs      : if True (default), create every directory named in the
                            ``data_dir`` column (the targets of the emitted
                            ``<change_directory>`` elements) so HAL's existence
                            check passes.  Set False when generating the recipe on a
                            machine other than the acquisition computer.
    print_estimate        : if True (default), print an estimated run time and raw
                            storage for the recipe (see
                            :func:`estimate_dave_experiment`).  Requires
                            ``kilroy_config`` for the fluidics portion.
    microscope            : microscope id (e.g. ``"MF3"``, ``"MFX"``, ``"ST2"``)
                            used to pick the camera frame size for the storage
                            estimate (MFX/ST2 → 2304², MF-series → 2048²; see
                            ``configs.get_camera_frame_size``).  When ``None`` it is
                            inferred from the ``series`` names in ``round_info``.
    estimate_frame_shape  : explicit ``(width, height)`` in pixels for the storage
                            estimate; overrides the microscope-derived size.  When
                            ``None`` (default) the size comes from ``microscope``.
    estimate_bytes_per_pixel : bytes per pixel for the storage estimate (2 = uint16)

    Returns
    -------
    ExperimentEstimate or None
        The estimate when ``print_estimate`` is True, else None.
    """
    round_ids = sorted(round_info["imaging_round"].unique())
    n_rounds  = len(round_ids)
    has_data_dir = "data_dir" in round_info.columns

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

    def _round_has_bits(rid: int) -> bool:
        """True if round *rid* images bits (not just cells / transit)."""
        rrows = round_info[round_info["imaging_round"] == rid]
        if "imaging_type" in round_info.columns:
            types = {str(t).strip().lower() for t in rrows["imaging_type"].dropna()}
            if types:
                return "bits" in types
        return any("cells" not in str(s) for s in rrows["series"])

    bits_round_ids   = [rid for rid in round_ids if _round_has_bits(rid)]
    first_bits_round = bits_round_ids[0] if bits_round_ids else None

    root = ET.Element("recipe")
    seq  = ET.SubElement(root, "command_sequence")

    imaging_loop_vars:  list[tuple[str, str]]       = []
    fluidics_loop_vars: list[tuple[str, list[str]]] = []
    created_dirs:       set[str]                    = set()
    current_dir:        Optional[str]               = None   # last <change_directory> emitted

    def _add_change_directory(dir_value) -> None:
        """
        Emit a ``<change_directory>`` (sets HAL's save dir for the FOLLOWING loop)
        from *dir_value*, and — when ``create_data_dirs`` — create that folder.

        HAL rejects a directory that does not exist, and nothing in Dave/HAL makes
        it, so the directory is created here. De-duplicated: emitting the directory
        that is already active is a no-op, so setting the round's directory before
        its fluidics does not repeat it before the round's imaging loop. No-op when
        the round_info has no ``data_dir`` column or the value is blank/NaN.
        """
        nonlocal current_dir
        if not has_data_dir or not pd.notna(dir_value):
            return
        dpath = str(dir_value).strip()
        if not dpath or dpath == current_dir:
            return
        ET.SubElement(seq, "change_directory").text = dpath
        current_dir = dpath
        if create_data_dirs and dpath not in created_dirs:
            created_dirs.add(dpath)
            try:
                Path(dpath).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                # Non-fatal: recipe still written. Warn so the user creates the
                # folder on the acquisition machine (HAL requires it to exist).
                print(f"[create_dave_config] WARNING: could not create data dir "
                      f"{dpath!r}: {exc}. Create it on the acquisition computer "
                      f"before running Dave.")

    def _add_movie(parent_loop: ET.Element, row: pd.Series, variable_name: str) -> None:
        """Append one <movie> (resolving its HAL frame count) to *parent_loop*."""
        movie_name   = series_to_movie_name(str(row["series"]))
        hal_stem     = Path(str(row["hal_config"])).stem
        hal_path     = settings_dir / (hal_stem + ".xml")
        try:
            n_frames = get_hal_frame_count(hal_path)
        except (FileNotFoundError, ValueError):
            n_frames = 0

        movie   = ET.SubElement(parent_loop, "movie")
        name_el = ET.SubElement(movie, "name")
        name_el.set("increment", "Yes")
        name_el.text = movie_name
        # Continuous FOV numbering across per-segment loops (multi-boundary layout):
        # when round_info carries fov_start/fov_pad, emit them as the patched Dave's
        # <name start=… pad=…> so boundary (and transit) movies share one name yet
        # keep a single running, non-colliding index. Absent columns → stock Dave
        # numbering (single-positions layout is unaffected).
        if "fov_start" in row.index and pd.notna(row.get("fov_start")):
            name_el.set("start", str(int(row["fov_start"])))
        if "fov_pad" in row.index and pd.notna(row.get("fov_pad")):
            name_el.set("pad", str(int(row["fov_pad"])))
        ET.SubElement(movie, "length").text     = str(n_frames)
        ET.SubElement(movie, "parameters").text = hal_stem
        cf = ET.SubElement(movie, "check_focus")
        ET.SubElement(cf, "num_focus_checks").text = str(num_focus_checks)
        ET.SubElement(cf, "focus_scan")
        ET.SubElement(movie, "overwrite").text = "False"
        ve = ET.SubElement(movie, "variable_entry")
        ve.set("name", variable_name)

    def _add_fluidics(round_id: int, is_last: bool) -> None:
        """Append the between-round fluidics loop that FOLLOWS *round_id*."""
        precede_dir = None   # save dir for the imaging round this fluidics precedes
        if not is_last:
            next_round = round_id + 1
            fl_name    = f"Fluidics Round {next_round:02d}"

            # The directory for the upcoming imaging round (from round_info.data_dir)
            # is set BEFORE its fluidics block, so a <change_directory> precedes every
            # fluidics loop. In the multi-segment layout the round spans several
            # directories; the first segment's is set here and the remaining segments
            # set theirs before their own loops.
            if has_data_dir:
                next_rows = round_info[round_info["imaging_round"] == next_round]
                if len(next_rows):
                    precede_dir = next_rows.iloc[0].get("data_dir")

            # Hyb number tracks the bit/hyb index of the NEXT imaging round, so a
            # leading cells round does not shift the Kilroy protocol numbers.
            if first_bits_round is not None and next_round >= first_bits_round:
                hyb_idx = next_round - first_bits_round + 1
            else:
                hyb_idx = next_round   # no cells offset detected; legacy numbering

            # The fluidics that precedes the FIRST bits round omits the cleave.
            is_first_hyb = (first_bits_round is not None and next_round == first_bits_round)
            skip_cleave  = is_first_hyb and first_hyb_no_cleave

            if fluidics_protocols is not None:
                fl_protocols = list(fluidics_protocols)
                if resolver is not None:
                    resolver.validate(fl_protocols)
            elif resolver is not None:
                # Names taken from the Kilroy config (see kilroy_config).
                cleave = [] if skip_cleave else [resolver.cleave(adaptors=use_adaptors)]
                if use_adaptors:
                    steps = [resolver.hybridize(hyb_idx, adaptors=True), resolver.readouts()]
                else:
                    steps = [resolver.hybridize(hyb_idx, adaptors=False)]
                # Some Kilroy protocols (e.g. "Hybridize N") already end by
                # setting/flowing the imaging buffer themselves -- appending the
                # standalone image-buffer protocol on top would flow it twice.
                # Detected by comparing the step immediately preceding it
                # (readouts, or the hybridize step itself) against the
                # standalone protocol's own first valve command, rather than
                # hard-coding which Dave step this can happen after.
                image_buffer = resolver.image_buffer()
                preceding_valves = protocol_valve_commands(kilroy_config, steps[-1])
                buffer_valves    = protocol_valve_commands(kilroy_config, image_buffer)
                already_flowed = bool(preceding_valves and buffer_valves
                                      and preceding_valves[-1].lower() == buffer_valves[0].lower())
                if not already_flowed:
                    steps.append(image_buffer)
                fl_protocols = cleave + steps
            elif use_adaptors:
                # Legacy hard-coded names (no Kilroy cross-check).
                fl_protocols = ([] if skip_cleave else ["Cleave adaptors"]) + [
                    f"Hyb adaptors {hyb_idx}",
                    "Hyb readouts",
                    "Flow Image Buffer",
                ]
            else:
                fl_protocols = ([] if skip_cleave else ["Cleave direct"]) + [
                    f"Hybridize {hyb_idx}",
                    "Wash and Imaging Buffers",
                ]
        elif include_final_cleave:
            fl_name = "Fluidics Final"
            if resolver is not None:
                fl_protocols = [resolver.cleave(adaptors=use_adaptors)]
            else:
                fl_protocols = ["Cleave adaptors" if use_adaptors else "Cleave direct"]
        else:
            return

        _add_change_directory(precede_dir)   # <change_directory> before this fluidics
        fl_loop = ET.SubElement(seq, "loop")
        fl_loop.set("name", fl_name)
        ve = ET.SubElement(fl_loop, "variable_entry")
        ve.set("name", fl_name)
        fluidics_loop_vars.append((fl_name, fl_protocols))

    for idx, round_id in enumerate(round_ids):
        is_last = (idx == n_rounds - 1)
        rows    = round_info[round_info["imaging_round"] == round_id]

        if segment_mode:
            # One loop per (round, segment) -- a Dave loop iterates a single
            # positions file. Each loop gets its OWN loop_variable, named
            # identically to the loop itself: Dave's real v2Generator
            # (handleLoop) looks up `self.loop_variable_names.index(loop.attrib["name"])`
            # -- every <loop> MUST have a <loop_variable> of the exact same
            # name, full stop. A <movie>'s <variable_entry> cannot "alias" a
            # differently-named loop_variable declared elsewhere: it does its
            # own independent index() lookup and reads the CURRENT iterator
            # state for THAT loop_variable, which is only ever advanced by the
            # one <loop> whose name matches it. So when several rounds visit
            # the same segment, each round still declares its own
            # loop_variable pointing at the same positions file -- Dave has no
            # mechanism to share one across differently-named loops (verified
            # directly against storm_control's real v2Generator.py source; a
            # shared name here previously caused ValueError: '<loop name>' is
            # not in list). Each segment sets its own save directory just
            # before its loop.
            for _, row in rows.iterrows():
                seg   = str(row.get("segment", "")).strip() or series_to_movie_name(str(row["series"]))
                lname = f"Imaging Round {round_id:02d} - {seg}"
                _add_change_directory(row.get("data_dir"))
                loop  = ET.SubElement(seq, "loop")
                loop.set("name", lname)
                _add_movie(loop, row, lname)
                imaging_loop_vars.append((lname, str(positions_dir / str(row["positions_file"]))))
        else:
            # Single loop for the round; all movies share positions_file and one
            # save directory (from the round's first row's data_dir). This loop
            # gets its own loop_variable (named identically to it) even though
            # every round points at the same positions file -- see the
            # segment_mode branch above for why Dave requires this per-loop
            # declaration rather than one shared across rounds.
            img_name = f"Imaging Round {round_id:02d}"
            _add_change_directory(rows.iloc[0].get("data_dir") if has_data_dir else None)
            img_loop = ET.SubElement(seq, "loop")
            img_loop.set("name", img_name)
            for _, row in rows.iterrows():
                _add_movie(img_loop, row, img_name)
            imaging_loop_vars.append((img_name, str(positions_file)))

        _add_fluidics(round_id, is_last)

    # ── Loop variables ─────────────────────────────────────────────────────────
    for lname, pos_path in imaging_loop_vars:
        lv = ET.SubElement(root, "loop_variable")
        lv.set("name", lname)
        ET.SubElement(lv, "file_path").text = pos_path

    for lname, protocols in fluidics_loop_vars:
        lv = ET.SubElement(root, "loop_variable")
        lv.set("name", lname)
        val = ET.SubElement(lv, "value")
        for protocol in protocols:
            ET.SubElement(val, "valve_protocol").text = protocol

    _write_dave_xml(root, Path(output_path))

    if print_estimate:
        # Frame size: explicit override wins; otherwise from the microscope (given
        # or inferred from the round_info series names).
        if estimate_frame_shape is not None:
            frame_w, frame_h = int(estimate_frame_shape[0]), int(estimate_frame_shape[1])
        else:
            frame_w, frame_h = get_camera_frame_size(microscope or _infer_microscope(round_info))
        est = estimate_dave_experiment(
            Path(output_path),
            kilroy_config   = kilroy_config,
            settings_dir    = settings_dir,
            frame_width     = frame_w,
            frame_height    = frame_h,
            bytes_per_pixel = estimate_bytes_per_pixel,
        )
        print(format_experiment_estimate(est, per_round=True))
        return est
    return None


# ── Dave annotation ────────────────────────────────────────────────────────────

def annotate_dave_with_round_info(
    dave_path:       Path,
    round_bit_color: list[tuple],
) -> None:
    """
    Insert XML comments into an existing Dave recipe XML describing which bits
    are imaged in each round.

    For round 1: comment is placed before the ``<loop name="Imaging Round 01">``
    block.  For rounds 2+: comment is placed before the corresponding
    ``<loop name="Fluidics Round NN">`` block (which precedes that imaging
    round).  A blank line is inserted before each comment for readability.

    In the default cells-first layout, imaging round 1 is the cells acquisition
    (no bits), so it normally has no entry here; the bits comments attach to the
    ``Fluidics Round NN`` loops for rounds 2…N+1.  The ``round_1indexed`` values
    passed in must therefore be **imaging-round** indices (bits start at 2), not
    bit/hyb indices — see ``notebooks/prepare_imaging/04``.

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
            new_lines.append("")
            new_lines.append(f"{indent}<!-- Round {n}:")
            for s in round_comments[n]:
                new_lines.append(f"{indent}        {s}")
            new_lines.append(f"{indent}-->")

        # Round 1: insert before "Imaging Round 01" loop
        if stripped == '<loop name="Imaging Round 01">':
            _append_comment(1)
        else:
            # Rounds 2+: insert before the corresponding "Fluidics Round NN" loop
            m = re.match(r'^<loop name="Fluidics Round (\d+)">', stripped)
            if m:
                _append_comment(int(m.group(1)))

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
    per_round       : list of ``{"round", "label", "imaging_s", "fluidics_s",
                      "bytes", "movies", "series"}`` dicts, numbered rounds in
                      order followed by any non-numbered row (e.g. the closing
                      "Fluidics Final" step, labeled by its own loop name
                      rather than a fabricated round number). ``series`` is
                      the list of distinct movie names (e.g. ``"hal-st2_01"``)
                      imaged in that round.
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
    per_round:       List[dict] = field(default_factory=list)
    assumptions:     List[str]  = field(default_factory=list)
    warnings:        List[str]  = field(default_factory=list)


def _read_hal_exposure(hal_config_path: Path) -> Optional[float]:
    """Return the camera ``<exposure_time>`` (seconds) from a HAL config, or None."""
    try:
        with open(hal_config_path, "rb") as fh:
            text = fh.read().decode("ISO-8859-1").replace("\r\n", "\n")
        root = ET.fromstring(text)
        el = root.find(".//exposure_time")
        if el is not None and el.text:
            return float(el.text.strip())
    except (OSError, ValueError, ET.ParseError):
        pass
    return None


def estimate_dave_experiment(
    dave_recipe:          Path,
    kilroy_config:        Optional[Path] = None,
    settings_dir:         Optional[Path] = None,
    frame_width:          int   = 2048,
    frame_height:         int   = 2048,
    bytes_per_pixel:      int   = 2,
    frame_time_s:         Optional[float] = None,
    readout_overhead_s:   float = 0.0,
    per_movie_overhead_s: float = 0.0,
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
    ``frame_width``/``frame_height``/``bytes_per_pixel`` are parameters (default a
    2048×2048 uint16 sCMOS frame = 8 MiB/frame); adjust them for other cameras.

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
                           Dave's own estimate omits these, so the default is 0

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
            exp = _read_hal_exposure(Path(settings_dir) / (hal_stem + ".xml")) \
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

    def _round_no(lname: str) -> Optional[int]:
        m = re.search(r"Round (\d+)", lname)
        return int(m.group(1)) if m else None

    for loop in seq.findall("loop"):
        lname  = loop.get("name", "")
        rno    = _round_no(lname)
        # Loops without a numbered round in their name (e.g. the closing
        # "Fluidics Final" cleave-only step) get their own row keyed and
        # labeled by their real name, instead of being silently folded into a
        # fabricated "Round 00" -- which not only mislabeled the step but
        # sorted it to the front of the report even though it runs LAST.
        key    = rno if rno is not None else lname
        rec    = per_round.setdefault(
            key, {"round": rno, "label": (f"Round {rno:02d}" if rno is not None else lname),
                  "imaging_s": 0.0, "fluidics_s": 0.0, "bytes": 0, "movies": 0, "series": []})
        movies = loop.findall("movie")
        if movies:                                   # imaging loop
            # The loop_variable a movie references is its OWN <variable_entry
            # name="...">, not necessarily the parent <loop>'s own name: since
            # positions loop_variables are shared across every round visiting
            # the same segment (see create_dave_config), a loop named e.g.
            # "Imaging Round 02" can reference a loop_variable named "B1" or
            # "Positions". Every movie within one loop references the same
            # variable, so the first movie's is enough.
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
        # Numbered rounds sorted numerically first; any non-numbered row (e.g.
        # "Fluidics Final") sorted after, in the order it was first seen --
        # `sorted` is stable, so ties among non-int keys keep dict insertion order.
        per_round       = [per_round[k] for k in
                           sorted(per_round, key=lambda k: (0, k) if isinstance(k, int) else (1, 0))],
        assumptions     = assumptions,
        warnings        = warnings_list,
    )


def _fmt_duration(seconds: float) -> str:
    """Format seconds as ``Dd HHh MMm SSs`` (dropping leading zero units)."""
    total = int(round(seconds))
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s   = divmod(rem, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m {s:02d}s"
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_bytes(n: int) -> str:
    """Format a byte count in binary units (KiB/MiB/GiB/TiB)."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TiB"


def format_experiment_estimate(est: ExperimentEstimate, per_round: bool = False) -> str:
    """Render an :class:`ExperimentEstimate` as a readable multi-line report."""
    lines = [
        "Estimated experiment cost",
        f"  FOV-movies:    {est.n_fov_movies}",
        f"  Imaging time:  {_fmt_duration(est.imaging_time_s)}",
        f"  Fluidics time: {_fmt_duration(est.fluidics_time_s)}",
        f"  Total time:    {_fmt_duration(est.total_time_s)}",
        f"  Storage:       {_fmt_bytes(est.total_bytes)}",
    ]
    if per_round and est.per_round:
        lines.append("  Per round:")
        for r in est.per_round:
            series_str = ", ".join(r.get("series", [])) or "-"
            lines.append(
                f"    {r['label']} [{series_str}]: {r['movies']} movies, "
                f"img {_fmt_duration(r['imaging_s'])}, "
                f"flu {_fmt_duration(r['fluidics_s'])}, "
                f"{_fmt_bytes(r['bytes'])}")
    if est.assumptions:
        lines.append("  Assumptions: " + "; ".join(est.assumptions))
    if est.warnings:
        lines.append("  Warnings:")
        for w in dict.fromkeys(est.warnings):     # de-duplicate, keep order
            lines.append(f"    - {w}")
    return "\n".join(lines)


# ── XML writer ─────────────────────────────────────────────────────────────────

def _write_dave_xml(root: ET.Element, output_path: Path) -> None:
    """Serialize the recipe with indentation and CRLF line endings."""
    raw  = ET.tostring(root, encoding="utf-8")
    dom  = minidom.parseString(raw)
    text = dom.toprettyxml(indent="    ", encoding="ISO-8859-1").decode("ISO-8859-1")

    # Remove the extra blank line toprettyxml adds before every element
    text = re.sub(r"\n[ \t]*\n", "\n", text)
    text = text.replace("\n", "\r\n")

    with open(output_path, "w", encoding="ISO-8859-1", newline="") as fh:
        fh.write(text)
