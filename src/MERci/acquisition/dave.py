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
  dave ``<name>`` element (e.g. ``hal-mf3-epi_01_{fov:03d}`` → ``hal-mf3-epi_01``)
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence
from xml.dom import minidom

import pandas as pd

from .kilroy import KilroyProtocolResolver, load_kilroy_protocols


# ── Public helpers ─────────────────────────────────────────────────────────────

def series_to_movie_name(series: str) -> str:
    """
    Strip the ``_{fov:…}`` format-string suffix from a series pattern to get
    the Dave movie base name.

    Examples
    --------
    ``hal-mf3-epi_01_{fov:03d}``    → ``hal-mf3-epi_01``
    ``hal-mf3-epi-cells_{fov:03d}`` → ``hal-mf3-epi-cells``
    """
    return re.sub(r"_\{[^}]+\}$", "", series)


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


# ── round_info builder ─────────────────────────────────────────────────────────

def create_round_info(
    microscope:       str,
    n_bits:           int,
    bits_hal_config:  str,
    cells_hal_config: str,
    sample_dir:       Path,
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

    Returns
    -------
    pd.DataFrame with columns ``imaging_round``, ``imaging_type``, ``series``,
    ``hal_config``, ``data_dir``
    """
    mic      = microscope.lower()
    data     = Path(sample_dir) / "data"
    rows: List[dict] = []

    # Imaging Round 1: CELLS ONLY (no fluidics precedes it).
    rows.append({
        "imaging_round": 1,
        "imaging_type":  "cells",
        "series":        f"hal-{mic}-epi-cells_{{fov:03d}}",
        "hal_config":    cells_hal_config,
        "data_dir":      str(data / "cells"),
    })

    # Imaging Rounds 2 … N+1: bits #1 … #N.  The series number tracks the
    # bit/hyb index (1…N); the imaging_round is bit_idx + 1.  Each bits round
    # writes into its own subfolder ``data/hybs/H{NN}`` (NN = bit/hyb index), so
    # the rounds are spread across folders instead of piling into one ``data/``.
    # ``create_dave_config`` emits a matching ``<change_directory>`` before each
    # round's imaging loop.
    for bit_idx in range(1, n_bits + 1):
        rows.append({
            "imaging_round": bit_idx + 1,
            "imaging_type":  "bits",
            "series":        f"hal-{mic}-epi_{bit_idx:02d}_{{fov:03d}}",
            "hal_config":    bits_hal_config,
            "data_dir":      str(data / "hybs" / f"H{bit_idx:02d}"),
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
) -> pd.DataFrame:
    """
    Build a **segment-aware** ``round_info`` for a multi-boundary experiment.

    Each imaging round visits the boundaries in order with a transit segment
    between consecutive boundaries (wrapping the last back to the first), exactly
    as laid out by ``notebooks/prepare_imaging/02``. This produces **one row per
    (round, segment)** — so each round has several movies: a boundary movie
    (cells/bits HAL config) per boundary and a transit movie (transit HAL config,
    blank frames) per transit.

    Round 1 is the cells acquisition; rounds 2…N+1 are bits #1…#N. Every row also
    carries the ``positions_file`` (basename in ``positions/``) and the per-tissue
    ``data_dir`` subfolder the segment writes to.

    **Consolidated movie names + continuous FOV index.** Within a round, all
    boundary movies share ONE movie name (e.g. ``hal-mf3-epi-cells`` /
    ``hal-mf3-epi_01``) and all transit movies share one name
    (``hal-mf3-epi-transit_rNN``) — the per-segment label is dropped from the movie
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

    Returns
    -------
    pd.DataFrame with columns ``imaging_round``, ``imaging_type`` (``cells`` /
    ``bits`` / ``transit``), ``series``, ``hal_config``, ``data_dir``,
    ``positions_file``, ``tissue``, ``segment``, ``fov_start``, ``fov_pad``.
    """
    mic          = microscope.lower()
    n_boundaries = len(boundaries)
    # As many transits as boundaries (each wraps to the next), unless there is
    # only one boundary — then there is nothing to transit between.
    n_transits   = n_boundaries if n_boundaries > 1 else 0
    data         = Path(sample_dir) / "data"
    pos_dir      = Path(sample_dir) / "positions"

    def _seg_dir(tissue: int, kind: str, is_cells: bool, hyb_idx: Optional[int] = None) -> str:
        base = data / f"tissue_{tissue}" if mode == "multi" else data
        if kind == "transit":
            return str(base / "transit")
        if is_cells:
            return str(base / "cells")
        # bits: separate each hyb round into its own subfolder ``hybs/H{NN}``
        # (NN = bit/hyb index) so rounds are spread across folders.
        return str(base / "hybs" / f"H{hyb_idx:02d}")

    # Ordered segment templates for one round's traversal (round-independent):
    # (kind, tissue, label, positions_file).
    seg_templates: List[tuple] = []
    for k, spec in enumerate(boundaries):
        seg_templates.append(
            ("boundary", spec.tissue, spec.label,
             f"positions_{sample_name}_{spec.label}.txt")
        )
        if n_transits:
            seg_templates.append(
                ("transit", spec.tissue, f"transit_{k + 1}",
                 f"positions_{sample_name}_transit_{k + 1}.txt")
            )

    # ── Continuous FOV numbering across segments ────────────────────────────────
    # We want every boundary movie in a round to share ONE movie name (e.g.
    # ``hal-mf3-epi-cells``) with a single running FOV index 0…(ΣtBoundaryFOVs−1),
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

    def _group_pad(total: int) -> int:
        # Digits needed for the largest index (total-1); floor at 3 so the index
        # keeps the conventional 3-wide form the analysis side expects.
        return max(3, len(str(total - 1))) if total > 0 else 3

    boundary_total = sum(c for (t, c) in zip(seg_templates, counts) if t[0] == "boundary")
    transit_total  = sum(c for (t, c) in zip(seg_templates, counts) if t[0] == "transit")
    boundary_pad   = _group_pad(boundary_total)
    transit_pad    = _group_pad(transit_total)

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
                    "series":         f"hal-{mic}-epi-transit_r{rnd:02d}_{{fov:0{pad}d}}",
                    "hal_config":     transit_hal_config,
                    "data_dir":       _seg_dir(tissue, "transit", is_cells),
                    "positions_file": posfile,
                    "tissue":         tissue,
                    "segment":        label,
                    "fov_start":      start,
                    "fov_pad":        pad,
                })

    # Round 1: cells.
    _emit(1, is_cells=True, movie_prefix=f"hal-{mic}-epi-cells", hal_boundary=cells_hal_config)
    # Rounds 2…N+1: bits #1…#N (movie series number tracks the bit/hyb index).
    for bit_idx in range(1, n_bits + 1):
        _emit(bit_idx + 1, is_cells=False,
              movie_prefix=f"hal-{mic}-epi_{bit_idx:02d}", hal_boundary=bits_hal_config,
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
) -> None:
    """
    Write an explicit-block Dave recipe XML from ``round_info``.

    **Positions model.** Two layouts are supported:

    * *single-positions* (default) — every movie in a round iterates the one
      ``positions_file``; each imaging round is a single ``<loop>``.
    * *per-segment* — used when ``round_info`` has a ``positions_file`` column and
      ``positions_dir`` is given (the multi-boundary layout from
      ``create_round_info_multitissue``). Because a Dave loop iterates exactly one
      positions file, each segment (boundary or transit) becomes its **own**
      ``<loop>`` — named ``"Imaging Round NN - <segment>"`` — with its own movie,
      HAL config and positions file, in ``round_info`` row order. Fluidics loops
      still sit between rounds (after a round's last segment loop).

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

    **Save location per loop.** When ``round_info`` has a ``data_dir`` column, a
    ``<change_directory>`` element is emitted immediately before each imaging loop,
    setting HAL's save directory to that loop's ``data_dir`` (per-segment in the
    multi-boundary layout, per-round otherwise). This spreads rounds across folders
    (e.g. ``data/hybs/H01``, ``H02``, …). HAL **requires the directory to exist**
    (it errors otherwise), and neither Dave nor HAL creates it, so with
    ``create_data_dirs=True`` (default) this function creates every referenced
    directory. (``change_directory`` maps to HAL's "Set Directory" message, which is
    deprecated but still functional — it only emits a warning.)

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

    def _add_change_directory(dir_value) -> None:
        """
        Emit a ``<change_directory>`` (sets HAL's save dir for the FOLLOWING loop)
        from *dir_value*, and — when ``create_data_dirs`` — create that folder.

        HAL rejects a directory that does not exist, and nothing in Dave/HAL makes
        it, so the directory is created here. No-op when the round_info has no
        ``data_dir`` column or the value is blank/NaN.
        """
        if not has_data_dir or not pd.notna(dir_value):
            return
        dpath = str(dir_value).strip()
        if not dpath:
            return
        ET.SubElement(seq, "change_directory").text = dpath
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
        if not is_last:
            next_round = round_id + 1
            fl_name    = f"Fluidics Round {next_round:02d}"

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
                    fl_protocols = cleave + [
                        resolver.hybridize(hyb_idx, adaptors=True),
                        resolver.readouts(),
                        resolver.image_buffer(),
                    ]
                else:
                    fl_protocols = cleave + [
                        resolver.hybridize(hyb_idx, adaptors=False),
                        resolver.image_buffer(),
                    ]
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

        fl_loop = ET.SubElement(seq, "loop")
        fl_loop.set("name", fl_name)
        ve = ET.SubElement(fl_loop, "variable_entry")
        ve.set("name", fl_name)
        fluidics_loop_vars.append((fl_name, fl_protocols))

    for idx, round_id in enumerate(round_ids):
        is_last = (idx == n_rounds - 1)
        rows    = round_info[round_info["imaging_round"] == round_id]

        if segment_mode:
            # One loop per segment (a Dave loop iterates a single positions file).
            # Each segment sets its own save directory just before its loop.
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
            # save directory (from the round's first row's data_dir).
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
