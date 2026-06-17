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
    data_dir = str(sample_dir / "data")
    rows: List[dict] = []

    # Imaging Round 1: CELLS ONLY (no fluidics precedes it).
    rows.append({
        "imaging_round": 1,
        "imaging_type":  "cells",
        "series":        f"hal-{mic}-epi-cells_{{fov:03d}}",
        "hal_config":    cells_hal_config,
        "data_dir":      str(sample_dir / "data" / "cells"),
    })

    # Imaging Rounds 2 … N+1: bits #1 … #N.  The series number tracks the
    # bit/hyb index (1…N); the imaging_round is bit_idx + 1.
    for bit_idx in range(1, n_bits + 1):
        rows.append({
            "imaging_round": bit_idx + 1,
            "imaging_type":  "bits",
            "series":        f"hal-{mic}-epi_{bit_idx:02d}_{{fov:03d}}",
            "hal_config":    bits_hal_config,
            "data_dir":      data_dir,
        })

    return pd.DataFrame(
        rows,
        columns=["imaging_round", "imaging_type", "series", "hal_config", "data_dir"],
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
) -> None:
    """
    Write an explicit-block Dave recipe XML from ``round_info``.

    Fluidics loops are named by the NEXT imaging round (e.g. "Fluidics Round 02"
    precedes "Imaging Round 02").  The hyb-protocol number tracks the bit/hyb
    index (the count of bits rounds reached so far), not the imaging-round
    number, so a leading cells round does not shift the Kilroy protocol names.
    The last imaging round has no trailing fluidics unless
    ``include_final_cleave=True``.

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
    """
    round_ids = sorted(round_info["imaging_round"].unique())
    n_rounds  = len(round_ids)

    # Resolve fluidic protocol names against the Kilroy config that will run this
    # experiment, so every protocol written here exists as a Kilroy <protocol>.
    resolver = (
        KilroyProtocolResolver(load_kilroy_protocols(kilroy_config))
        if kilroy_config is not None else None
    )

    def _round_is_cells(rid: int) -> bool:
        """A round is a cells round if its imaging_type is 'cells' (or, absent
        that column, every series name contains 'cells')."""
        rrows = round_info[round_info["imaging_round"] == rid]
        if "imaging_type" in round_info.columns:
            types = {str(t).strip().lower() for t in rrows["imaging_type"].dropna()}
            if types:
                return types == {"cells"}
        return all("cells" in str(s) for s in rrows["series"])

    bits_round_ids   = [rid for rid in round_ids if not _round_is_cells(rid)]
    first_bits_round = bits_round_ids[0] if bits_round_ids else None

    root = ET.Element("recipe")
    seq  = ET.SubElement(root, "command_sequence")

    imaging_loop_vars:  list[tuple[str, str]]       = []
    fluidics_loop_vars: list[tuple[str, list[str]]] = []

    for idx, round_id in enumerate(round_ids):
        is_last  = (idx == n_rounds - 1)
        rows     = round_info[round_info["imaging_round"] == round_id]
        img_name = f"Imaging Round {round_id:02d}"

        # ── Imaging loop ───────────────────────────────────────────────────────
        img_loop = ET.SubElement(seq, "loop")
        img_loop.set("name", img_name)

        for _, row in rows.iterrows():
            movie_name   = series_to_movie_name(str(row["series"]))
            hal_filename = str(row["hal_config"])
            hal_stem     = Path(hal_filename).stem
            hal_path     = settings_dir / (hal_stem + ".xml")

            try:
                n_frames = get_hal_frame_count(hal_path)
            except (FileNotFoundError, ValueError):
                n_frames = 0

            movie = ET.SubElement(img_loop, "movie")
            name_el = ET.SubElement(movie, "name")
            name_el.set("increment", "Yes")
            name_el.text = movie_name

            ET.SubElement(movie, "length").text = str(n_frames)
            ET.SubElement(movie, "parameters").text = hal_stem

            cf = ET.SubElement(movie, "check_focus")
            ET.SubElement(cf, "num_focus_checks").text = str(num_focus_checks)
            ET.SubElement(cf, "focus_scan")

            ET.SubElement(movie, "overwrite").text = "False"

            ve = ET.SubElement(movie, "variable_entry")
            ve.set("name", img_name)

        imaging_loop_vars.append((img_name, str(positions_file)))

        # ── Fluidics loop ──────────────────────────────────────────────────────
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
            fl_name      = "Fluidics Final"
            if resolver is not None:
                fl_protocols = [resolver.cleave(adaptors=use_adaptors)]
            else:
                fl_protocols = ["Cleave adaptors" if use_adaptors else "Cleave direct"]
        else:
            fl_name = None

        if fl_name is not None:
            fl_loop = ET.SubElement(seq, "loop")
            fl_loop.set("name", fl_name)
            ve = ET.SubElement(fl_loop, "variable_entry")
            ve.set("name", fl_name)
            fluidics_loop_vars.append((fl_name, fl_protocols))

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
