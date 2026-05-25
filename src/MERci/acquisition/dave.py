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
Round 1 (imaging): bits acquisition + cells acquisition for all FOVs
Fluidics Round 02: Cleave → Hybridize/Adaptor 2 → [Readouts] → Buffer
Round 2 (imaging): bits acquisition for all FOVs
Fluidics Round 03: Cleave → Hybridize/Adaptor 3 → [Readouts] → Buffer
…
Round N (imaging): bits acquisition
[Optional] Fluidics Final: Cleave only

Fluidics loops are named by the NEXT imaging round (e.g. "Fluidics Round 02"
precedes "Imaging Round 02").

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

    The first imaging round contains both a bits acquisition and a cells
    acquisition.  Subsequent rounds contain only the bits acquisition.

    Parameters
    ----------
    microscope        : microscope identifier in lowercase, e.g. ``"mf3"``
    n_bits            : number of bits (hybridisation) rounds
    bits_hal_config   : HAL config filename for bits rounds (with ``.xml``)
    cells_hal_config  : HAL config filename for the cells round (with ``.xml``)
    sample_dir        : experiment root directory; used to build ``data_dir`` paths

    Returns
    -------
    pd.DataFrame with columns ``imaging_round``, ``series``, ``hal_config``, ``data_dir``
    """
    mic      = microscope.lower()
    data_dir = str(sample_dir / "data")
    rows: List[dict] = []

    # Round 1: bits + cells (cells series uses hyphen: hal-{mic}-epi-cells)
    rows.append({
        "imaging_round": 1,
        "series":        f"hal-{mic}-epi_01_{{fov:03d}}",
        "hal_config":    bits_hal_config,
        "data_dir":      data_dir,
    })
    rows.append({
        "imaging_round": 1,
        "series":        f"hal-{mic}-epi-cells_{{fov:03d}}",
        "hal_config":    cells_hal_config,
        "data_dir":      str(sample_dir / "data" / "cells"),
    })

    # Rounds 2 … N: bits only
    for i in range(2, n_bits + 1):
        rows.append({
            "imaging_round": i,
            "series":        f"hal-{mic}-epi_{i:02d}_{{fov:03d}}",
            "hal_config":    bits_hal_config,
            "data_dir":      data_dir,
        })

    return pd.DataFrame(rows, columns=["imaging_round", "series", "hal_config", "data_dir"])


# ── Dave config builder ────────────────────────────────────────────────────────

def create_dave_config(
    round_info:           pd.DataFrame,
    positions_file:       Path,
    settings_dir:         Path,
    output_path:          Path,
    use_adaptors:         bool = False,
    include_final_cleave: bool = False,
    num_focus_checks:     int  = 50,
    fluidics_protocols:   Optional[Sequence[str]] = None,
) -> None:
    """
    Write an explicit-block Dave recipe XML from ``round_info``.

    Fluidics loops are named by the NEXT imaging round (e.g. "Fluidics Round 02"
    precedes "Imaging Round 02").  The last imaging round has no trailing
    fluidics unless ``include_final_cleave=True``.

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
    num_focus_checks      : value for ``<num_focus_checks>``
    fluidics_protocols    : if provided, use this fixed list of Kilroy protocol
                            names for every between-round fluidics block,
                            overriding ``use_adaptors``
    """
    round_ids = sorted(round_info["imaging_round"].unique())
    n_rounds  = len(round_ids)

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
            if fluidics_protocols is not None:
                fl_protocols = list(fluidics_protocols)
            elif use_adaptors:
                fl_protocols = [
                    "Cleave adaptors",
                    f"Hyb adaptors {next_round}",
                    "Hyb readouts",
                    "Flow Image Buffer",
                ]
            else:
                fl_protocols = [
                    "Cleave direct",
                    f"Hybridize {next_round}",
                    "Wash and Imaging Buffers",
                ]
        elif include_final_cleave:
            fl_name      = "Fluidics Final"
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

    Parameters
    ----------
    dave_path       : path to the Dave XML file to annotate (modified in-place)
    round_bit_color : list of ``(round_1indexed, bit_number, color_nm)`` tuples
    """
    # Group bits by round and build comment strings
    bits_by_round: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for round_1idx, bit, color_nm in round_bit_color:
        bits_by_round[round_1idx].append((bit, color_nm))

    round_comments: dict[int, str] = {}
    for round_1idx, bit_colors in sorted(bits_by_round.items()):
        # Sort bits by color descending (750 → 650 → 560)
        bit_strs = ", ".join(
            f"Bit {bit} ({color} nm)"
            for bit, color in sorted(bit_colors, key=lambda x: x[1], reverse=True)
        )
        round_comments[round_1idx] = f"Round {round_1idx}: {bit_strs}"

    # Read file preserving raw CRLF so split("\r\n") works correctly
    with open(dave_path, "r", encoding="ISO-8859-1", newline="") as fh:
        content = fh.read()

    lines = content.split("\r\n")
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        indent   = line[: len(line) - len(line.lstrip())]

        # Round 1: insert before "Imaging Round 01" loop
        if stripped == '<loop name="Imaging Round 01">':
            if 1 in round_comments:
                new_lines.append("")
                new_lines.append(f"{indent}<!-- {round_comments[1]} -->")
        else:
            # Rounds 2+: insert before the corresponding "Fluidics Round NN" loop
            m = re.match(r'^<loop name="Fluidics Round (\d+)">', stripped)
            if m:
                n = int(m.group(1))
                if n in round_comments:
                    new_lines.append("")
                    new_lines.append(f"{indent}<!-- {round_comments[n]} -->")

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
