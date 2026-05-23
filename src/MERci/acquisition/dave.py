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
Fluidics 1:        Cleave → Hybridize 1 → Wash and Imaging Buffers
Round 2 (imaging): bits acquisition for all FOVs
Fluidics 2:        Cleave → Hybridize 2 → Wash and Imaging Buffers
…
Round N (imaging): bits acquisition
Final fluidics:    Cleave only

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
    ``hal-mf3-epi_01_{fov:03d}``   → ``hal-mf3-epi_01``
    ``hal-mf3-epi_cells_{fov:03d}`` → ``hal-mf3-epi_cells``
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
    sample_dir        : experiment root directory; used to build ``dir`` paths

    Returns
    -------
    pd.DataFrame with columns ``imaging_round``, ``series``, ``hal_config``, ``dir``
    """
    mic   = microscope.lower()
    rows: List[dict] = []

    # Round 1: bits + cells
    rows.append({
        "imaging_round": 1,
        "series":        f"hal-{mic}-epi_01_{{fov:03d}}",
        "hal_config":    bits_hal_config,
        "dir":           str(sample_dir / "data" / "H01"),
    })
    rows.append({
        "imaging_round": 1,
        "series":        f"hal-{mic}-epi_cells_{{fov:03d}}",
        "hal_config":    cells_hal_config,
        "dir":           str(sample_dir / "data" / "cells"),
    })

    # Rounds 2 … N: bits only
    for i in range(2, n_bits + 1):
        rows.append({
            "imaging_round": i,
            "series":        f"hal-{mic}-epi_{i:02d}_{{fov:03d}}",
            "hal_config":    bits_hal_config,
            "dir":           str(sample_dir / "data" / f"H{i:02d}"),
        })

    return pd.DataFrame(rows, columns=["imaging_round", "series", "hal_config", "dir"])


# ── Dave config builder ────────────────────────────────────────────────────────

def create_dave_config(
    round_info:              pd.DataFrame,
    positions_file:          Path,
    settings_dir:            Path,
    output_path:             Path,
    num_focus_checks:        int = 50,
    fluidics_protocols:      Optional[Sequence[str]] = None,
    final_fluidics_protocol: str = "Cleave direct",
) -> None:
    """
    Write an explicit-block Dave recipe XML from ``round_info``.

    Each unique ``imaging_round`` value produces one imaging loop followed by
    one fluidics loop.  The last imaging round is followed only by
    ``final_fluidics_protocol``.

    Parameters
    ----------
    round_info              : DataFrame with columns ``imaging_round``,
                              ``series``, ``hal_config``
    positions_file          : path to ``positions_*.txt``; written as-is into
                              each ``<loop_variable>/<file_path>``
    settings_dir            : directory containing the HAL config XML files
                              (used to read ``<frames>`` counts)
    output_path             : where to write the recipe XML
    num_focus_checks        : value for ``<num_focus_checks>``
    fluidics_protocols      : ordered list of Kilroy protocol names executed
                              between imaging rounds; defaults to
                              ``["Cleave direct", "Hybridize {N}", "Wash and Imaging Buffers"]``
                              where N is the imaging round number
    final_fluidics_protocol : single protocol run after the last imaging round
    """
    round_ids = sorted(round_info["imaging_round"].unique())
    n_rounds  = len(round_ids)

    root = ET.Element("recipe")
    seq  = ET.SubElement(root, "command_sequence")

    imaging_loop_vars: list[tuple[str, str]] = []   # (loop_name, positions_path)
    fluidics_loop_vars: list[tuple[str, list[str]]] = []  # (loop_name, [protocols])

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
            hal_stem     = Path(hal_filename).stem   # strip .xml if present
            hal_path     = settings_dir / (hal_stem + ".xml")

            try:
                n_frames = get_hal_frame_count(hal_path)
            except (FileNotFoundError, ValueError):
                n_frames = 0   # leave as 0 if config not yet written

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
        if is_last:
            fl_name      = "Fluidics Final"
            fl_protocols = [final_fluidics_protocol]
        else:
            fl_name = f"Fluidics Round {round_id:02d}"
            if fluidics_protocols is not None:
                fl_protocols = list(fluidics_protocols)
            else:
                fl_protocols = [
                    "Cleave direct",
                    f"Hybridize {round_id}",
                    "Wash and Imaging Buffers",
                ]

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


# ── XML writer ─────────────────────────────────────────────────────────────────

def _write_dave_xml(root: ET.Element, output_path: Path) -> None:
    """Serialize the recipe with indentation and CRLF line endings."""
    raw  = ET.tostring(root, encoding="utf-8")
    dom  = minidom.parseString(raw)
    text = dom.toprettyxml(indent="\t", encoding="ISO-8859-1").decode("ISO-8859-1")

    # Remove the extra blank line toprettyxml adds before every element
    text = re.sub(r"\n\t*\n", "\n", text)
    text = text.replace("\n", "\r\n")

    with open(output_path, "w", encoding="ISO-8859-1", newline="") as fh:
        fh.write(text)
