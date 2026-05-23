# MERci/acquisition/configs.py
"""
Generate HAL shutter XML and HAL config XML files for MERFISH imaging rounds.

A HAL imaging round is fully described by a *frame table*: one row per camera
frame, with columns ``color`` (laser wavelength in nm, or NaN for a blank
frame), ``channel`` (hardware channel index, 0–4), and ``z`` (distance from
the locked focus in µm).

Typical round structure
-----------------------
  [bead_seq]   at z=bead_z           ← fiducial images (e.g. 488)
  [color_seq]  at z=z_pos[0]
  [color_seq]  at z=z_pos[1]
  …
  [end_seq]    at z=bead_z           ← blanks to allow stage return

Usage
-----
>>> frame_table = get_frame_table(
...     bead_z=0, bead_seq=[488, np.nan], color_seq=[560, 650],
...     end_seq=[np.nan, np.nan], z_pos=np.arange(1, 20.5, 0.5),
... )
>>> name = get_color_sequence_name(frame_table)   # e.g. "blkf2-560f49-650f49"
>>> create_shutter_file(frame_table, output_dir / f"shutter_{name}.xml")
>>> create_hal_config(template, frame_table, f"shutter_{name}.xml",
...                   output_dir / f"hal-config-mf3-epi_{name}.xml")
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# ── Channel / colour mapping ─────────────────────────────────────────────────

_COLOUR_TO_CHANNEL: Dict[str, Dict] = {
    "MF3": {np.nan: np.nan, 405: 4, 488: 3, 560: 2, 650: 1, 750: 0},
    "MF5": {np.nan: np.nan, 405: 4, 488: 3, 560: 2, 650: 1, 750: 0},
}


def get_color_to_channel_dict(microscope: str = "MF3") -> Dict:
    """
    Return the wavelength (nm) → hardware-channel-index mapping for
    *microscope*.

    Parameters
    ----------
    microscope : ``"MF3"`` or ``"MF5"``
    """
    if microscope not in _COLOUR_TO_CHANNEL:
        raise ValueError(
            f"Unknown microscope '{microscope}'. "
            f"Supported values: {list(_COLOUR_TO_CHANNEL)}"
        )
    return _COLOUR_TO_CHANNEL[microscope]


# ── Frame table ───────────────────────────────────────────────────────────────

def get_frame_table(
    bead_z:     float,
    bead_seq:   List,
    color_seq:  List,
    end_seq:    List,
    z_pos:      np.ndarray,
    microscope: str = "MF3",
) -> pd.DataFrame:
    """
    Build the frame table describing one imaging round.

    Parameters
    ----------
    bead_z     : z position (µm above coverslip) for fiducial bead frames
    bead_seq   : ordered list of colours (nm) or ``np.nan`` for blank frames,
                 acquired at *bead_z* **before** the z-stack
    color_seq  : colours acquired at every position in *z_pos*
    end_seq    : colours acquired at *bead_z* **after** the z-stack;
                 typically blank frames (``[np.nan, np.nan]``) to allow the
                 stage to return before the next round begins
    z_pos      : 1-D array of z positions for data frames
    microscope : used to resolve hardware channel indices

    Returns
    -------
    pd.DataFrame with columns ``["color", "channel", "z"]`` and an integer
    index equal to the camera frame number (0-based).
    """
    ch_map = get_color_to_channel_dict(microscope)
    rows: List[Dict] = []

    for color in bead_seq:
        rows.append({"color": color, "channel": ch_map[color], "z": bead_z})

    for z in z_pos:
        for color in color_seq:
            rows.append({"color": color, "channel": ch_map[color], "z": z})

    for color in end_seq:
        rows.append({"color": color, "channel": ch_map[color], "z": bead_z})

    return pd.DataFrame(rows, columns=["color", "channel", "z"])


def get_color_sequence_name(
    frame_table: pd.DataFrame,
    separator:   str = "-",
) -> str:
    """
    Build a compact human-readable name for a colour sequence.

    Returns something like ``"blkf2-560f49-650f49"`` where ``blkf2`` means
    two blank (NaN) frames and ``560f49`` means forty-nine 560 nm frames.

    Parameters
    ----------
    frame_table : DataFrame with a ``"color"`` column
    separator   : string placed between name tokens (default ``"-"``)
    """
    col     = frame_table["color"]
    n_blank = int(col.isna().sum())
    counts  = col.value_counts(dropna=True)

    parts = []
    if n_blank > 0:
        parts.append(f"blkf{n_blank}")
    for wavelength in sorted(counts.index.astype(float)):
        parts.append(f"{int(wavelength)}f{int(counts.loc[wavelength])}")

    return separator.join(parts)


# ── Shutter XML ───────────────────────────────────────────────────────────────

def create_shutter_file(
    frame_table:   pd.DataFrame,
    output_path:   Path,
    oversampling:  int   = 1,
    default_power: float = 1.0,
) -> None:
    """
    Write a HAL shutter XML file from *frame_table*.

    Parameters
    ----------
    frame_table   : DataFrame produced by :func:`get_frame_table`
    output_path   : destination path; written with Windows (CRLF) line endings
    oversampling  : value for the ``<oversampling>`` XML element
    default_power : laser power written for every ``<event>``
    """
    df   = frame_table.sort_index()
    root = ET.Element("repeat")

    ET.SubElement(root, "oversampling").text = str(oversampling)
    ET.SubElement(root, "frames").text       = str(len(df))

    last_z = None
    for frame, row in df.iterrows():
        channel = row["channel"]
        z       = row["z"]
        color   = row.get("color", np.nan)

        if pd.isna(channel):
            continue

        if last_z is None or z != last_z:
            root.append(ET.Comment(_z_comment(z, color)))
            last_z = z

        event = ET.SubElement(root, "event")
        ET.SubElement(event, "channel").text = str(int(channel))
        ET.SubElement(event, "power").text   = f"{default_power:.1f}"
        ET.SubElement(event, "on").text      = f"{float(frame):.1f}"
        ET.SubElement(event, "off").text     = f"{float(frame + 1):.1f}"

    _write_pretty_xml(root, Path(output_path))


def _z_comment(z: float, color) -> str:
    if z == 0 and not pd.isna(color) and int(color) == 405:
        return f" z = {int(z)} um, 405 beads"
    z_str = f"{int(z)}" if float(z).is_integer() else f"{z}"
    return f" z = {z_str} um"


# ── HAL config XML ────────────────────────────────────────────────────────────

def create_hal_config(
    template_path:    Path,
    frame_table:      pd.DataFrame,
    shutter_filename: str,
    output_path:      Path,
    default_power:    Optional[List[float]] = None,
) -> None:
    """
    Patch a HAL config XML template and write the result to *output_path*.

    The following XML elements are updated:

    * ``<frames>``        → ``len(frame_table)``
    * ``<shutters>``      → *shutter_filename*
    * ``<z_offsets>``     → derived from ``frame_table["z"]``
    * ``<default_power>`` → *default_power* (only if provided)

    All comments and overall formatting are preserved.

    Parameters
    ----------
    template_path    : path to an existing HAL config XML file
    frame_table      : DataFrame produced by :func:`get_frame_table`
    shutter_filename : basename of the shutter file,
                       e.g. ``"shutter_blkf2-560f49-650f49.xml"``
    output_path      : where to write the patched config
    default_power    : list of per-channel power values; omitted if ``None``
    """
    with open(template_path, "rb") as fh:
        text = fh.read().decode("ISO-8859-1").replace("\r\n", "\n")

    def _sub(tag: str, new_text: str) -> str:
        def repl(m):
            return f"{m.group(1)}{new_text}{m.group(3)}"
        return re.sub(
            rf"(<{tag}[^>]*>)(.*?)(</{tag}>)",
            repl,
            text,
            flags=re.DOTALL,
        )

    text = _sub("frames",    str(len(frame_table)))
    text = _sub("shutters",  shutter_filename)
    text = _sub("z_offsets", format_z_offsets_from_frame_table(frame_table))

    if default_power is not None:
        text = _sub("default_power", ",".join(str(v) for v in default_power))

    # Ensure exactly one blank line before each XML comment
    text = re.sub(r"(?<!\n\n)\n([ \t]*)<!--", r"\n\n\1<!--", text)

    with open(Path(output_path), "w", encoding="ISO-8859-1", newline="") as fh:
        fh.write(text.replace("\n", "\r\n"))


def format_z_offsets_from_frame_table(frame_table: pd.DataFrame) -> str:
    """
    Build the text content of the ``<z_offsets>`` XML element from
    ``frame_table["z"]``.

    Values are laid out in rows matching the colour sequence length (the most
    common consecutive-run length), with a comma after every value except the
    last.  Bead and end frames with a different run length are handled gracefully.
    """
    from collections import Counter

    z_vals = frame_table["z"].astype(float).tolist()
    n      = len(z_vals)

    # Determine row width from the most common consecutive run length
    run_lengths: List[int] = []
    count, last = 0, object()
    for val in z_vals:
        if val == last:
            count += 1
        else:
            if last is not object():
                run_lengths.append(count)
            count, last = 1, val
    if count:
        run_lengths.append(count)

    group_size = Counter(run_lengths).most_common(1)[0][0] if run_lengths else 1

    # Format all z values in rows of group_size
    indent  = "         "
    lines:  List[str] = []
    row_buf: List[str] = []

    for i, val in enumerate(z_vals):
        suffix = "," if i < n - 1 else ""
        row_buf.append(f"{val:.1f}{suffix}")
        if len(row_buf) == group_size:
            lines.append(indent + "  ".join(row_buf))
            row_buf = []

    if row_buf:
        lines.append(indent + "  ".join(row_buf))

    return "\n" + "\n".join(lines) + "\n      "


# ── Pretty-print helper ───────────────────────────────────────────────────────

def _write_pretty_xml(root: ET.Element, output_path: Path) -> None:
    """
    Serialize *root* as indented XML with CRLF line endings.
    Adds one blank line before each comment for legibility.
    """
    raw  = ET.tostring(root, encoding="utf-8")
    dom  = minidom.parseString(raw)
    text = dom.toprettyxml(indent="  ", encoding="ISO-8859-1").decode("ISO-8859-1")
    text = text.replace("\n  <!--", "\n\n  <!--")
    text = text.replace("\n", "\r\n")

    with open(output_path, "w", encoding="ISO-8859-1", newline="") as fh:
        fh.write(text)