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
  [end_seq]    at z=bead_z           ← blanks to allow Z-nanopositioner return to z_bead

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
    "MF2": {np.nan: np.nan, 405: 4, 488: 3, 560: 2, 650: 1, 750: 0},
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
    scan_mode:  str = "interleaved",
) -> pd.DataFrame:
    """
    Build the frame table describing one imaging round.

    Parameters
    ----------
    bead_z     : z position (µm above coverslip) for fiducial bead frames
    bead_seq   : ordered list of colours (nm) or ``np.nan`` for blank frames,
                 acquired at *bead_z* **before** the z-stack
    color_seq  : colours to acquire across the z-stack
    end_seq    : colours acquired at *bead_z* **after** the z-stack;
                 typically blank frames (``[np.nan, np.nan]``) to allow the
                 Z-nanopositioner to return to z_bead before the next round begins
    z_pos      : 1-D array of z positions for data frames
    microscope : used to resolve hardware channel indices
    scan_mode  : acquisition order for the data frames:

                 ``"interleaved"`` *(default)* — all colors acquired at each
                 Z-nanopositioner position before stepping to the next z.
                 Optimised for AOTF / fast electronic channel switching::

                   z=z_pos[0]:  color_seq[0]  color_seq[1]  …
                   z=z_pos[1]:  color_seq[0]  color_seq[1]  …
                   …

                 ``"sequential"`` — full Z-nanopositioner sweep acquired per
                 color, then switch to the next color.  Z is traversed in a
                 boustrophedon (snake) pattern to avoid retracing.  Optimised
                 for physical shutters / slow channel switching::

                   color_seq[0]:  z_pos[0] … z_pos[-1]   (ascending)
                   color_seq[1]:  z_pos[-1] … z_pos[0]   (descending)
                   color_seq[2]:  z_pos[0] … z_pos[-1]   (ascending)
                   …

    Returns
    -------
    pd.DataFrame with columns ``["color", "channel", "z"]`` and an integer
    index equal to the camera frame number (0-based).
    """
    if scan_mode not in ("interleaved", "sequential"):
        raise ValueError(
            f"Unknown scan_mode {scan_mode!r}. Use 'interleaved' or 'sequential'."
        )

    ch_map = get_color_to_channel_dict(microscope)
    rows: List[Dict] = []

    for color in bead_seq:
        rows.append({"color": color, "channel": ch_map[color], "z": bead_z})

    if scan_mode == "interleaved":
        for z in z_pos:
            for color in color_seq:
                rows.append({"color": color, "channel": ch_map[color], "z": z})
    else:  # "sequential"
        for i, color in enumerate(color_seq):
            z_sweep = z_pos if i % 2 == 0 else z_pos[::-1]
            for z in z_sweep:
                rows.append({"color": color, "channel": ch_map[color], "z": z})

    for color in end_seq:
        rows.append({"color": color, "channel": ch_map[color], "z": bead_z})

    return pd.DataFrame(rows, columns=["color", "channel", "z"])


def get_color_sequence_name(
    frame_table: pd.DataFrame,
    separator:   str = "-",
    scan_mode:   str = "interleaved",
) -> str:
    """
    Build a compact human-readable name for a colour sequence.

    Returns something like ``"blkf2-560f49-650f49"`` where ``blkf2`` means
    two blank (NaN) frames and ``560f49`` means forty-nine 560 nm frames.
    A ``"-seq"`` suffix is appended when *scan_mode* is ``"sequential"`` to
    prevent filename collisions with the interleaved variant.

    Parameters
    ----------
    frame_table : DataFrame with a ``"color"`` column
    separator   : string placed between name tokens (default ``"-"``)
    scan_mode   : ``"interleaved"`` (default) or ``"sequential"``
    """
    col     = frame_table["color"]
    n_blank = int(col.isna().sum())
    counts  = col.value_counts(dropna=True)

    parts = []
    if n_blank > 0:
        parts.append(f"blkf{n_blank}")
    for wavelength in sorted(counts.index.astype(float)):
        parts.append(f"{int(wavelength)}f{int(counts.loc[wavelength])}")

    name = separator.join(parts)
    if scan_mode == "sequential":
        name += separator + "seq"
    return name


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
    file_type:        str                   = ".zarr",
    exposure_time:    float                 = 0.25,
) -> None:
    """
    Patch a HAL config XML template and write the result to *output_path*.

    The following XML elements are updated:

    * ``<frames>``        → ``len(frame_table)``
    * ``<shutters>``      → *shutter_filename*
    * ``<z_offsets>``     → derived from ``frame_table["z"]``
    * ``<filetype>``      → *file_type* (e.g. ``".zarr"``, ``".dax"``, ``".tiff"``)
    * ``<exposure_time>`` → *exposure_time* (seconds)
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
    file_type        : image file format written by HAL (``".zarr"``, ``".dax"``,
                       or ``".tiff"``); default ``".zarr"``
    exposure_time    : camera exposure time in seconds; default ``0.25``
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

    text = _sub("frames",        str(len(frame_table)))
    text = _sub("shutters",      shutter_filename)
    text = _sub("z_offsets",     format_z_offsets_from_frame_table(frame_table))
    text = _sub("filetype",      file_type)
    text = _sub("exposure_time", f"{exposure_time:.4f}")

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


# ── HAL config inspection helpers ────────────────────────────────────────────

def read_hal_flip_vertical(hal_config_path: Path) -> bool:
    """
    Return ``True`` if ``<flip_vertical>1</flip_vertical>`` is set in the
    HAL config at *hal_config_path*.  Returns ``False`` on any parse error.
    """
    try:
        with open(hal_config_path, "rb") as fh:
            text = fh.read().decode("ISO-8859-1")
        m = re.search(r"<flip_vertical[^>]*>(\d+)</flip_vertical>", text)
        return bool(m and int(m.group(1)) == 1)
    except Exception:
        return False


def find_frame_table_for_hal_config(
    hal_config_path: Path,
    metadata_dir:    Path,
) -> "Optional[Path]":
    """
    Locate the frame-table CSV that corresponds to *hal_config_path*.

    Strategy: read the ``<shutters>`` element from the HAL config
    (``shutter-{name}.xml``), strip the ``shutter-`` prefix to recover
    *name*, and return ``metadata_dir/frame_table_{name}.csv`` if it exists.

    Returns ``None`` when the frame table cannot be found.
    """
    try:
        with open(hal_config_path, "rb") as fh:
            text = fh.read().decode("ISO-8859-1")
        m = re.search(r"<shutters[^>]*>([^<]+)</shutters>", text)
        if not m:
            return None
        shutter_stem = Path(m.group(1).strip()).stem  # shutter-{name}
        if shutter_stem.startswith("shutter-"):
            name = shutter_stem[len("shutter-"):]
        else:
            name = shutter_stem
        candidate = Path(metadata_dir) / f"frame_table_{name}.csv"
        return candidate if candidate.exists() else None
    except Exception:
        return None


def get_color_frame_indices(
    frame_table: pd.DataFrame,
    bead_z:      float = 0.0,
) -> "Dict[float, int]":
    """
    For each non-blank, non-bead color in *frame_table* return the frame index
    at the z position closest to the midpoint of the z-stack.

    Parameters
    ----------
    frame_table : DataFrame with columns ``["color", "channel", "z"]`` and
                  integer index equal to the camera frame number
    bead_z      : z value used for fiducial (bead) frames; those rows are
                  excluded from the search

    Returns
    -------
    ``{color_nm: frame_idx}`` — one entry per non-blank color; the frame index
    is the first occurrence of that color at the middle-z slice.
    """
    df = frame_table.copy()
    df["frame_idx"] = df.index

    # Keep only data frames (not bead/end frames at bead_z) with real colors
    data = df[(df["z"] != bead_z) & df["color"].notna()].copy()
    if data.empty:
        return {}

    z_vals  = data["z"].unique()
    z_mid   = (z_vals.min() + z_vals.max()) / 2.0
    best_z  = z_vals[int(np.argmin(np.abs(z_vals - z_mid)))]

    at_mid = data[data["z"] == best_z]
    result: Dict[float, int] = {}
    for _, row in at_mid.iterrows():
        color = float(row["color"])
        if color not in result:
            result[color] = int(row["frame_idx"])
    return result


# ── Frame table reconstruction (inverse of get_frame_table) ──────────────────

def read_shutter_reference(hal_config_path: Path) -> str:
    """
    Return the shutter filename referenced by the ``<shutters>`` element of the
    HAL config at *hal_config_path* (e.g. ``"shutter-blkf2-560f49-650f49.xml"``).

    Raises
    ------
    ValueError
        If no ``<shutters>`` element is found.
    """
    with open(hal_config_path, "rb") as fh:
        text = fh.read().decode("ISO-8859-1")
    m = re.search(r"<shutters[^>]*>([^<]+)</shutters>", text)
    if not m:
        raise ValueError(
            f"No <shutters> element found in HAL config: {hal_config_path}"
        )
    return m.group(1).strip()


def read_hal_frame_count(hal_config_path: Path) -> "Optional[int]":
    """Return the ``<frames>`` value from the HAL config, or ``None`` if absent."""
    with open(hal_config_path, "rb") as fh:
        text = fh.read().decode("ISO-8859-1")
    m = re.search(r"<frames[^>]*>\s*(\d+)\s*</frames>", text)
    return int(m.group(1)) if m else None


def parse_z_offsets(hal_config_path: Path) -> List[float]:
    """
    Parse the ``<z_offsets>`` element of a HAL config into one float per frame.

    Inverse of :func:`format_z_offsets_from_frame_table`.

    Raises
    ------
    ValueError
        If the HAL config has no (non-empty) ``<z_offsets>`` element — e.g. it
        does not use the hardware Z-nanopositioner scan — so per-frame z cannot
        be determined.
    """
    with open(hal_config_path, "rb") as fh:
        text = fh.read().decode("ISO-8859-1")
    m = re.search(r"<z_offsets[^>]*>(.*?)</z_offsets>", text, flags=re.DOTALL)
    if not m or not m.group(1).strip():
        raise ValueError(
            f"HAL config has no <z_offsets>; per-frame z cannot be determined: "
            f"{hal_config_path}"
        )
    values = re.findall(r"[-+]?\d*\.?\d+", m.group(1))
    return [float(v) for v in values]


def parse_shutter_events(shutter_path: Path) -> "Dict[int, int]":
    """
    Map ``frame_index -> channel`` for every ``<event>`` in a shutter XML file.

    Inverse of the per-frame events written by :func:`create_shutter_file`.
    The frame index is ``round(float(<on>))``; blank frames have no event and
    are therefore absent from the returned mapping.

    Raises
    ------
    ValueError
        If two events map to the same frame index (ambiguous).
    """
    with open(shutter_path, "rb") as fh:
        root = ET.fromstring(fh.read().decode("ISO-8859-1"))

    oversampling_el = root.find("oversampling")
    if (oversampling_el is not None and oversampling_el.text
            and int(float(oversampling_el.text)) != 1):
        import warnings
        warnings.warn(
            f"Shutter <oversampling> is {oversampling_el.text.strip()} "
            f"(expected 1); frame indices are read from <on> as written by MERci.",
            stacklevel=2,
        )

    events: Dict[int, int] = {}
    for event in root.findall("event"):
        on_el      = event.find("on")
        channel_el = event.find("channel")
        if on_el is None or channel_el is None:
            continue
        frame_idx = int(round(float(on_el.text)))
        if frame_idx in events:
            raise ValueError(
                f"Two shutter events map to frame index {frame_idx} in {shutter_path}"
            )
        events[frame_idx] = int(channel_el.text)
    return events


def reconstruct_frame_table(
    hal_config_path: Path,
    shutter_path:    "Optional[Path]" = None,
    microscope:      str              = "MF3",
) -> pd.DataFrame:
    """
    Rebuild a frame table from a HAL config XML and its shutter XML — the inverse
    of :func:`get_frame_table` / :func:`create_shutter_file`.

    Per-frame ``z`` comes from the HAL config ``<z_offsets>``; per-frame
    ``channel`` comes from the shutter ``<event>`` list (frames with no event
    are blank); ``color`` is recovered by inverting
    :func:`get_color_to_channel_dict`.

    Parameters
    ----------
    hal_config_path : HAL config XML written by :func:`create_hal_config`
    shutter_path    : shutter XML; if ``None`` it is resolved from the HAL
                      config's ``<shutters>`` element, relative to the HAL
                      config's directory
    microscope      : microscope name for the channel→color mapping

    Returns
    -------
    DataFrame with columns ``["color", "channel", "z"]`` and a RangeIndex equal
    to the camera frame number (0-based). Blank frames have ``NaN`` color and
    channel — identical in structure to :func:`get_frame_table`'s output.

    Raises
    ------
    FileNotFoundError
        If the referenced/derived shutter file does not exist.
    ValueError
        On a frame-count mismatch, a missing ``<z_offsets>``, or an event
        channel that is not in the microscope's channel map.
    """
    hal_config_path = Path(hal_config_path)

    if shutter_path is None:
        shutter_path = hal_config_path.parent / read_shutter_reference(hal_config_path)
    shutter_path = Path(shutter_path)
    if not shutter_path.exists():
        raise FileNotFoundError(f"Shutter file not found: {shutter_path}")

    z_offsets = parse_z_offsets(hal_config_path)
    events    = parse_shutter_events(shutter_path)
    n_frames  = len(z_offsets)

    # Cross-check the frame count declared in the HAL config.
    hal_frames = read_hal_frame_count(hal_config_path)
    if hal_frames is not None and hal_frames != n_frames:
        raise ValueError(
            f"Frame-count mismatch: HAL <frames>={hal_frames} but <z_offsets> "
            f"has {n_frames} values ({hal_config_path})"
        )

    # Every shutter event must fall within the frame range implied by z_offsets.
    stray = sorted(i for i in events if i < 0 or i >= n_frames)
    if stray:
        raise ValueError(
            f"Shutter has events for frame(s) {stray} outside the {n_frames}-frame "
            f"range implied by <z_offsets> — mismatched HAL/shutter pair "
            f"({shutter_path})"
        )

    # channel → color (drop the NaN→NaN entry)
    inv = {
        int(ch): color
        for color, ch in get_color_to_channel_dict(microscope).items()
        if not pd.isna(ch)
    }

    rows: List[Dict] = []
    for i in range(n_frames):
        if i in events:
            channel = events[i]
            if channel not in inv:
                raise ValueError(
                    f"Shutter event at frame {i} has channel {channel}, which is "
                    f"not in the '{microscope}' channel map {sorted(inv)}"
                )
            rows.append({"color":   float(inv[channel]),
                         "channel": float(channel),
                         "z":       z_offsets[i]})
        else:
            rows.append({"color": np.nan, "channel": np.nan, "z": z_offsets[i]})

    return pd.DataFrame(rows, columns=["color", "channel", "z"])


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