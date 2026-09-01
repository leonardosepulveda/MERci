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
  [z return]   z_pos[-1] → z_bead    ← how the objective travels back to the coverslip
  [end_seq]    at z=bead_z           ← frames acquired at z_bead before the next round

The z return is controlled by ``z_return_mode``:
  • ``"progressive"`` *(default)* — blank (laser-off) frames step the objective
    down from the last stack position to z_bead in increments of ``return_step``.
  • ``"instant"`` — the objective jumps straight to z_bead (no intermediate frames).

Usage
-----
>>> frame_table = get_frame_table(
...     bead_z=0, bead_seq=[488, np.nan], color_seq=[560, 650],
...     end_seq=[np.nan, np.nan], z_pos=np.arange(1, 20.5, 0.5),
... )
>>> name = get_color_sequence_name(frame_table)   # e.g. "blkf2_560f49_650f49"
>>> create_shutter_file(frame_table, output_dir / shutter_filename("bits", name))
>>> create_hal_config(template, frame_table, shutter_filename("bits", name),
...                   output_dir / hal_config_filename("mf3", "bits", name))
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# ── Channel / colour mapping ─────────────────────────────────────────────────

_COLOUR_TO_CHANNEL: Dict[str, Dict] = {
    "MF2": {np.nan: np.nan, 405: 4, 488: 3, 560: 2, 650: 1, 750: 0},
    "MF3": {np.nan: np.nan, 405: 4, 488: 3, 560: 2, 650: 1, 750: 0},
    "MF4": {np.nan: np.nan, 405: 4, 488: 3, 560: 2, 650: 1, 750: 0},
    "MF5": {np.nan: np.nan, 405: 4, 488: 3, 560: 2, 650: 1, 750: 0},
    # MFX has only 4 channels (no 750) with a distinct ordering: 0:650, 1:560, 2:488, 3:405
    "MFX": {np.nan: np.nan, 405: 3, 488: 2, 560: 1, 650: 0},
    # ST2 shares MFX's 4-channel mapping (no 750)
    "ST2": {np.nan: np.nan, 405: 3, 488: 2, 560: 1, 650: 0},
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


# ── Camera frame size ────────────────────────────────────────────────────────
# Camera sensor size (square, pixels) per microscope. MFX and ST2 have 2304-px
# cameras; the MF-series (MF2-MF5) have 2048-px cameras. Choosing the microscope
# therefore fixes the frame size, which drives the storage estimate
# (bytes = width * height * bytes_per_pixel * frames).
_CAMERA_PIXELS: Dict[str, int] = {
    "MF2": 2048, "MF3": 2048, "MF4": 2048, "MF5": 2048,
    "MFX": 2304, "ST2": 2304,
}
_DEFAULT_CAMERA_PIXELS = 2048


def get_camera_frame_size(microscope: Optional[str]) -> Tuple[int, int]:
    """
    Return the ``(width, height)`` camera frame size in pixels for *microscope*.

    MFX and ST2 have 2304×2304 sensors; the MF-series (MF2–MF5) have 2048×2048.
    An unknown or ``None`` microscope falls back to 2048×2048 (no error, so
    estimates still run) — extend ``_CAMERA_PIXELS`` for new scopes.

    Parameters
    ----------
    microscope : microscope id, case-insensitive (e.g. ``"MF3"``, ``"mfx"``); may
                 be ``None``

    Returns
    -------
    (width, height) : frame size in pixels (square sensor)
    """
    key = str(microscope).strip().upper() if microscope is not None else ""
    n   = _CAMERA_PIXELS.get(key, _DEFAULT_CAMERA_PIXELS)
    return (n, n)


# Camera pixel size projected onto the sample (µm/pixel), per (microscope,
# objective). Pixel size depends on both the camera's physical pixel pitch
# (fixed per microscope, see _CAMERA_PIXELS above) and the objective's
# magnification -- unlike frame size/channel map/acquisition type above,
# which are camera/scope hardware properties independent of which objective
# is mounted, this needs a second key. Historically every scope ran a single
# 60X objective, so MFX/ST2 (2304-px sensors) image at 0.0878 µm/px and the
# MF-series (MF2-MF5, 2048-px) at 0.108 µm/px at 60X. ST2 additionally
# supports a 40X objective: 0.1317 µm/px = 0.0878 * (60/40) -- a SCALED
# placeholder (magnification ratio applied to the 60X-calibrated value), not
# yet a real calibration measurement; replace with a measured value once one
# is available (see e.g. `misc/MF2_60XSil1.3_zcorrection.ipynb` for how a
# real per-objective calibration is done). Together with the sensor size
# this fixes the FOV footprint (fov_size_um = pixel_size_um * image_size_px),
# used to lay out the scanning grid in before_imaging/02.
_OBJECTIVE_PIXEL_SIZE_UM: Dict[Tuple[str, str], float] = {
    ("MF2", "60X"): 0.108, ("MF3", "60X"): 0.108, ("MF4", "60X"): 0.108, ("MF5", "60X"): 0.108,
    ("MFX", "60X"): 0.0878,
    ("ST2", "60X"): 0.0878,
    ("ST2", "40X"): 0.0878 * 60.0 / 40.0,   # 0.1317 -- scaled placeholder, see comment above
}
# Which objective each microscope uses when the caller doesn't name one --
# keeps every existing single-objective-per-scope call site working
# unchanged. Extend this + _OBJECTIVE_PIXEL_SIZE_UM together when a
# microscope gains a new objective.
_DEFAULT_OBJECTIVE: Dict[str, str] = {
    "MF2": "60X", "MF3": "60X", "MF4": "60X", "MF5": "60X", "MFX": "60X", "ST2": "60X",
}
_DEFAULT_CAMERA_PIXEL_SIZE_UM = 0.108


class FOVGeometry(NamedTuple):
    """FOV geometry for a microscope+objective: sample-plane pixel size and sensor size."""
    pixel_size_um: float   # µm per camera pixel projected onto the sample
    image_size_px: int     # camera sensor size in pixels (square)


def get_camera_pixel_size_um(microscope: Optional[str], objective: Optional[str] = None) -> float:
    """
    Return the sample-plane pixel size (µm/pixel) for *microscope* + *objective*.

    *objective* (e.g. ``"60X"``, ``"40X"``) defaults to that microscope's
    entry in ``_DEFAULT_OBJECTIVE`` (today, every scope has exactly one) --
    omit it to keep prior single-objective-per-scope behaviour unchanged.
    MFX/ST2 → 0.0878 µm/px at 60X, MF-series (MF2–MF5) → 0.108 µm/px at 60X;
    ST2 also has a 40X objective. Unknown microscope/objective falls back to
    0.108 (no error, so estimates still run) — extend
    ``_OBJECTIVE_PIXEL_SIZE_UM``/``_DEFAULT_OBJECTIVE`` for new scopes/objectives.
    """
    key = str(microscope).strip().upper() if microscope is not None else ""
    obj = str(objective).strip().upper() if objective is not None else _DEFAULT_OBJECTIVE.get(key, "")
    return _OBJECTIVE_PIXEL_SIZE_UM.get((key, obj), _DEFAULT_CAMERA_PIXEL_SIZE_UM)


def get_fov_geometry(microscope: Optional[str], objective: Optional[str] = None) -> FOVGeometry:
    """
    Return the FOV geometry ``(pixel_size_um, image_size_px)`` for
    *microscope* + *objective*.

    Bundles :func:`get_camera_pixel_size_um` and the sensor size from
    :func:`get_camera_frame_size` so ``before_imaging/02`` can derive the
    scanning grid from the microscope (+ optional objective) alone instead
    of hard-coding both numbers:

    * MFX, ST2 (60X) → ``(0.0878 µm/px, 2304 px)``
    * ST2 (40X)       → ``(0.1317 µm/px, 2304 px)``
    * MF2–MF5 (60X)   → ``(0.108 µm/px, 2048 px)``

    *objective* defaults to *microscope*'s own default objective (see
    ``_DEFAULT_OBJECTIVE``) — omit it to keep prior behaviour unchanged.
    Unknown microscope/objective falls back to the MF-series 60X values.

    Parameters
    ----------
    microscope : microscope id, case-insensitive (e.g. ``"MF3"``, ``"mfx"``)
    objective   : objective id, case-insensitive (e.g. ``"60X"``, ``"40X"``);
                  ``None`` uses *microscope*'s default objective

    Returns
    -------
    FOVGeometry : named tuple ``(pixel_size_um, image_size_px)`` — unpacks as a
                  plain tuple, e.g. ``px_um, size_px = get_fov_geometry("ST2", "40X")``
    """
    width, _ = get_camera_frame_size(microscope)   # square sensor → width == height
    return FOVGeometry(pixel_size_um=get_camera_pixel_size_um(microscope, objective),
                       image_size_px=width)


# Acquisition type (imaging modality) per microscope. This is independent of the
# channel-mapping / camera-geometry groupings above — e.g. MF2 shares MF3-MF5's
# channel map and camera but is physically a spinning-disk confocal scope, not
# epifluorescence. MFX, ST2 and MF2 are spinning-disk confocal ("disk"); MF3,
# MF4 and MF5 are epifluorescence ("epi").
_ACQUISITION_TYPE: Dict[str, str] = {
    "MF2": "disk", "MFX": "disk", "ST2": "disk",
    "MF3": "epi",  "MF4": "epi",  "MF5": "epi",
}


def get_acquisition_type(microscope: Optional[str]) -> Optional[str]:
    """
    Return the acquisition type — ``"epi"`` (epifluorescence) or ``"disk"``
    (spinning-disk confocal) — for *microscope*.

    Parameters
    ----------
    microscope : microscope id, case-insensitive (e.g. ``"MF3"``, ``"st2"``);
                 may be ``None``

    Returns
    -------
    str or None
        ``"epi"`` or ``"disk"``, or ``None`` if *microscope* is unrecognised —
        extend ``_ACQUISITION_TYPE`` for new scopes rather than guessing.
    """
    key = str(microscope).strip().upper() if microscope is not None else ""
    return _ACQUISITION_TYPE.get(key)


def _normalise_colour_key(color) -> Optional[int]:
    """Return *color* as an int wavelength for dict lookup, or ``None`` for NaN.

    Colour values move through the code as ints (``750``) and floats (``750.0``);
    normalising both a power-dict key and a frame's colour to ``int`` lets them
    compare equal regardless of which form was used.
    """
    if color is None or (isinstance(color, float) and np.isnan(color)):
        return None
    return int(round(float(color)))


def power_dict_to_channel_list(
    power:         Mapping,
    microscope:    str   = "MF3",
    default_power: float = 1.0,
) -> List[float]:
    """
    Convert a ``{wavelength_nm: power}`` mapping into the per-channel list that
    the HAL config ``<default_power>`` element expects.

    The HAL ``<default_power>`` holds one value per hardware channel, ordered by
    channel index (0, 1, 2, …). Each wavelength maps to a channel via the
    microscope's colour→channel table, so this places each colour's power at the
    right index. Channels with no entry in *power* get *default_power*.

    Parameters
    ----------
    power         : mapping ``{wavelength_nm: power}`` (keys may be int or float)
    microscope    : microscope whose colour→channel map is used
    default_power : value for channels not named in *power*

    Returns
    -------
    list of float
        One power per channel, index = channel number, length = channel count.
    """
    ch_map = get_color_to_channel_dict(microscope)
    # Real (non-blank) channels only: {int wavelength -> channel index}.
    colour_to_channel = {
        _normalise_colour_key(c): int(ch)
        for c, ch in ch_map.items()
        if _normalise_colour_key(c) is not None
    }
    n_channels = max(colour_to_channel.values()) + 1
    channel_power = [float(default_power)] * n_channels
    for color, value in power.items():
        key = _normalise_colour_key(color)
        if key is None or key not in colour_to_channel:
            raise ValueError(
                f"Power specified for wavelength {color!r}, which has no channel "
                f"on microscope '{microscope}'. Known wavelengths: "
                f"{sorted(k for k in colour_to_channel)}"
            )
        channel_power[colour_to_channel[key]] = float(value)
    return channel_power


# ── Frame table ───────────────────────────────────────────────────────────────

def get_frame_table(
    bead_z:        float,
    bead_seq:      List,
    color_seq:     List,
    end_seq:       List,
    z_pos:         np.ndarray,
    microscope:    str   = "MF3",
    scan_mode:     str   = "interleaved",
    z_return_mode: str   = "progressive",
    return_step:   float = 5,
) -> pd.DataFrame:
    """
    Build the frame table describing one imaging round.

    Parameters
    ----------
    bead_z     : z position (µm above coverslip) for fiducial bead frames
    bead_seq   : ordered list of colours (nm) or ``np.nan`` for blank frames,
                 acquired at *bead_z* **before** the z-stack
    color_seq  : colours to acquire across the z-stack
    end_seq    : colours acquired at *bead_z* **after** the z-stack (and after
                 the z return); typically blank frames (``[np.nan, np.nan]``)
    z_pos      : 1-D array of z positions for data frames
    microscope : used to resolve hardware channel indices
    z_return_mode : how the objective travels back to *bead_z* after the z-stack:

                 ``"progressive"`` *(default)* — blank (laser-off) frames step
                 the objective down from the last stack position to *bead_z* in
                 increments of *return_step*, so the Z-nanopositioner descends in
                 controlled steps rather than one large jump. For example, a stack
                 ending at z=25 with ``bead_z=0`` and ``return_step=5`` adds blank
                 frames at z = 20, 15, 10, 5, 0 before *end_seq*.

                 ``"instant"`` — no intermediate frames; the objective returns to
                 *bead_z* in a single jump (only the *end_seq* frames follow the
                 z-stack). This was the previous behaviour.
    return_step : z decrement (µm) between blank frames in ``"progressive"`` mode;
                  ignored when ``z_return_mode="instant"``. Must be > 0.
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
    if z_return_mode not in ("progressive", "instant"):
        raise ValueError(
            f"Unknown z_return_mode {z_return_mode!r}. "
            f"Use 'progressive' or 'instant'."
        )
    if z_return_mode == "progressive" and return_step <= 0:
        raise ValueError(f"return_step must be > 0, got {return_step!r}.")

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

    if z_return_mode == "progressive":
        # Step the objective back toward the coverslip with blank (laser-off)
        # frames, descending from the last stack position to bead_z in
        # increments of return_step. The final blank lands exactly on bead_z.
        last_z = rows[-1]["z"] if rows else bead_z
        z = last_z - return_step
        while z > bead_z + 1e-9:
            rows.append({"color": np.nan, "channel": np.nan, "z": z})
            z -= return_step
        rows.append({"color": np.nan, "channel": np.nan, "z": bead_z})

    for color in end_seq:
        rows.append({"color": color, "channel": ch_map[color], "z": bead_z})

    return pd.DataFrame(rows, columns=["color", "channel", "z"])


def get_transit_frame_table(
    bead_z:  float = 0,
    n_blank: int   = 2,
) -> pd.DataFrame:
    """
    Build the frame table for a **transit** segment: *n_blank* laser-off frames
    at *bead_z*.

    Transit FOVs sit between two tissue boundaries and are visited only to move
    the stage smoothly (no data is collected), so every frame is blank (colour
    and channel ``NaN``) and stays at the bead focus. Written to a shutter file
    this yields no ``<event>`` elements — i.e. *n_blank* dark frames — and the
    HAL ``<frames>`` count is *n_blank*.

    Parameters
    ----------
    bead_z  : z position (µm above coverslip) held for every transit frame
    n_blank : number of blank frames per transit FOV (default 2)

    Returns
    -------
    pd.DataFrame with columns ``["color", "channel", "z"]``.
    """
    if n_blank < 1:
        raise ValueError(f"n_blank must be >= 1, got {n_blank!r}.")
    rows = [{"color": np.nan, "channel": np.nan, "z": bead_z} for _ in range(n_blank)]
    return pd.DataFrame(rows, columns=["color", "channel", "z"])


def get_color_sequence_name(
    frame_table: pd.DataFrame,
    separator:   str = "_",
    scan_mode:   str = "interleaved",
) -> str:
    """
    Build a compact human-readable name for a colour sequence.

    Returns something like ``"blkf2_560f49_650f49"`` where ``blkf2`` means
    two blank (NaN) frames and ``560f49`` means forty-nine 560 nm frames.
    A ``"_seq"`` suffix is appended when *scan_mode* is ``"sequential"`` to
    prevent filename collisions with the interleaved variant.

    The tokens are joined by underscores so the whole colour name reads as a
    single filename field; the hyphen is reserved for the structural prefix of
    config/shutter/frame-table filenames (see :func:`sequence_stem` and the
    ``*_filename`` helpers).

    Parameters
    ----------
    frame_table : DataFrame with a ``"color"`` column
    separator   : string placed between name tokens (default ``"_"``)
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


# ── Filename naming rule ────────────────────────────────────────────────────────
# A single round's three artefacts (HAL config, shutter, frame table) share one
# stem ``{kind}-{name}`` where *kind* is ``bits`` / ``cells`` / ``transit`` /
# ``drift`` and *name* is the underscore-joined colour-sequence name from
# ``get_color_sequence_name`` (e.g. ``blkf5_488f2_560f25_650f25_750f25``). Hyphens
# delimit the structural prefix; underscores live only inside *name*. This keeps
# the analysis-side resolver ``find_frame_table_for_hal_config`` a simple
# ``shutter-`` -> ``frame-table-`` rewrite.

_VALID_KINDS = ("bits", "cells", "transit", "drift", "focustest")


def sequence_stem(kind: str, name: str, tier: Optional[str] = None) -> str:
    """
    Return the shared ``{kind}-{name}`` stem for a round's filenames, or
    ``{kind}-{tier}-{name}`` when *tier* is given.

    Parameters
    ----------
    kind : ``"bits"``, ``"cells"``, ``"transit"``, ``"drift"`` (a stage-
           drift-check round -- see ``notebooks/misc/calculate_stage_drift.ipynb``),
           or ``"focustest"`` (the auto-generated focus-lock-test movie config
           -- see ``04_create_dave_config.ipynb``'s "Focus-lock test recipe"
           section)
    name : colour-sequence name from :func:`get_color_sequence_name`
           (underscore-joined, e.g. ``"blkf5_488f2_560f25_650f25_750f25"``)
    tier : optional z-depth-tier label (e.g. ``"shallow"``, ``"deep"``) for a
           variable-z-per-FOV experiment's per-tier hal_config/shutter/
           frame-table set -- see notebook 04 in
           ``before_imaging/multi_z/``. ``None`` (default)
           reproduces the original ``{kind}-{name}`` stem exactly. Not
           strictly required for uniqueness (a shorter tier's frame table
           already yields a different colour-sequence *name*, since it has
           fewer frames per colour), but makes the files legible to a human
           picking the right one to load into HAL.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}.")
    return f"{kind}-{tier}-{name}" if tier else f"{kind}-{name}"


def hal_config_filename(microscope: str, kind: str, name: str, tier: Optional[str] = None) -> str:
    """HAL config filename, e.g. ``hal-config-mf3-bits-blkf5_488f2_….xml``."""
    return f"hal-config-{microscope.lower()}-{sequence_stem(kind, name, tier=tier)}.xml"


def shutter_filename(kind: str, name: str, tier: Optional[str] = None) -> str:
    """Shutter filename, e.g. ``shutter-bits-blkf5_488f2_….xml``."""
    return f"shutter-{sequence_stem(kind, name, tier=tier)}.xml"


def frame_table_filename(kind: str, name: str, tier: Optional[str] = None) -> str:
    """Frame-table filename, e.g. ``frame-table-bits-blkf5_488f2_….csv``."""
    return f"frame-table-{sequence_stem(kind, name, tier=tier)}.csv"


# ── Shutter XML ───────────────────────────────────────────────────────────────

def create_shutter_file(
    frame_table:   pd.DataFrame,
    output_path:   Path,
    oversampling:  int   = 1,
    default_power: float = 1.0,
) -> None:
    """
    Write a HAL shutter XML file from *frame_table*.

    Every ``<event>``'s ``<power>`` is *default_power* (1.000 by default),
    regardless of frame colour. This value is a full-modulation flag relative
    to whatever the HAL config's own ``<default_power>`` sets for that
    channel (see :func:`power_dict_to_channel_list`), not an independent
    absolute laser power -- the two are not equivalent, and setting the same
    real per-colour intensity in both places silently double-applies the
    intensity scaling on real hardware. Per-colour power belongs ONLY in the
    HAL config's ``<default_power>``.

    Parameters
    ----------
    frame_table   : DataFrame produced by :func:`get_frame_table`
    output_path   : destination path; written with Windows (CRLF) line endings
    oversampling  : value for the ``<oversampling>`` XML element
    default_power : laser power written for every ``<event>`` (see above)
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
        ET.SubElement(event, "power").text   = f"{float(default_power):.3f}"
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


def read_hal_exposure_time(hal_config_path: Path) -> "Optional[float]":
    """
    Return the ``<exposure_time>`` value (seconds) from the HAL config at
    *hal_config_path*, or ``None`` on any parse error or missing element.
    """
    try:
        with open(hal_config_path, "rb") as fh:
            text = fh.read().decode("ISO-8859-1")
        m = re.search(r"<exposure_time[^>]*>([\d.]+)</exposure_time>", text)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def find_frame_table_for_hal_config(
    hal_config_path: Path,
    metadata_dir:    Path,
) -> "Optional[Path]":
    """
    Locate the frame-table CSV that corresponds to *hal_config_path*.

    Strategy: read the ``<shutters>`` element from the HAL config
    (``shutter-{stem}.xml``), strip the ``shutter-`` prefix to recover the
    ``{kind}-{name}`` stem, and return ``metadata_dir/frame-table-{stem}.csv``
    if it exists. (Legacy ``frame_table_{stem}.csv`` is accepted as a fallback.)

    Returns ``None`` when the frame table cannot be found.
    """
    try:
        with open(hal_config_path, "rb") as fh:
            text = fh.read().decode("ISO-8859-1")
        m = re.search(r"<shutters[^>]*>([^<]+)</shutters>", text)
        if not m:
            return None
        shutter_stem = Path(m.group(1).strip()).stem  # shutter-{stem}
        if shutter_stem.startswith("shutter-"):
            stem = shutter_stem[len("shutter-"):]
        else:
            stem = shutter_stem
        candidates = [
            Path(metadata_dir) / f"frame-table-{stem}.csv",   # current convention
            Path(metadata_dir) / f"frame_table_{stem}.csv",   # legacy fallback
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
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


def get_all_color_frame_indices(
    frame_table: pd.DataFrame,
    color:       float,
    bead_z:      float = 0.0,
) -> List[int]:
    """
    Return every frame index for *color* in *frame_table* (every z-plane),
    not just the single middle-z one :func:`get_color_frame_indices` returns.

    Used by flat-field-correction sample selection when treating a single
    FOV's own full z-stack as multiple independent illumination samples of
    the same channel (vignetting is a property of the optics, not of z).

    Parameters
    ----------
    frame_table : DataFrame with columns ``["color", "channel", "z"]`` and
                  integer index equal to the camera frame number
    color       : excitation wavelength (nm) to match
    bead_z      : z value used for fiducial (bead) frames; those rows are
                  excluded

    Returns
    -------
    Frame indices for *color*, in ascending z order.
    """
    df = frame_table.copy()
    df["frame_idx"] = df.index

    data = df[(df["z"] != bead_z) & (df["color"] == color)].copy()
    if data.empty:
        return []

    return data.sort_values("z")["frame_idx"].astype(int).tolist()


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


# ── Mosaic-helper configs ─────────────────────────────────────────────────────

def find_mosaic_helper_configs(microscope: str, mosaic_helper_dir: Path) -> List[Path]:
    """
    Find the hand-crafted low/high-mag mosaic setup configs (HAL + shutter
    XML, e.g. ``hal-config-mf3-10x-mosaic-405.xml``) for *microscope* in
    *mosaic_helper_dir* (matched case-insensitively on the microscope name in
    the filename). These are not generated by MERci -- they're pre-made
    configs used to run the Steve mosaic tool at a given magnification --
    so not every microscope has them; returns an empty list rather than
    raising when none are found.

    Parameters
    ----------
    microscope        : microscope identifier, e.g. "MF3"
    mosaic_helper_dir : directory of mosaic-helper config files (e.g.
                        ``MERci/data/configs/hal/mosaic_helper/``)

    Returns
    -------
    Sorted list of matching file paths (empty if none exist for this microscope).
    """
    return sorted(
        p for p in Path(mosaic_helper_dir).glob("*.xml")
        if microscope.lower() in p.name.lower()
    )


def copy_mosaic_helper_configs(
    microscope:        str,
    mosaic_helper_dir: Path,
    settings_dir:      Path,
) -> List[Path]:
    """
    Copy every mosaic-helper config file for *microscope* (see
    :func:`find_mosaic_helper_configs`) into *settings_dir*, so the imaging
    setup files for this microscope's mosaic tool land alongside the
    per-round HAL configs/shutters this notebook already writes there.

    Returns the list of destination paths written (empty, not an error, if
    this microscope has no mosaic-helper configs available).
    """
    import shutil
    dests = []
    for src in find_mosaic_helper_configs(microscope, mosaic_helper_dir):
        dest = Path(settings_dir) / src.name
        shutil.copy2(src, dest)
        dests.append(dest)
    return dests


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