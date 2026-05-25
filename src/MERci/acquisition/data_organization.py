# MERci/acquisition/data_organization.py
"""
Build the MERlin ``data_organization_*.csv`` from a frame table and a
round–bit–color mapping.

The data-organization file tells MERlin where to find each bit's images:
which files to open, which frames correspond to each z-slice, and which
frame carries the fiducial (bead) reference.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import pandas as pd


# ── Public API ─────────────────────────────────────────────────────────────────

def create_data_organization(
    bits_frame_table:  pd.DataFrame,
    cells_frame_table: pd.DataFrame,
    round_bit_color:   List[Tuple[int, int, int]],
    readouts:          pd.DataFrame,
    bits_series:       str,
    cells_series:      str,
    include_dapi:      bool = True,
    dapi_bit_number:   int  = 47,
) -> pd.DataFrame:
    """
    Build the MERlin data-organization DataFrame.

    Parameters
    ----------
    bits_frame_table  : frame table for bits rounds (from ``metadata/frame_table_*.csv``)
    cells_frame_table : frame table for the cells round
    round_bit_color   : list of ``(round_1indexed, bit_number, color_nm)`` tuples
    readouts          : readouts.csv DataFrame; must have columns
                        ``"Bit number"`` and ``"Name"`` (e.g. ``"b1-RS0015"``)
    bits_series       : series pattern for bits, e.g. ``"hal-mf3-epi_01_{fov:03d}"``
    cells_series      : series pattern for cells, e.g. ``"hal-mf3-epi_cells_{fov:03d}"``
    include_dapi      : whether to append a DAPI row from the cells frame table
    dapi_bit_number   : bit number assigned to DAPI (default 47)

    Returns
    -------
    pd.DataFrame with the 14 MERlin data-organization columns.
    """
    bits_image_type  = _series_to_image_type(bits_series)
    bits_regexp      = _series_to_regexp(bits_series)
    cells_image_type = _series_to_image_type(cells_series)
    cells_regexp     = _series_to_regexp(cells_series)
    fid_frame_bits   = _fiducial_frame(bits_frame_table)
    fid_frame_cells  = _fiducial_frame(cells_frame_table)

    readout_name_map = dict(
        zip(readouts["Bit number"].astype(int), readouts["Name"].astype(str))
    )

    rows: list[dict] = []

    for round_1idx, bit, color_nm in round_bit_color:
        rows.append({
            "readoutName":         readout_name_map[bit],
            "channelName":         f"bit{bit:02d}",
            "imageType":           bits_image_type,
            "imageRegExp":         bits_regexp,
            "bitNumber":           bit,
            "imagingRound":        round_1idx,
            "color":               color_nm,
            "frame":               _frames_for_color(bits_frame_table, color_nm),
            "zPos":                _zpos_for_color(bits_frame_table, color_nm),
            "fiducialImageType":   bits_image_type,
            "fiducialRegExp":      bits_regexp,
            "fiducialImagingRound": round_1idx,
            "fiducialFrame":       fid_frame_bits,
            "fiducialColor":       488,
        })

    if include_dapi:
        rows.append({
            "readoutName":         "DAPI",
            "channelName":         "DAPI",
            "imageType":           cells_image_type,
            "imageRegExp":         cells_regexp,
            "bitNumber":           dapi_bit_number,
            "imagingRound":        -1,
            "color":               405,
            "frame":               _frames_for_color(cells_frame_table, 405),
            "zPos":                _zpos_for_color(cells_frame_table, 405),
            "fiducialImageType":   cells_image_type,
            "fiducialRegExp":      cells_regexp,
            "fiducialImagingRound": -1,
            "fiducialFrame":       fid_frame_cells,
            "fiducialColor":       488,
        })

    cols = [
        "readoutName", "channelName", "imageType", "imageRegExp",
        "bitNumber", "imagingRound", "color", "frame", "zPos",
        "fiducialImageType", "fiducialRegExp", "fiducialImagingRound",
        "fiducialFrame", "fiducialColor",
    ]
    return pd.DataFrame(rows, columns=cols)


# ── Internal helpers ────────────────────────────────────────────────────────────

def _frames_for_color(ft: pd.DataFrame, color_nm: int) -> list:
    """Row indices (= frame numbers) where ``color`` matches *color_nm*."""
    return ft.index[ft["color"] == color_nm].tolist()


def _zpos_for_color(ft: pd.DataFrame, color_nm: int) -> list:
    """Sorted z values for the rows belonging to *color_nm*."""
    return sorted(ft.loc[ft["color"] == color_nm, "z"].tolist())


def _fiducial_frame(ft: pd.DataFrame) -> int:
    """Index of the first 488-nm bead frame (z == 0)."""
    mask = (ft["color"] == 488) & (ft["z"] == 0)
    idx  = ft.index[mask]
    if len(idx) == 0:
        raise ValueError("No 488-nm bead frame (z==0) found in frame table.")
    return int(idx[0])


def _series_to_image_type(series: str) -> str:
    """
    Extract the MERlin ``imageType`` from a series pattern.

    Strips the ``_{fov:…}`` suffix, then strips a trailing ``_\\d{2}``
    round-number segment if present.

    Examples
    --------
    ``"hal-mf3-epi_01_{fov:03d}"``   → ``"hal-mf3-epi"``
    ``"hal-mf3-epi_cells_{fov:03d}"`` → ``"hal-mf3-epi_cells"``
    """
    s = re.sub(r"_\{[^}]+\}$", "", series)   # strip _{fov:03d}
    s = re.sub(r"_\d{2}$", "", s)             # strip _01, _02, … if present
    return s


def _series_to_regexp(series: str) -> str:
    """
    Build the MERlin ``imageRegExp`` from a series pattern.

    If the series base (after stripping the ``_{fov:…}`` placeholder) still
    has a trailing ``_\\d+`` round segment, the regexp captures
    ``imagingRound`` before ``fov``.  Otherwise (e.g. cells) only ``fov``
    is captured.

    Examples
    --------
    ``"hal-mf3-epi_01_{fov:03d}"``   →
        ``(?P<imageType>[\\w|-]+)_(?P<imagingRound>[\\w|-]+)_(?P<fov>[0-9]+)``
    ``"hal-mf3-epi_cells_{fov:03d}"`` →
        ``(?P<imageType>[\\w|-]+)_(?P<fov>[0-9]+)``
    """
    base = re.sub(r"_\{[^}]+\}$", "", series)   # strip _{fov:03d}
    if re.search(r"_\d+$", base):
        return r"(?P<imageType>[\w|-]+)_(?P<imagingRound>[\w|-]+)_(?P<fov>[0-9]+)"
    return r"(?P<imageType>[\w|-]+)_(?P<fov>[0-9]+)"
