# MERci/acquisition/display.py
"""
Jupyter display helpers for acquisition setup notebooks.

Note: this module was previously named ``print.py``, which shadows
the Python built-in.
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display


# ── Scan-mode inference ───────────────────────────────────────────────────────

def _infer_scan_mode(frame_table: pd.DataFrame) -> str:
    """
    Return ``'interleaved'`` or ``'sequential'`` by inspecting the first two
    consecutive data frames (those not at the bead z-position).

    - If the first two data frames share the same z → color is the fast axis
      → ``'interleaved'``
    - If they share the same color (or both blank) → z is the fast axis
      → ``'sequential'``
    """
    bead_z = frame_table["z"].iloc[0]
    data   = frame_table[frame_table["z"] != bead_z]
    if len(data) < 2:
        return "interleaved"
    if data["z"].iloc[0] == data["z"].iloc[1]:
        return "interleaved"
    c0, c1     = data["color"].iloc[0], data["color"].iloc[1]
    same_color = (c0 == c1) or (pd.isna(c0) and pd.isna(c1))
    return "sequential" if same_color else "interleaved"


def _split_groups(frame_table: pd.DataFrame, scan_mode: str):
    """
    Split *frame_table* into a list of consecutive-row groups.

    - ``'interleaved'``: new group whenever z changes.
    - ``'sequential'``: new group whenever color changes (NaN == NaN).
    """
    groups: list = []
    start  = 0
    n      = len(frame_table)
    for i in range(1, n):
        prev_row = frame_table.iloc[i - 1]
        curr_row = frame_table.iloc[i]
        if scan_mode == "interleaved":
            split = curr_row["z"] != prev_row["z"]
        else:
            pc, cc = prev_row["color"], curr_row["color"]
            same   = (pc == cc) if not (pd.isna(pc) or pd.isna(cc)) \
                     else (pd.isna(pc) and pd.isna(cc))
            split  = not same
        if split:
            groups.append(frame_table.iloc[start:i])
            start = i
    groups.append(frame_table.iloc[start:])
    return groups


# ── Frame table printer ───────────────────────────────────────────────────────

def print_frame_table(frame_table: pd.DataFrame) -> None:
    """
    Print *frame_table* in a compact, aligned format for quick inspection.

    The scan mode is inferred automatically:

    **Interleaved** (all colors at each Z-nanopositioner position): one row
    per z-plane.  All rows are padded to the width of the widest group so that
    columns stay aligned even when bead/end sequences are shorter than
    ``color_seq``.

    **Sequential** (full z-sweep per color, boustrophedon): one summary row
    per color block showing the frame range, color, channel, z-range, and
    sweep direction.

    Parameters
    ----------
    frame_table : DataFrame with columns ``["color", "channel", "z"]``
                  and an integer index equal to the frame number
    """
    scan_mode = _infer_scan_mode(frame_table)
    groups    = _split_groups(frame_table, scan_mode)
    col_w     = 6
    sep       = " " * 6

    def _fmt_int(val) -> str:
        return f'{"nan":>{col_w}}' if pd.isna(val) else f"{int(val):{col_w}d}"

    # ── Interleaved: one row per z-plane, columns padded to max group width ───
    if scan_mode == "interleaved":
        max_g = max(len(g) for g in groups)
        hdr_w = col_w * max_g

        print(
            f'{"frames":{hdr_w}s}{sep}'
            f'{"color":{hdr_w}s}{sep}'
            f'{"channel":{hdr_w}s}{sep}'
            f'{"z":{col_w}s}'
        )
        print()

        for group in groups:
            n   = len(group)
            pad = " " * (col_w * (max_g - n))
            frames_str   = "".join(f"{int(idx):{col_w}d}" for idx in group.index) + pad
            colors_str   = "".join(_fmt_int(v) for v in group["color"])    + pad
            channels_str = "".join(_fmt_int(v) for v in group["channel"])  + pad
            z_val        = group["z"].iloc[0]
            print(f"{frames_str}{sep}{colors_str}{sep}{channels_str}{sep}{z_val:{col_w}.2f}")

    # ── Sequential: one summary row per color sweep ────────────────────────
    else:
        print(f'{"frames":15s}  {"color":8s}  {"ch":4s}  z-range')
        print()

        for group in groups:
            n           = len(group)
            first_frame = int(group.index[0])
            last_frame  = int(group.index[-1])
            color       = group["color"].iloc[0]
            channel     = group["channel"].iloc[0]
            z_first     = group["z"].iloc[0]
            z_last      = group["z"].iloc[-1]

            frame_str   = (f"{first_frame}"
                           if n == 1 else f"{first_frame} - {last_frame}")
            color_str   = "nan" if pd.isna(color)   else str(int(color))
            channel_str = "nan" if pd.isna(channel)  else str(int(channel))

            if z_first == z_last:
                # Bead, end, or any single-z group
                print(f"{frame_str:15s}  {color_str:8s}  {channel_str:4s}  {z_first:.2f}")
            else:
                direction = "^" if z_last > z_first else "v"
                print(
                    f"{frame_str:15s}  {color_str:8s}  {channel_str:4s}"
                    f"  {z_first:.2f} -> {z_last:.2f}"
                    f"  {direction}  ({n} frames)"
                )


def display_xml(path: Path, encoding: str = "ISO-8859-1") -> None:
    """
    Render an XML file as a collapsible code block in a Jupyter notebook.

    Parameters
    ----------
    path     : path to the XML file
    encoding : file encoding (default ``"ISO-8859-1"``)
    """
    with open(path, "rb") as fh:
        text = fh.read().decode(encoding)

    title         = os.path.basename(path)
    escaped_title = html.escape(title)
    escaped_text  = html.escape(text)

    display(HTML(f"""
    <details>
      <summary><b>{escaped_title}</b></summary>
      <pre style="background:#f8f8f8;padding:8px;border-radius:4px;">{escaped_text}</pre>
    </details>
    """))
