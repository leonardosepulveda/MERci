# merfish_pipeline/acquisition/display.py
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


def print_frame_table(frame_table: pd.DataFrame) -> None:
    """
    Print *frame_table* in a compact, aligned format for quick inspection.

    Each line shows one group of frames that share the same z position,
    with columns for frame index, colour (wavelength), channel, and z.

    Parameters
    ----------
    frame_table : DataFrame with columns ``["color", "channel", "z"]``
                  and an integer index equal to the frame number
    """
    counts       = frame_table["z"].value_counts()
    frames_per_z = int(counts.min())
    col_w        = 6
    sep          = " " * 6
    hdr_w        = col_w * frames_per_z

    print(
        f'{"frames":{hdr_w}s}{sep}'
        f'{"color":{hdr_w}s}{sep}'
        f'{"channel":{hdr_w}s}{sep}'
        f'{"z":{hdr_w}s}'
    )
    print()

    n = len(frame_table)

    def _fmt_int(val) -> str:
        return f'{"nan":>{col_w}}' if pd.isna(val) else f"{int(val):{col_w}d}"

    for start in range(0, n, frames_per_z):
        group = frame_table.iloc[start : start + frames_per_z]
        if len(group) < frames_per_z:
            break

        frames_str   = "".join(f"{int(idx):{col_w}d}" for idx in group.index)
        colors_str   = "".join(_fmt_int(v)               for v in group["color"])
        channels_str = "".join(_fmt_int(v)               for v in group["channel"])
        z_str        = "".join(f"{float(v):{col_w}.2f}"  for v in group["z"])

        print(f"{frames_str}{sep}{colors_str}{sep}{channels_str}{sep}{z_str}")


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