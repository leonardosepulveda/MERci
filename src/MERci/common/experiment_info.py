# MERci/common/experiment_info.py
"""
Per-experiment metadata: a small, human-readable YAML file living in each
experiment's ``SAMPLE_DIR/metadata/``, and a batch collector that reads many
of them back into one table shaped like a project's master experiment-info
CSV (e.g. ``lt_experiment_info.csv``).

Why YAML instead of the master CSV's own row-per-experiment shape: a single
31-column CSV row is hard to read/edit by hand, and doesn't nest well. A flat
YAML mapping is both a natural single-record view and a native Python dict.

Why core fields + an ``extra`` dict instead of one big fixed schema: the
three per-project master CSVs (``bc_``/``lt_``/``mf_experiment_info.csv``)
share most columns but differ in a few (``lt`` has ``positions_path``/
``positions_name``; ``mf`` has a single ``positions_file`` instead) — the
same "columns vary slightly by source" problem ``SeriesInfo.extra_meta``
already solves in ``common/metadata.py`` for round_info.csv columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

# Columns handled by ExperimentInfo's named fields; everything else read from
# a CSV row (or passed to `extra`) is a project-specific straggler.
_CORE_FIELDS = (
    "sample_name", "project", "microscope", "lib_name",
    "data_home", "merlin_home", "folder_name",
)


@dataclass
class ExperimentInfo:
    """
    One experiment's metadata — the fields common to the bc/lt/mf master
    CSVs, plus an ``extra`` dict for every project-specific column.

    Attributes
    ----------
    sample_name : experiment/sample id, e.g. ``"LT048_sample_26"``
    project     : which master-CSV schema this maps to — ``"bc"``, ``"lt"``,
                  or ``"mf"``
    microscope  : microscope id, e.g. ``"MF3"``
    lib_name    : codebook/probe-library name, e.g. ``"LT2"``
    data_home   : cluster root directory holding the raw data
    merlin_home : cluster root directory for MERlin's analysis output
    folder_name : path (relative to ``data_home``) to this experiment's data
    extra       : every other master-CSV column (e.g. ``exposure``,
                  ``hyb_temp``, ``fix_type``, ``positions_path``,
                  ``positions_name`` / ``positions_file``, ...)
    """

    sample_name: str
    project:     str
    microscope:  str
    lib_name:    str
    data_home:   str
    merlin_home: str
    folder_name: str
    extra: Dict[str, Any] = field(default_factory=dict)


def save_experiment_info(info: ExperimentInfo, path: Path) -> None:
    """
    Write *info* to *path* as a flat YAML mapping (core fields and ``extra``
    entries side by side — the file on disk reads like one flat record, not
    two visually separated blocks).
    """
    flat = {f: getattr(info, f) for f in _CORE_FIELDS}
    flat.update(info.extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(flat, fh, sort_keys=False, default_flow_style=False)


def load_experiment_info(path: Path) -> ExperimentInfo:
    """Read an ``experiment_info.yaml`` written by :func:`save_experiment_info`."""
    with open(path, "r", encoding="utf-8") as fh:
        flat = yaml.safe_load(fh) or {}
    core = {f: flat.get(f, "") for f in _CORE_FIELDS}
    extra = {k: v for k, v in flat.items() if k not in _CORE_FIELDS}
    return ExperimentInfo(**core, extra=extra)


def collect_experiment_info(paths: Sequence[Path]) -> pd.DataFrame:
    """
    Read many ``experiment_info.yaml`` files and combine them into one
    DataFrame shaped like a project's master experiment-info CSV — one row
    per experiment, columns outer-joined (so an experiment missing a
    project-specific column just gets ``NaN`` there, rather than the whole
    batch being dropped to the smallest common column set).

    Ready to append to (or replace rows in) the real master CSV by hand.
    """
    rows: List[Dict[str, Any]] = []
    for p in paths:
        info = load_experiment_info(Path(p))
        row = {f: getattr(info, f) for f in _CORE_FIELDS}
        row.update(info.extra)
        rows.append(row)
    return pd.DataFrame(rows)
