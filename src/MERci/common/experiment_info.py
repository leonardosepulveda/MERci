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
    "sample_name", "project", "microscope", "acquisition_type", "lib_name",
    "data_home", "merlin_home", "folder_name",
)


@dataclass
class ExperimentInfo:
    """
    One experiment's metadata — the fields common to the bc/lt/mf master
    CSVs, plus an ``extra`` dict for every project-specific column.

    Attributes
    ----------
    sample_name      : experiment/sample id, e.g. ``"LT048_sample_26"``
    project          : which master-CSV schema this maps to — ``"bc"``, ``"lt"``,
                       or ``"mf"``
    microscope       : microscope id, e.g. ``"MF3"``
    acquisition_type : imaging modality derived from ``microscope`` —
                       ``"epi"`` (epifluorescence) or ``"disk"`` (spinning-disk
                       confocal); see
                       :func:`MERci.acquisition.configs.get_acquisition_type`
    lib_name         : codebook/probe-library name, e.g. ``"LT2"``
    data_home        : cluster root directory holding the raw data
    merlin_home      : cluster root directory for MERlin's analysis output
    folder_name      : path (relative to ``data_home``) to this experiment's data
    extra            : every other master-CSV column (e.g. ``exposure``,
                       ``hyb_temp``, ``fix_type``, ``positions_path``,
                       ``positions_name`` / ``positions_file``, ...)
    """

    sample_name:      str
    project:          str
    microscope:       str
    acquisition_type: str
    lib_name:         str
    data_home:        str
    merlin_home:      str
    folder_name:      str
    extra: Dict[str, Any] = field(default_factory=dict)


# Acquisition-type subfolder names already used throughout the codebase (see
# CLAUDE.md's tumor/{epi,disk} and lineage_tracing/{merfish,lineage} variants)
# -- the fixed vocabulary resolve_sample_identity checks MERCI_DIR's parent
# folder name against to detect the split-subfolder layout.
_ACQUISITION_SUBFOLDER_TOKENS = {"merfish", "lineage", "epi", "disk"}


def resolve_sample_identity(merci_dir: Path) -> tuple[str, str]:
    """
    Determine the true experiment id and acquisition-subfolder name from
    where this MERci clone lives on disk.

    Two layouts are possible for ``SAMPLE_DIR = merci_dir.parent``:

    * **Flat** (default): ``SAMPLE_DIR`` itself is the experiment folder, e.g.
      ``.../251225_LT027_saving_time/MERci``. ``SAMPLE_DIR.name`` IS the
      experiment id; there is no acquisition subfolder.
    * **Split**: this acquisition lives under its own acquisition-type
      subfolder, sibling to another acquisition of the same sample, e.g.
      ``.../LT058_sample_07/merfish/MERci`` (with ``.../LT058_sample_07/lineage/``
      alongside it). ``SAMPLE_DIR.name`` (``"merfish"``) is just this
      acquisition's own local file-naming tag, NOT the experiment id -- the
      true id is one level further up.

    Distinguishing the two from folder structure alone, without depending on
    the experiment id following any particular naming convention (older
    experiments are date-prefixed, e.g. ``"251225_LT027_saving_time"``, which
    doesn't match a newer ``"LT058_sample_07"``-style pattern at all): a
    split layout's acquisition subfolder name is always one of a small, fixed
    vocabulary already hard-coded throughout the codebase for exactly this
    purpose (``_ACQUISITION_SUBFOLDER_TOKENS``) -- the notebook variant
    itself is duplicated per acquisition type, so no other subfolder name is
    ever a real possibility here. If ``SAMPLE_DIR.name`` is one of those
    tokens, treat it as the split layout; otherwise, flat.

    This does NOT change most per-notebook local file naming (``dave-{mic}-
    {N}hybs-{name}.xml``, data-organization/merlin/fishtank script filenames,
    etc. all still use the bare ``sample_name`` returned here). The one
    exception is ``positions_*.txt``/``fov_layout_*.png`` filenames, which use
    :func:`positions_file_tag` instead -- see that function for why. Use this
    function where the TRUE top-level experiment id is needed: constructing
    cluster-facing paths (``DATA_HOME``/``MERLIN_HOME``/``FOLDER_NAME`` in
    notebook 06, ``resolve_cluster_sample_dir`` in notebook 07).

    Parameters
    ----------
    merci_dir : path to this MERci clone (``MERCI_DIR`` in every notebook)

    Returns
    -------
    (sample_name, imaging_dir) : the true experiment id, and the acquisition-
    type subfolder name (``""`` in the flat layout).
    """
    sample_dir = Path(merci_dir).parent
    if sample_dir.name.lower() in _ACQUISITION_SUBFOLDER_TOKENS:
        sample_name, imaging_dir = sample_dir.parent.name, sample_dir.name
    else:
        sample_name, imaging_dir = sample_dir.name, ""
    if not sample_name:
        # Only possible if the split layout's acquisition subfolder sits at
        # a drive root (no real experiment folder above it) -- degenerate,
        # but fail loudly rather than silently writing an empty experiment id.
        raise ValueError(
            f"Could not determine a non-empty sample_name from {merci_dir!r} "
            f"(resolved imaging_dir={imaging_dir!r}). Check MERCI_DIR's location."
        )
    return sample_name, imaging_dir


def resolve_data_home(merci_dir: Path) -> str:
    """
    Infer ``DATA_HOME`` (notebook 06's cluster root directory holding the
    raw data) from where this MERci clone actually lives on disk, instead of
    a hardcoded path.

    ``DATA_HOME`` is the directory that directly contains the true
    ``sample_name`` folder from :func:`resolve_sample_identity` -- one level
    up from ``SAMPLE_DIR`` in the flat layout, two levels up in the split
    layout (``SAMPLE_DIR`` itself is the acquisition-type subfolder there).

    Parameters
    ----------
    merci_dir : path to this MERci clone (``MERCI_DIR`` in every notebook)

    Returns
    -------
    str : ``DATA_HOME``, as an absolute path string.
    """
    _, imaging_dir = resolve_sample_identity(merci_dir)
    sample_dir = Path(merci_dir).parent
    data_home  = sample_dir.parent.parent if imaging_dir else sample_dir.parent
    return str(data_home)


def positions_file_tag(sample_name: str, imaging_dir: str) -> str:
    """
    Token for ``positions_*.txt``/``fov_layout_*.png`` filenames: ``sample_name``
    alone in the flat layout, or ``"{sample_name}_{imaging_dir}"`` in the split
    layout.

    Two sibling split-layout acquisitions of the same sample (e.g.
    ``tumor/epi`` + ``tumor/disk``, or ``lineage_tracing/merfish`` +
    ``lineage_tracing/lineage``) resolve to the same ``sample_name`` via
    :func:`resolve_sample_identity` -- without this tag, each acquisition's
    notebook 02 would write an identically-named ``positions_{sample_name}.txt``
    in its own ``positions/`` folder, which only avoids colliding because the
    two folders happen to be different.

    Parameters
    ----------
    sample_name : the true experiment id (``resolve_sample_identity``'s first
        return value)
    imaging_dir : the acquisition-type subfolder name, or ``""`` in the flat
        layout (``resolve_sample_identity``'s second return value)

    Returns
    -------
    str : the token to substitute for ``sample_name`` in a ``positions_*.txt``/
    ``fov_layout_*.png`` filename.
    """
    return f"{sample_name}_{imaging_dir}" if imaging_dir else sample_name


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
