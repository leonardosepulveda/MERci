# MERci/acquisition/merlin_config.py
"""
Generate MERlin's input/config files for one experiment, writing into a
per-experiment ``SAMPLE_DIR/merlin/`` folder instead of the shared cluster
location (``~/Software/merfish-parameters/``) used before.

Every schema in this module was verified against real, current (2026)
template/output files read directly from ``R:\\Software\\merfish-parameters\\``
(microscope-parameter JSONs, codebook CSVs, the cluster-resource-allocation
template, a real generated snakemake-parameters file, and a real, currently
live slurm submit script) — nothing here is a guess.

Functions
---------
create_microscope_parameters_json — MERlin's per-scope calibration JSON
create_codebook_csv               — MERlin's gene/barcode codebook CSV
create_cluster_resource_allocation — per-task slurm resource overrides
create_snakemake_parameters       — snakemake's top-level parameters JSON
short_experiment_name             — (sample_name, imaging_dir) -> short,
    SLURM-job-name-safe prefix (job_name_prefix for create_snakemake_parameters)
resolve_cluster_sample_dir        — this experiment's acquisition root as
    addressed from the Linux cluster, whether generated on Windows (predicted
    from the sample name) or on the cluster itself (its own real path)
create_slurm_submit_script        — the sbatch script that runs ``merlin``,
    with every path relative to one ``$SAMPLE_DIR`` bash variable
resolve_codebook_filename         — lib_name -> codebook filename (dispatch only)
resolve_microscope_parameters_filename — microscope id -> params filename (dispatch only)
load_microscope_orientation      — read a microscope's flip_horizontal/flip_vertical/
    transpose flags (MERlin's own defaults when absent, confirmed against
    merlin.core.dataset.py, not assumed)
apply_microscope_orientation     — apply those flags to a raw frame in MERlin's own
    order (transpose, then flip_horizontal, then flip_vertical)
MerlinAnalysisSpec / create_merlin_analysis_parameters — build MERlin's
    warp/optimize/decode/segment task-parameters JSON from a compact spec
    (which steps to include), instead of copying and hand-editing a prior
    experiment's file.
build_merlin_analysis_parameters — the notebooks' DEFAULT analysis-JSON
    builder: assembles the same task-parameters JSON from atomic per-task
    YAML files (data/configs/merlin/analysis/tasks/) plus an explicit
    ordered recipe YAML (data/configs/merlin/analysis/recipes/) naming which
    tasks to include, instead of MerlinAnalysisSpec's Python dataclass/
    booleans -- see that function's own docstring and the module comment
    above it for the recipe/atom format and how its defaults were chosen.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ── Filename dispatch (ports of the old notebook's hardcoded if/elif chains) ───
# Dispatch-only: never touches file contents, just picks a filename, so there
# is no schema-risk in porting this as-is.

_CODEBOOK_BY_LIB = {
    "P1": "P1_codebook.csv",
    "C3": "C3v1_codebook.csv",
    "C2": "C2v6_codebook_mplx.csv",
    "I0": "I0_codebook.csv",
    "LT1": "LT1v0_codebook.csv",
    "LT2": "LT2v0_codebook.csv",
}

_MICROSCOPE_PARAMETERS_BY_SCOPE = {
    ("ST2", "60X"): "STORM2_60X.json",
    ("ST2", "40X"): "STORM2_40X.json",
    ("MF3", "60X"): "MERFISH3.json",
    ("MF4", "60X"): "MERFISH4.json",
    ("MF5", "60X"): "MERFISH5.json",
}
# Which objective each microscope uses when the caller doesn't name one --
# mirrors acquisition.configs._DEFAULT_OBJECTIVE, keeping every existing
# call site (single objective per scope, historically) working unchanged.
_DEFAULT_MICROSCOPE_PARAMETERS_OBJECTIVE = {
    "ST2": "60X", "MF3": "60X", "MF4": "60X", "MF5": "60X",
}


def resolve_codebook_filename(lib_name: str) -> str:
    """Return the codebook CSV filename for *lib_name*, raising if unknown."""
    try:
        return _CODEBOOK_BY_LIB[lib_name]
    except KeyError:
        raise ValueError(
            f"No codebook mapping for lib_name={lib_name!r}. "
            f"Known: {sorted(_CODEBOOK_BY_LIB)}"
        ) from None


def resolve_microscope_parameters_filename(microscope: str, objective: Optional[str] = None) -> str:
    """
    Return the microscope-parameters JSON filename for *microscope* +
    *objective*, raising if unknown.

    *objective* (e.g. ``"60X"``, ``"40X"``) defaults to that microscope's
    entry in ``_DEFAULT_MICROSCOPE_PARAMETERS_OBJECTIVE`` (today, every
    microscope has exactly one) -- omit it to keep prior single-objective-
    per-scope behaviour unchanged.
    """
    microscope = microscope.upper()
    obj = objective.upper() if objective is not None else \
        _DEFAULT_MICROSCOPE_PARAMETERS_OBJECTIVE.get(microscope, "")
    try:
        return _MICROSCOPE_PARAMETERS_BY_SCOPE[(microscope, obj)]
    except KeyError:
        raise ValueError(
            f"No microscope-parameters mapping for microscope={microscope!r}, "
            f"objective={obj!r}. Known: {sorted(_MICROSCOPE_PARAMETERS_BY_SCOPE)}"
        ) from None


# ── Microscope parameters ───────────────────────────────────────────────────

def create_microscope_parameters_json(
    output_path:       Path,
    flip_horizontal:    Optional[bool]              = None,
    flip_vertical:      Optional[bool]               = None,
    transpose:          Optional[bool]               = None,
    image_dimensions:   Optional[Tuple[int, int]]     = None,
    microns_per_pixel:  float                         = 0.109,
) -> Path:
    """
    Write a MERlin microscope-parameters JSON.

    Only non-``None`` fields are included, matching the real templates —
    e.g. ``MERFISH5.json`` has only ``microns_per_pixel``, while
    ``MERFISH3/4.json`` and ``STORM2_60X.json``/``STORM2_40X.json`` add
    ``flip_horizontal``/``flip_vertical``/``transpose``/``image_dimensions``.
    """
    params: Dict[str, Any] = {}
    if flip_horizontal is not None:
        params["flip_horizontal"] = flip_horizontal
    if flip_vertical is not None:
        params["flip_vertical"] = flip_vertical
    if transpose is not None:
        params["transpose"] = transpose
    if image_dimensions is not None:
        params["image_dimensions"] = list(image_dimensions)
    params["microns_per_pixel"] = microns_per_pixel

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=4)
    return output_path


def load_microscope_orientation(microscope: str, microscope_dir: Path) -> Dict[str, bool]:
    """
    Read a microscope's ``flip_horizontal``/``flip_vertical``/``transpose``
    flags from its MERlin microscope-parameters JSON (resolved via
    :func:`resolve_microscope_parameters_filename`).

    Defaults for an absent field match MERlin's own
    (``merlin.core.dataset.Dataset._load_microscope_parameters``) exactly --
    confirmed directly against that source, not assumed:
    ``flip_horizontal=True``, ``flip_vertical=False``, ``transpose=True``.
    A file with none of the three (e.g. ``MERFISH5.json``, which has only
    ``microns_per_pixel``) is therefore NOT "no transform" -- it's MERlin's
    full default orientation.

    Parameters
    ----------
    microscope      : microscope id, e.g. ``"ST2"``
    microscope_dir  : directory containing the microscope-parameters JSONs
                      (``MERci/data/configs/merlin/microscope/``)

    Returns
    -------
    dict with keys ``flip_horizontal``, ``flip_vertical``, ``transpose``
    (all ``bool``), ready to pass as ``**kwargs`` to
    :func:`apply_microscope_orientation`.
    """
    path = Path(microscope_dir) / resolve_microscope_parameters_filename(microscope)
    params = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {
        "flip_horizontal": params.get("flip_horizontal", True),
        "flip_vertical":   params.get("flip_vertical", False),
        "transpose":       params.get("transpose", True),
    }


def apply_microscope_orientation(
    image:           np.ndarray,
    flip_horizontal: bool = True,
    flip_vertical:   bool = False,
    transpose:       bool = True,
) -> np.ndarray:
    """
    Re-orient a raw camera frame to match MERlin's own camera->stage
    convention, in MERlin's own order (confirmed directly against
    ``merlin.core.dataset.Dataset.load_image``, not assumed):
    **transpose, then flip_horizontal (axis=1), then flip_vertical (axis=0)**
    -- each step applied only if its flag is ``True``.

    Use this (with :func:`load_microscope_orientation`'s output) anywhere a
    raw frame needs to be displayed/assembled in the same orientation MERlin
    itself decodes it in -- e.g. a diagnostic mosaic laid out by stage
    position, which otherwise appears rotated/transposed relative to the
    real tissue layout.

    Parameters
    ----------
    image           : 2-D array, any dtype
    flip_horizontal : mirror along axis 1 (columns)
    flip_vertical   : mirror along axis 0 (rows)
    transpose       : swap axes 0 and 1

    Returns
    -------
    Re-oriented array (a view where possible; do not rely on it sharing
    memory with *image*).
    """
    out = np.asarray(image)
    if transpose:
        out = np.transpose(out)
    if flip_horizontal:
        out = np.flip(out, axis=1)
    if flip_vertical:
        out = np.flip(out, axis=0)
    return out


# ── Codebook ─────────────────────────────────────────────────────────────────

def create_codebook_csv(
    output_path:    Path,
    codebook_name:  str,
    bit_names:      Sequence[str],
    genes:          Sequence[Tuple[str, str, str]],
    version:        str = "0.000000",
) -> Path:
    """
    Write a MERlin codebook CSV: ``version``/``codebook_name``/``bit_names``
    header rows, then one ``name, id, barcode`` row per entry in *genes*.

    Format verified against ``LT2v0_codebook.csv``/``C3v1_codebook.csv``
    (``MERci/data/configs/merlin/codebooks/``).

    Parameters
    ----------
    bit_names : ordered readout-probe names (bit 1..N), e.g. ``["RS0015",
                "RS0083", ...]``. Derivable from ``round_bit_color_map.csv``
                (bit -> round/color) + ``readouts.csv`` (bit -> probe name),
                both of which already exist per experiment.
    genes     : ``(name, id, barcode)`` rows, INCLUDING the ``Blank-N`` rows —
                this is a wet-lab library-design decision (which gene lights
                up in which bit combination) that MERci cannot generate; the
                caller supplies it (e.g. read from an existing codebook-design
                spreadsheet). ``barcode`` is a string of ``"0"``/``"1"``
                characters, one per entry in *bit_names*, in the same order.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"version, {version}\n")
        fh.write(f"codebook_name, {codebook_name}\n")
        fh.write("bit_names, " + ", ".join(bit_names) + "\n")
        fh.write("name, id, barcode\n")
        for name, gene_id, barcode in genes:
            if len(barcode) != len(bit_names):
                raise ValueError(
                    f"Barcode for {name!r} has {len(barcode)} bits, "
                    f"expected {len(bit_names)} (len(bit_names))."
                )
            fh.write(f"{name}, {gene_id}, {barcode}\n")
    return output_path


# ── Cluster resource allocation ─────────────────────────────────────────────

def create_cluster_resource_allocation(
    template_path:          Path,
    exp_name:                str,
    n_optimize_iterations:   int,
    output_path:             Path,
) -> Path:
    """
    Build a per-experiment cluster-resource-allocation JSON from the shared
    *template_path* (e.g. ``MERci/data/configs/merlin/snakemake/
    cluster_resource_allocation_basic.json``):

    1. Replace the ``"YourExperimentName"`` placeholder in
       ``__default__["out"]``/``["err"]`` with *exp_name*.
    2. Duplicate the template's ``"Optimize01"`` block into
       ``"Optimize02"``..``"Optimize{n:02d}"``.

    Verified safe as a plain deep-copy (no recursive string-rewrite needed):
    in the real template, each per-task entry is a flat resource-override
    dict (e.g. ``{"mem": 10000}``, occasionally with ``partition``/``gres``/
    ``time``/``exclude`` for GPU tasks) — nothing embeds its own task name in
    a leaf value, unlike the ``__default__`` block's ``out``/``err`` log paths.
    """
    with open(template_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    default = config.get("__default__", {})
    for key in ("out", "err"):
        if key in default:
            default[key] = default[key].replace("YourExperimentName", exp_name)

    if "Optimize01" in config:
        base = config["Optimize01"]
        for i in range(2, n_optimize_iterations + 1):
            config[f"Optimize{i:02d}"] = copy.deepcopy(base)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4)
    return output_path


# ── Snakemake parameters ─────────────────────────────────────────────────────

def create_snakemake_parameters(
    exp_name:             str,
    cluster_config_path:  Path,
    output_path:          Path,
    nodes:                int = 1000,
    restart_times:        int = 2,
    job_name_prefix:      Optional[str] = None,
) -> Path:
    """
    Write snakemake's top-level parameters JSON. Format verified against a
    real generated file (``parameters_LT048.json``).

    job_name_prefix : optional SLURM job-name prefix (MERlin's
        ``run_with_snakemake`` defaults to ``"merlin"`` when this key is
        absent) -- see :func:`short_experiment_name` to derive one from
        ``(sample_name, imaging_dir)``.
    """
    config = {
        "cluster": (
            "sbatch --gres={cluster.gres} --mem={cluster.mem} "
            "--constraint={cluster.constraint} --exclude={cluster.exclude} "
            "-A {cluster.account} -p {cluster.partition} -n {cluster.n}  "
            "-t {cluster.time} -o {cluster.out} -e {cluster.err} "
        ),
        "cluster_config": str(cluster_config_path),
        "nodes": nodes,
        "restart_times": restart_times,
    }
    if job_name_prefix is not None:
        config["job_name_prefix"] = job_name_prefix
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4)
    return output_path


# ── Slurm submit script ─────────────────────────────────────────────────────

# Cluster project roots, keyed by a token searched for (case-insensitively) in
# the sample name. Extend this for other projects -- see
# resolve_cluster_sample_dir.
_PROJECT_CLUSTER_ROOTS: Dict[str, str] = {
    "BC": "/n/holylfs05/LABS/zhuang_lab/Lab/shared/projects/breast_cancer/experiments",
    "LT": "/n/holylfs06/LABS/zhuang_lab/Lab/shared/Leonardo/projects/lineage_tracing/experiments",
}


def short_experiment_name(sample_name: str, imaging_dir: str) -> str:
    """
    Turn ``(sample_name, imaging_dir)`` -- the same two values
    :func:`resolve_cluster_sample_dir` takes -- into a short,
    SLURM-job-name-safe string, for use as ``job_name_prefix`` in
    :func:`create_snakemake_parameters`.

    Inferred from only two real examples -- verify/extend against more real
    sample names before relying on it beyond MERlin job-name prefixes:

    ============================  ===========  ===========
    sample_name                   imaging_dir  short name
    ============================  ===========  ===========
    BC553_sample_02_test          epi          s2-BC553e-t
    LT060_sample_05               merfish      s5-LT060m
    ============================  ===========  ===========

    Rule, for *sample_name* matching ``{exp_id}_sample_{NN}[_{suffix}]``:

    1. ``sample_{NN}`` -> ``s{int(NN)}`` (leading zeros stripped)
    2. ``{exp_id}`` + first letter of *imaging_dir* -> ``{exp_id}{c}``
       (no separator; omitted when *imaging_dir* is ``""``)
    3. optional trailing ``_{suffix}`` -> ``-{first letter of suffix}``
    4. join non-empty parts with ``-``: ``s{N}-{exp_id}{c}[-{suffix_letter}]``

    Falls back to *sample_name* unchanged when it doesn't match the
    ``{exp_id}_sample_{NN}[_{suffix}]`` pattern -- SLURM job names already
    accept any alphanumeric/underscore/hyphen string (enforced by
    ``snakemake-executor-plugin-slurm``), so this is always safe, just not
    necessarily short.

    Parameters
    ----------
    sample_name : experiment id, e.g. ``"LT048_sample_26"``
    imaging_dir : acquisition-type subfolder name, e.g. ``"epi"``/``"merfish"``;
        ``""`` when this acquisition isn't split into its own subfolder

    Returns
    -------
    str : short SLURM-job-name-safe string
    """
    match = re.match(r"^(?P<exp_id>.+)_sample_(?P<nn>\d+)(?:_(?P<suffix>.+))?$", sample_name)
    if match is None:
        return sample_name

    exp_part = match["exp_id"] + (imaging_dir[0] if imaging_dir else "")
    parts = [f"s{int(match['nn'])}", exp_part]
    if match["suffix"]:
        parts.append(match["suffix"][0])
    return "-".join(parts)


def resolve_cluster_sample_dir(sample_dir: Path, sample_name: str, imaging_dir: str) -> str:
    """
    Resolve this experiment's acquisition-root path as it will be addressed
    from the Linux cluster -- the directory holding this MERci clone,
    metadata/, positions/, and merlin/ as siblings.

    The generated slurm script works identically whether it's produced on the
    microscope computer / a laptop (Windows), before the experiment folder is
    transferred to the cluster, or regenerated on the cluster itself
    afterwards:

    * On Linux (already running on the cluster), *sample_dir*'s own current
      absolute path already IS that answer -- no guessing needed.
    * On Windows, that path doesn't exist yet, so it is predicted from
      *sample_name*: a name containing ``"BC"`` resolves under the
      breast_cancer project root, ``"LT"`` under lineage_tracing (see
      ``_PROJECT_CLUSTER_ROOTS``), joined with *imaging_dir* (the acquisition-
      type subfolder between the sample and its MERci clone -- ``"merfish"``,
      ``"lineage"``, ``"epi"``, ``"disk"`` -- or nothing when this acquisition
      doesn't (yet) live under its own subfolder).
    * **Fallback** when *sample_name* itself has no project token: this
      happens once an acquisition actually lives under its own subfolder
      (``sample_dir`` = e.g. ``.../LT058_sample_07/merfish``) and
      *sample_name* is really that subfolder's own local file-naming
      convention (``"merfish"``) rather than the true top-level experiment
      id -- notebook 06 derives ``sample_name`` from ``SAMPLE_DIR.name``,
      which is only the parent experiment folder in the flat, unsplit
      layout. In that case the true experiment id is one level up
      (``sample_dir.parent.name``); if *that* contains a project token, it
      is used instead, together with ``sample_dir.name`` as the acquisition
      subfolder -- regardless of what *imaging_dir* was set to, since the
      real folder structure is more reliable than possibly-stale metadata.

    Parameters
    ----------
    sample_dir  : this experiment's local acquisition-root ``Path`` (whatever
                  OS the notebook is currently running on)
    sample_name : experiment id, e.g. ``"251225_LT027_saving_time"`` -- or,
                  in the split-subfolder layout, this acquisition's own local
                  file-naming convention (e.g. ``"merfish"``); see the
                  fallback above
    imaging_dir : acquisition-type subfolder name (``experiment_info.yaml``'s
                  ``extra["imaging_dir"]``), e.g. ``"merfish"``; ``""`` when
                  this acquisition's data/metadata/positions/settings/merlin
                  sit directly under the sample folder (no subfolder)

    Returns
    -------
    str : POSIX absolute path to the acquisition root, for use as the
          generated script's ``$SAMPLE_DIR``
    """
    if sys.platform.startswith("linux"):
        return sample_dir.as_posix()

    sample_dir = Path(sample_dir)

    def _cluster_root(experiment_id: str) -> Optional[str]:
        upper = experiment_id.upper()
        for token, root in _PROJECT_CLUSTER_ROOTS.items():
            if token in upper:
                return root
        return None

    # Usual case: sample_name IS the top-level experiment id (unsplit layout,
    # or imaging_dir was set so sample_name was derived from the parent
    # folder already).
    root = _cluster_root(sample_name)
    if root is not None:
        acquisition_subpath = f"{sample_name}/{imaging_dir}" if imaging_dir else sample_name
        return f"{root}/{acquisition_subpath}"

    # Fallback: this acquisition already lives under its own acquisition-type
    # subfolder (sample_dir.name, e.g. "merfish"/"lineage"/"epi"/"disk") but
    # sample_name is that subfolder's own local file-naming convention, not
    # the true top-level experiment id -- which is one level up
    # (sample_dir.parent.name). Trust the actual folder structure over
    # possibly-unset/stale imaging_dir metadata, rather than requiring the
    # caller to have set imaging_dir correctly for this to work.
    parent_root = _cluster_root(sample_dir.parent.name)
    if parent_root is not None:
        return f"{parent_root}/{sample_dir.parent.name}/{sample_dir.name}"

    raise ValueError(
        f"Cannot infer the cluster project root for sample_name={sample_name!r} "
        f"or its parent folder {sample_dir.parent.name!r}: expected one of them "
        f"to contain one of {list(_PROJECT_CLUSTER_ROOTS)}. "
        f"Extend _PROJECT_CLUSTER_ROOTS in merlin_config.py for other projects."
    )


def create_slurm_submit_script(
    label:                    str,
    sample_dir:               str,
    parameters_file:          str,
    analysis_file:            str,
    data_organization_file:   str,
    positions_file:           str,
    codebook_file:            str,
    microscope_file:          str,
    data_home:                str,
    folder_name:              str,
    output_path:              Path,
    slurm_out_path:            Optional[str] = None,
    slurm_err_path:            Optional[str] = None,
    mem_mb:                    int    = 5000,
    time_limit:                str    = "2-00:00:00",
    partition:                 str    = "zhuang",
    conda_env:                 str    = "merlin_cp4_env",
    conda_pkgs_dir:            str    = "/n/holylabs/zhuang_lab/Lab/lsepulvedaduran/conda/pkgs",
    conda_envs_path:           str    = "/n/holylabs/zhuang_lab/Lab/lsepulvedaduran/conda/envs",
    allow_ragged_z_stacks:     bool   = False,
    analysis_name:             Optional[str] = None,
) -> Path:
    """
    Write the sbatch script that runs ``merlin`` for one experiment.

    Ports the CURRENT live template (``merlin_slurm_LT048_sample_26_epi_
    deconv.sh``, May 2026) — this is newer than, and differs from, the old
    analysis notebook's own hardcoded version (that one used an older conda
    env and no cuda/cudnn modules; the notebook itself had drifted out of
    sync with current cluster practice).

    Every path the script needs is written relative to one ``$SAMPLE_DIR``
    bash variable (see *sample_dir*) instead of five separately-resolved
    absolute paths, so the script stays readable and portable between the
    machine that generated it and the cluster that runs it.

    Parameters
    ----------
    label          : job label, used in ``echo`` and default log filenames
                     (e.g. ``"LT048_sample_26_epi_deconv"``)
    sample_dir     : this experiment's acquisition-root path as addressed from
                     the cluster (see :func:`resolve_cluster_sample_dir`) —
                     written into the script as ``SAMPLE_DIR="..."``, which
                     every ``*_file`` below is resolved against
    *_file         : POSIX paths *relative to sample_dir* (e.g.
                     ``"merlin/analysis/merlin_analysis_X.json"``) — matches
                     the ``-k -a -o -p -c -m`` flags below, each emitted as
                     ``"$SAMPLE_DIR/<file>"``
    data_home      : merlin's ``-e`` flag — the root of ALL experiments on the
                     cluster; independent of *sample_dir* (a different,
                     broader root), so passed through as-is
    folder_name    : merlin's positional arg — the raw-data path, relative to
                     *data_home*
    output_path    : where to write this script
    slurm_out_path/slurm_err_path : POSIX paths *relative to sample_dir* for
                     the SBATCH log files; default to
                     ``"merlin/slurm/{out,err}/{label}.{out,err}"``. SBATCH
                     directives are parsed by the scheduler before the script
                     runs as shell, so they can't expand ``$SAMPLE_DIR`` —
                     these are written as a literal ``{sample_dir}/...`` path
    conda_pkgs_dir/conda_envs_path : lab-wide conda storage convention;
                     override if this changes
    allow_ragged_z_stacks : passes ``--allow-ragged-z-stacks`` to ``merlin``
                     -- a variable-z-per-FOV experiment's raw files
                     legitimately have fewer frames than the deepest FOV's,
                     which the CLI otherwise treats as a hard error. Requires
                     a MERlin checkout with ragged-z-stack support (as of
                     this writing, still in progress -- see
                     ``prompt_history/2026_07_16_1015_variable_z_per_fov_
                     storm_control_investigation.md``); ``False`` (default)
                     leaves the generated script identical to before.
    analysis_name  : passes ``-x <analysis_name>`` to ``merlin``, so its
                     analysis/output folder is ``analysisHome/<analysis_name>``
                     (e.g. ``"$SAMPLE_DIR/merlin/output"``) instead of
                     ``analysisHome/<folder_name>`` -- MERlin otherwise mirrors
                     the raw-data folder's own (possibly deep/nested) path
                     under the analysis home. Requires a MERlin checkout with
                     the ``-x``/``--analysis-name`` flag (``~/Software/
                     merlin_cc`` as of this writing; NOT present in the older
                     ``merlin_cp4_env`` fork). ``None`` (default) omits the
                     flag, leaving the generated script identical to before.
    """
    output_path = Path(output_path)
    if slurm_out_path is None:
        slurm_out_path = f"merlin/slurm/out/{label}.out"
    if slurm_err_path is None:
        slurm_err_path = f"merlin/slurm/err/{label}.err"

    sample_dir = sample_dir.rstrip("/")
    ragged_flag = " \\\n       --allow-ragged-z-stacks" if allow_ragged_z_stacks else ""
    analysis_name_flag = f' \\\n       -x "{analysis_name}"' if analysis_name else ""

    script = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p {partition}
#SBATCH -t {time_limit}
#SBATCH --mem {mem_mb}
#SBATCH --open-mode=append
#SBATCH -o {sample_dir}/{slurm_out_path}
#SBATCH -e {sample_dir}/{slurm_err_path}

date +'Starting at %R.'

module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01
module load python
export CONDA_PKGS_DIRS={conda_pkgs_dir}
export CONDA_ENVS_PATH={conda_envs_path}
source activate {conda_env}
echo {label}

SAMPLE_DIR="{sample_dir}"

merlin -k "$SAMPLE_DIR/{parameters_file}" \\
       -a "$SAMPLE_DIR/{analysis_file}" \\
       -o "$SAMPLE_DIR/{data_organization_file}" \\
       -p "$SAMPLE_DIR/{positions_file}" \\
       -c "$SAMPLE_DIR/{codebook_file}" \\
       -m "$SAMPLE_DIR/{microscope_file}" \\
       -n 1000 \\
       -e {data_home} \\
       -s "$SAMPLE_DIR/merlin"{analysis_name_flag}{ragged_flag} \\
       {folder_name}

date +'Finished at %R.'
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(script)
    return output_path


# ── MERlin analysis-parameters JSON ─────────────────────────────────────────
# Replaces "copy a prior experiment's file and hand-edit" entirely. Verified
# against 4 real files: merlin_analysis_LT048.json (== merlin_analysis_
# ref_no_cells.json byte-for-byte -- the no-segmentation baseline these
# defaults are copied from), merlin_analysis_BC522.json (baseline + reporting
# + full CellPoseSegment3D chain), analysis_cellposeSAM_Dark_only.json
# (the lighter CellPoseSegmentSAM variant), and analysis_decode_v2_
# aaron_260816.json (same overall task chain -- FiducialCorrelationWarp ->
# DeconvolutionPreprocess -> chained OptimizeIteration -> Decode ->
# GenerateAdaptiveThreshold -> AdaptiveFilterBarcodes -> ExportBarcodes ->
# PlotPerformance -- with a global-alignment task in the mix).

# The global-alignment task used everywhere below. LeastSquaresGlobalAlignment
# (`merlin.analysis.globalalign`, same module as SimpleGlobalAlignment) is
# merlin_cc's current implementation -- corrects each fov's nominal position
# via a joint sparse least-squares solve over real pairwise image-registration
# measurements, instead of trusting the nominal stage position outright as
# SimpleGlobalAlignment (used by the older analysis_decode_v2_aaron_260816.json
# reference) does. See that class's own docstring in `merlin/analysis/
# globalalign.py` (`~/Software/merlin_cc`) for why this correction needs a
# joint per-fov solve rather than one global affine transform.
_GLOBAL_ALIGN_TASK = "LeastSquaresGlobalAlignment"

_WARP_DEFAULTS = {
    "highpass_sigma": 20,
    "median_filter": True,
    "percentile_pixel_to_keep": 10,
    "edge_width_to_remove": 300,
    "write_fiducial_images": True,
    "write_aligned_images": False,
}
_PREPROCESS_DEFAULTS = {
    "decon_iterations": 5,
    "save_pixel_histogram": True,
    "write_preprocess_images": False,
}
_OPTIMIZE_DEFAULTS = {
    "area_threshold": 4,
    "fov_per_iteration": 50,
    "optimize_chromatic_correction": True,
    "use_gpu": False,
    "optimize_background": True,
    "write_decoded_images": True,
    "distance_threshold": 0.52,
    "min_barcodes_for_refactoring": 3,
}
_DECODE_DEFAULTS = {
    "minimum_area": 3,
    "lowpass_sigma": 1,
    "crop_width": 106,
    "distance_threshold": 0.65,
    "write_decoded_images": True,
    "use_gpu": False,
}
_FILTER_DEFAULTS = {
    "misidentification_rate": 0.05,
    "remove_z_duplicated_barcodes": False,
    "z_duplicate_zPlane_threshold": 2,
    "z_duplicate_xy_pixel_threshold": 1.4,
}
_EXPORT_COLUMNS_DEFAULT = [
    "barcode_id", "global_x", "global_y", "global_z", "x", "y", "fov", "cell_index",
]

# CellPoseSegment3D defaults, from merlin_analysis_BC522.json.
_SEGMENT_3D_DEFAULTS = {
    "diameter": 50,
    "channel_1_name": "DAPI",
    "cellpose_2D_3D_stitching": True,
    "stitch_threshold": 0.2,
    "dump_segmented_masks": True,
    "dump_segmented_images": True,
    "dump_rgb_masks": True,
}
# CellPoseSegmentSAM defaults, from analysis_cellposeSAM_Dark_only.json.
_SEGMENT_SAM_DEFAULTS = {
    "channel_1_name": "DAPI",
    "do_3D": True,
    "dump_segmented_masks": True,
    "dump_segmented_images": False,
    "downsample_factor": 4,
    "flow3D_smooth": 1,
    "expand_mask": 2,
    "min_size": 100,
}


@dataclass
class MerlinAnalysisSpec:
    """
    A compact description of which MERlin analysis steps to include and how
    to tune them — the "smaller file that describes which steps want to be
    included", replacing copy+hand-edit of a prior experiment's task-
    parameters JSON. Each ``*_params`` dict is merged OVER the verified
    defaults above (only override what differs for this experiment).

    Round-trips to/from YAML via :meth:`save`/:meth:`load`, so a spec can live
    as its own small per-experiment (or shared default) file.
    """
    n_optimize_iterations: int = 15
    warp_params:           Dict[str, Any] = field(default_factory=dict)
    preprocess_params:     Dict[str, Any] = field(default_factory=dict)
    optimize_params:       Dict[str, Any] = field(default_factory=dict)
    decode_params:         Dict[str, Any] = field(default_factory=dict)
    filter_params:         Dict[str, Any] = field(default_factory=dict)
    export_columns:        List[str] = field(default_factory=lambda: list(_EXPORT_COLUMNS_DEFAULT))
    include_reporting:     bool = True
    include_segmentation:  bool = False
    segmentation_method:   str  = "CellPoseSegment3D"   # or "CellPoseSegmentSAM"
    segmentation_params:   Dict[str, Any] = field(default_factory=dict)
    include_smfish:        bool = False   # smFISH spot detection on sequential (non-barcode) bits
    smfish_channel_names:  List[str] = field(default_factory=list)
    smfish_params:         Dict[str, Any] = field(default_factory=dict)
    include_sum_signal:    bool = False   # simple summed per-cell intensity on sequential bits
    sum_signal_params:     Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> "MerlinAnalysisSpec":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)


def _task(task: str, module: str, parameters: Optional[dict] = None,
          analysis_name: Optional[str] = None) -> dict:
    entry: Dict[str, Any] = {"task": task, "module": module}
    if analysis_name is not None:
        entry["analysis_name"] = analysis_name
    if parameters is not None:
        entry["parameters"] = parameters
    return entry


def create_merlin_analysis_parameters(spec: MerlinAnalysisSpec, output_path: Path) -> Path:
    """
    Build MERlin's ``analysis_tasks`` JSON from *spec*.

    Assembles, in order: FiducialCorrelationWarp, DeconvolutionPreprocess,
    ``n_optimize_iterations`` chained OptimizeIteration tasks, Decode,
    LeastSquaresGlobalAlignment, GenerateAdaptiveThreshold, AdaptiveFilterBarcodes,
    ExportBarcodes — the verified no-segmentation baseline (matches
    ``merlin_analysis_ref_no_cells.json``/``merlin_analysis_LT048.json``
    exactly when every ``*_params`` override is empty, aside from the global-
    alignment task -- those two reference files, like
    ``analysis_decode_v2_aaron_260816.json``, still use the older
    ``SimpleGlobalAlignment``) — then, if requested,
    PlotPerformance + SlurmReport (``include_reporting``), then a full
    segmentation chain (``include_segmentation``): CellPoseSegment3D or
    CellPoseSegmentSAM -> CleanCellBoundaries -> CombineCleanedBoundaries ->
    RefineCellDatabases -> PartitionBarcodes -> ExportPartitionedBarcodes ->
    ExportCellMetadata (matches ``merlin_analysis_BC522.json``'s tail) -- and
    finally two independent optional tail steps for sequential (non-barcode)
    bits: ``include_smfish`` (SmfishSignal spot detection) and
    ``include_sum_signal`` (SumSignal + ExportSumSignals, simple summed
    per-cell intensity). Both read the bits MERlin considers "sequential"
    from the data-organization file itself (any ``readoutName`` absent from
    the codebook's ``bit_names``) -- see ``merlin.data.dataorganization.
    DataOrganization.get_sequential_rounds``.

    Building this programmatically (rather than copying and hand-editing, as
    before) also avoids a real bug present in the live ``merlin_analysis_
    LT048.json`` template: a copy-pasted duplicate ``"warp_task"`` key inside
    its ``Optimize05`` block's parameters.
    """
    warp_params       = {**_WARP_DEFAULTS, **spec.warp_params}
    preprocess_params = {**_PREPROCESS_DEFAULTS, "warp_task": "FiducialCorrelationWarp",
                          **spec.preprocess_params}
    decode_params     = {**_DECODE_DEFAULTS, "preprocess_task": "DeconvolutionPreprocess",
                          "optimize_task": f"Optimize{spec.n_optimize_iterations:02d}",
                          "global_align_task": _GLOBAL_ALIGN_TASK,
                          **spec.decode_params}
    filter_params     = {**_FILTER_DEFAULTS, "decode_task": "Decode",
                          "adaptive_task": "GenerateAdaptiveThreshold", **spec.filter_params}

    tasks = [
        _task("FiducialCorrelationWarp", "merlin.analysis.warp", warp_params),
        _task("DeconvolutionPreprocess", "merlin.analysis.preprocess", preprocess_params),
    ]

    for i in range(1, spec.n_optimize_iterations + 1):
        name = f"Optimize{i:02d}"
        params = {
            "preprocess_task": "DeconvolutionPreprocess",
            "warp_task": "FiducialCorrelationWarp",
            **_OPTIMIZE_DEFAULTS,
            "random_seed": i,
            **spec.optimize_params,
        }
        if i > 1:
            params["previous_iteration"] = f"Optimize{i - 1:02d}"
        tasks.append(_task("OptimizeIteration", "merlin.analysis.optimize", params, name))

    tasks.append(_task("Decode", "merlin.analysis.decode", decode_params, "Decode"))
    tasks.append(_task(_GLOBAL_ALIGN_TASK, "merlin.analysis.globalalign"))
    tasks.append(_task(
        "GenerateAdaptiveThreshold", "merlin.analysis.filterbarcodes",
        {"decode_task": "Decode", "run_after_task": "Decode"},
    ))
    tasks.append(_task("AdaptiveFilterBarcodes", "merlin.analysis.filterbarcodes", filter_params))
    tasks.append(_task(
        "ExportBarcodes", "merlin.analysis.exportbarcodes",
        {"filter_task": "AdaptiveFilterBarcodes", "columns": list(spec.export_columns),
         "exclude_blanks": False},
    ))

    if spec.include_reporting:
        tasks.append(_task(
            "PlotPerformance", "merlin.analysis.plotperformance",
            {"preprocess_task": "DeconvolutionPreprocess",
             "optimize_task": f"Optimize{spec.n_optimize_iterations:02d}",
             "decode_task": "Decode", "filter_task": "AdaptiveFilterBarcodes",
             "run_after_task": "ExportBarcodes"},
        ))
        tasks.append(_task(
            "SlurmReport", "merlin.analysis.slurmreport",
            {"run_after_task": "ExportBarcodes"}, "SlurmReport",
        ))

    if spec.include_segmentation:
        method = spec.segmentation_method
        if method == "CellPoseSegment3D":
            seg_params = {"warp_task": "FiducialCorrelationWarp",
                          "global_align_task": _GLOBAL_ALIGN_TASK,
                          **_SEGMENT_3D_DEFAULTS, **spec.segmentation_params}
        elif method == "CellPoseSegmentSAM":
            seg_params = {"warp_task": "FiducialCorrelationWarp",
                          "global_align_task": _GLOBAL_ALIGN_TASK,
                          **_SEGMENT_SAM_DEFAULTS, **spec.segmentation_params}
        else:
            raise ValueError(
                f"Unknown segmentation_method={method!r}; "
                f"expected 'CellPoseSegment3D' or 'CellPoseSegmentSAM'."
            )
        tasks.append(_task(method, "merlin.analysis.segment", seg_params))
        tasks.append(_task(
            "CleanCellBoundaries", "merlin.analysis.segment",
            {"segment_task": method, "global_align_task": _GLOBAL_ALIGN_TASK},
        ))
        tasks.append(_task(
            "CombineCleanedBoundaries", "merlin.analysis.segment",
            {"cleaning_task": "CleanCellBoundaries"},
        ))
        tasks.append(_task(
            "RefineCellDatabases", "merlin.analysis.segment",
            {"segment_task": method, "combine_cleaning_task": "CombineCleanedBoundaries"},
        ))
        tasks.append(_task(
            "PartitionBarcodes", "merlin.analysis.partition",
            {"filter_task": "AdaptiveFilterBarcodes", "assignment_task": "RefineCellDatabases",
             "alignment_task": _GLOBAL_ALIGN_TASK, "codebook_index": 0},
            "PartitionBarcodes",
        ))
        tasks.append(_task(
            "ExportPartitionedBarcodes", "merlin.analysis.partition",
            {"partition_task": "PartitionBarcodes", "codebook_index": 0},
            "ExportPartitionedBarcodes",
        ))
        tasks.append(_task(
            "ExportCellMetadata", "merlin.analysis.segment",
            {"segment_task": "RefineCellDatabases"},
        ))

    if spec.include_smfish:
        if not spec.smfish_channel_names:
            raise ValueError("include_smfish=True requires smfish_channel_names.")
        smfish_params = {
            "warp_task": "FiducialCorrelationWarp",
            "global_align_task": _GLOBAL_ALIGN_TASK,
            "channel_names": list(spec.smfish_channel_names),
            **({"segment_task": "RefineCellDatabases"} if spec.include_segmentation else {}),
            **spec.smfish_params,
        }
        tasks.append(_task("SmfishSignal", "merlin.analysis.sequential", smfish_params))

    if spec.include_sum_signal:
        if not spec.include_segmentation:
            raise ValueError("include_sum_signal=True requires include_segmentation=True "
                              "(SumSignal needs a segment_task).")
        sum_signal_params = {
            "warp_task": "FiducialCorrelationWarp",
            "global_align_task": _GLOBAL_ALIGN_TASK,
            "segment_task": "RefineCellDatabases",
            **spec.sum_signal_params,
        }
        tasks.append(_task("SumSignal", "merlin.analysis.sequential", sum_signal_params, "SumSignal"))
        tasks.append(_task("ExportSumSignals", "merlin.analysis.sequential",
                            {"sequential_task": "SumSignal"}))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({"analysis_tasks": tasks}, fh, indent=4)
    return output_path


# ── MERlin analysis-parameters JSON, atomic-task/recipe path (default) ──────
# Replaces the MerlinAnalysisSpec/create_merlin_analysis_parameters path above
# as the notebooks' default: each MERlin task's own tunable defaults live in
# its own small YAML file under data/configs/merlin/analysis/tasks/ (one file
# per literal MERlin task, including pure cross-reference "wiring" tasks that
# carry no independent tunables), and a "recipe" YAML under
# data/configs/merlin/analysis/recipes/ names the explicit, ordered list of
# task-file names to assemble -- replacing the old include_reporting/
# include_segmentation/include_smfish/include_sum_signal boolean toggles with
# a literal, editable list. Structural cross-references between tasks
# (warp_task, preprocess_task, optimize_task, previous_iteration,
# global_align_task, segment_task, ...) are NEVER stored in an atom file --
# build_merlin_analysis_parameters() injects them, resolved from whichever
# atoms are actually present in the recipe (e.g. global_align_task points at
# whichever of global_align_simple/global_align_least_squares was included).
#
# Default recipe values (data/configs/merlin/analysis/recipes/default_*.yaml)
# were built to match analysis_decode_v2_aaron_260816.json (`~/Software/
# merfish-parameters/analysis/`) field-for-field for every task it defines
# (warp, preprocess, optimize, decode, filter, export, plot) -- explicit user
# choice, verified real difference from the OLDER MerlinAnalysisSpec Python
# defaults above (those trace to merlin_analysis_LT048.json/BC522 instead):
# n_optimize_iterations 10 (not 15), decon_iterations 0 (not 5), highpass_sigma
# 3 (not 20), crop_width 100 (not 106); several fields that reference file
# doesn't set at all (median_filter, percentile_pixel_to_keep,
# edge_width_to_remove, min_barcodes_for_refactoring, use_gpu,
# remove_z_duplicated_barcodes) are likewise left unset here -- confirmed
# directly against merlin_cc's own source (warp.py/optimize.py/decode.py/
# filterbarcodes.py) that MERlin's own internal defaults then apply, matching
# analysis_decode_v2_aaron_260816.json's real runtime behavior exactly, NOT a
# regression from the old explicit values. Two deliberate departures from
# that reference file (explicit user choice): the global-alignment atom is
# global_align_least_squares (LeastSquaresGlobalAlignment) instead of that
# file's SimpleGlobalAlignment, and slurm_report (absent from that file) is
# included. chromatic_correction_file (that file sets an absolute path under
# a different user's home directory) is deliberately left unset in
# optimize_iteration.yaml -- opt-in per experiment via `overrides`, never a
# shared repo default. Segmentation/smfish/sum_signal atoms (absent from that
# reference file entirely) keep the older MerlinAnalysisSpec defaults' values
# (BC522/analysis_cellposeSAM_Dark_only.json-derived), since there's nothing
# in analysis_decode_v2_aaron_260816.json to match them against.

# Atom names that name ONE of several mutually-exclusive alternatives for the
# same structural role -- build_merlin_analysis_parameters uses whichever one
# is actually present in the recipe's task list to fill every OTHER task's
# global_align_task/segment_task reference.
_ALIGN_ATOM_NAMES = ("global_align_simple", "global_align_least_squares")
_SEGMENT_ATOM_NAMES = ("cellpose_segment_3d", "cellpose_segment_sam")


def _load_task_atom(name: str, tasks_dir: Path) -> dict:
    path = Path(tasks_dir) / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_merlin_analysis_parameters(
    recipe_path:            Path,
    tasks_dir:               Path,
    output_path:             Path,
    overrides:                Optional[Dict[str, Dict[str, Any]]] = None,
    extra_tasks:              Optional[List[str]] = None,
    n_optimize_iterations:    Optional[int] = None,
) -> Path:
    """
    Build MERlin's ``analysis_tasks`` JSON from a recipe (explicit ordered
    list of atomic task-file names -- see the module comment above) instead
    of a :class:`MerlinAnalysisSpec`.

    Parameters
    ----------
    recipe_path : YAML file with ``n_optimize_iterations`` and an ordered
        ``tasks`` list of atom names (files under *tasks_dir*, without the
        ``.yaml`` extension) -- e.g. ``data/configs/merlin/analysis/recipes/
        default_no_segmentation.yaml``.
    tasks_dir   : directory holding the atom YAML files (``data/configs/
        merlin/analysis/tasks/`` in this repo).
    overrides   : optional ``{atom_name: {param: value}}`` -- merged OVER
        that atom's own YAML parameters (and over any structural
        cross-reference this function injects), for per-experiment tuning
        without editing the shared atom/recipe files. ``smfish_signal``
        needs ``channel_names`` supplied this way (no sane shared default);
        ``optimize_iteration`` overrides apply identically to every
        iteration.
    extra_tasks : optional atom names appended after the recipe's own
        ``tasks`` list (e.g. ``["smfish_signal"]``), for opt-in tails the
        shared default recipe doesn't include.
    n_optimize_iterations : overrides the recipe file's own value if given
        (mirrors ``MerlinAnalysisSpec.n_optimize_iterations``, e.g. sourced
        from ``experiment_info.yaml``'s ``extra.n_opt``).

    Returns
    -------
    Path : *output_path*, unchanged
    """
    with open(recipe_path, "r", encoding="utf-8") as fh:
        recipe = yaml.safe_load(fh) or {}
    n_opt = n_optimize_iterations if n_optimize_iterations is not None else recipe.get("n_optimize_iterations", 1)
    task_names = list(recipe.get("tasks", [])) + list(extra_tasks or [])
    overrides = overrides or {}
    tasks_dir = Path(tasks_dir)

    if "smfish_signal" in task_names and not overrides.get("smfish_signal", {}).get("channel_names"):
        raise ValueError("smfish_signal requires overrides={'smfish_signal': {'channel_names': [...]}}.")
    if "sum_signal" in task_names and "cellpose_segment_3d" not in task_names and "cellpose_segment_sam" not in task_names:
        raise ValueError("sum_signal requires a segmentation atom (cellpose_segment_3d/cellpose_segment_sam) "
                          "in the recipe -- SumSignal needs a segment_task.")

    align_atom = next((n for n in task_names if n in _ALIGN_ATOM_NAMES), None)
    segment_atom = next((n for n in task_names if n in _SEGMENT_ATOM_NAMES), None)
    align_task_name = _load_task_atom(align_atom, tasks_dir)["task"] if align_atom else None
    segment_task_name = _load_task_atom(segment_atom, tasks_dir)["task"] if segment_atom else None

    tasks = []
    for name in task_names:
        atom = _load_task_atom(name, tasks_dir)

        if name == "optimize_iteration":
            for i in range(1, n_opt + 1):
                params = {
                    **atom.get("parameters", {}),
                    "preprocess_task": "DeconvolutionPreprocess",
                    "warp_task": "FiducialCorrelationWarp",
                    "random_seed": i,
                }
                if i > 1:
                    params["previous_iteration"] = f"Optimize{i - 1:02d}"
                params.update(overrides.get(name, {}))
                tasks.append(_task("OptimizeIteration", atom["module"], params, f"Optimize{i:02d}"))
            continue

        params = dict(atom.get("parameters", {}))
        if name == "deconvolution_preprocess":
            params["warp_task"] = "FiducialCorrelationWarp"
        elif name == "decode":
            params["preprocess_task"] = "DeconvolutionPreprocess"
            params["optimize_task"] = f"Optimize{n_opt:02d}"
            params["global_align_task"] = align_task_name
        elif name == "generate_adaptive_threshold":
            params["decode_task"] = "Decode"
            params["run_after_task"] = "Decode"
        elif name == "adaptive_filter_barcodes":
            params["decode_task"] = "Decode"
            params["adaptive_task"] = "GenerateAdaptiveThreshold"
        elif name == "export_barcodes":
            params["filter_task"] = "AdaptiveFilterBarcodes"
        elif name == "plot_performance":
            params["preprocess_task"] = "DeconvolutionPreprocess"
            params["optimize_task"] = f"Optimize{n_opt:02d}"
            params["decode_task"] = "Decode"
            params["filter_task"] = "AdaptiveFilterBarcodes"
        elif name == "slurm_report":
            params["run_after_task"] = "ExportBarcodes"
        elif name in _SEGMENT_ATOM_NAMES:
            params["warp_task"] = "FiducialCorrelationWarp"
            params["global_align_task"] = align_task_name
        elif name == "clean_cell_boundaries":
            params["segment_task"] = segment_task_name
            params["global_align_task"] = align_task_name
        elif name == "combine_cleaned_boundaries":
            params["cleaning_task"] = "CleanCellBoundaries"
        elif name == "refine_cell_databases":
            params["segment_task"] = segment_task_name
            params["combine_cleaning_task"] = "CombineCleanedBoundaries"
        elif name == "partition_barcodes":
            params["filter_task"] = "AdaptiveFilterBarcodes"
            params["assignment_task"] = "RefineCellDatabases"
            params["alignment_task"] = align_task_name
        elif name == "export_partitioned_barcodes":
            params["partition_task"] = "PartitionBarcodes"
        elif name == "export_cell_metadata":
            params["segment_task"] = "RefineCellDatabases"
        elif name == "smfish_signal":
            params["warp_task"] = "FiducialCorrelationWarp"
            params["global_align_task"] = align_task_name
            if segment_atom:
                params["segment_task"] = "RefineCellDatabases"
        elif name == "sum_signal":
            params["warp_task"] = "FiducialCorrelationWarp"
            params["global_align_task"] = align_task_name
            params["segment_task"] = "RefineCellDatabases"
        elif name == "export_sum_signals":
            params["sequential_task"] = "SumSignal"

        params.update(overrides.get(name, {}))
        tasks.append(_task(atom["task"], atom["module"], params if params else None, atom.get("analysis_name")))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({"analysis_tasks": tasks}, fh, indent=4)
    return output_path
