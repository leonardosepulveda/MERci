# MERci/acquisition/fishtank_config.py
"""
Generate fishtank's input/config files and run scripts for one
``lineage_tracing/lineage`` experiment, writing into a per-experiment
``SAMPLE_DIR/fishtank/`` folder (sibling to ``positions/``, ``metadata/``,
``settings/``, ``data/``, ``analysis/`` inside the lineage acquisition's own
``SAMPLE_DIR``).

Fishtank is a different pipeline from MERlin (used for ``lineage_tracing/
merfish`` and the other variants, see ``merlin_config.py``): it needs a
``color_usage`` CSV (not ``data_organization``) and its own set of run
scripts (cellpose segmentation, spot detection/decoding, mosaics) instead of
merlin/slurm/snakemake configs.

Every schema/script in this module was verified against a real reference
experiment's files, read directly from
``...251225_LT027_saving_time\\fishtank\\`` (``params/color_usage_*.csv``,
``params/decoding_strategy_*.csv``, ``scripts/*.slurm``) — nothing here is a
guess. One likely bug in the reference ``decode_spots_ft.slurm`` (a dangling
``\\`` line-continuation before a blank line, which would glue the following
``echo`` onto the ``aggregate-polygons`` command) is NOT reproduced here.

Functions
---------
create_color_usage_csv        — fishtank's ``color_usage`` CSV (per-round/color
                                 tag table); the round-tag mapping is supplied
                                 by the caller — it is a manual, per-protocol
                                 choice, not derivable from round_info.csv.
create_decoding_strategy_csv  — fishtank's ``decoding_strategy`` CSV (per-target
                                 decode method + reference file)
resolve_fishtank_reference_dir — library-version -> shared reference dir
                                 (dispatch only, mirrors resolve_codebook_filename)
create_fishtank_folder_skeleton — pre-creates params/reference/scripts/output/log
                                 and copies the shared static utility scripts
FishtankScriptsSpec / create_fishtank_scripts — build every fishtank run
    script from a compact, fully-overridable spec (defaults copied verbatim
    from the reference experiment's scripts)
"""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

_COLORS_DEFAULT = ("650", "560", "488", "405")

# ── Reference-file dispatch (mirrors resolve_codebook_filename) ────────────────

_REFERENCE_DIR_BY_VERSION = {
    "v2": "v2",
}


def resolve_fishtank_reference_dir(lib_version: str) -> str:
    """Return the ``MERci/data/configs/fishtank/reference/`` subfolder name for
    *lib_version*, raising if unknown."""
    try:
        return _REFERENCE_DIR_BY_VERSION[lib_version]
    except KeyError:
        raise ValueError(
            f"No fishtank reference mapping for lib_version={lib_version!r}. "
            f"Known: {sorted(_REFERENCE_DIR_BY_VERSION)}"
        ) from None


# ── color_usage / decoding_strategy ─────────────────────────────────────────

def create_color_usage_csv(
    output_path: Path,
    rows:        Sequence[Dict[str, str]],
    colors:      Sequence[str] = _COLORS_DEFAULT,
) -> Path:
    """
    Write a fishtank ``color_usage`` CSV: header ``series,frames,{colors...}``
    then one row per imaging series.

    Format verified against ``color_usage_LT056s04.csv``/``color_usage_
    LT056s04_mf.csv`` (plain CSV, no padding spaces — unlike MERlin's codebook
    CSV format).

    Parameters
    ----------
    rows   : one dict per row, each with keys ``"series"``, ``"frames"``, and
             one key per entry in *colors* (e.g. ``"650"``) holding that
             color's round tag for this series — a plain label such as
             ``"r53"``, ``"beads"``, ``"DAPI"``, ``"empty"``, or ``""`` (blank
             = not imaged in this series). This mapping is supplied by the
             caller: it is a manual, per-protocol choice (which round images
             which target, in which color) that isn't derivable from
             MERci's own ``round_info.csv``/``round_bit_color_map.csv``.
    colors : column order for the per-color round-tag columns; default
             matches the reference example's ``650,560,488,405``.
    """
    header = ["series", "frames", *colors]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            missing = [c for c in ("series", "frames") if c not in row]
            if missing:
                raise ValueError(f"Row missing required key(s) {missing}: {row}")
            fh.write(",".join(str(row.get(col, "")) for col in header) + "\n")
    return output_path


def create_decoding_strategy_csv(
    output_path: Path,
    targets:     Sequence[Tuple[str, str, str, str]],
) -> Path:
    """
    Write a fishtank ``decoding_strategy`` CSV: header
    ``name,method,file,whitelist``, one row per decode target.

    Format verified against ``decoding_strategy_LT056s04.csv``.

    Parameters
    ----------
    targets : ``(name, method, reference_file, whitelist)`` rows, e.g.
              ``("intBC", "expectation_maximization", ".../intBC_codebook_v2.csv",
              ".../embryo_integration_whitelist.txt")``. *whitelist* may be
              ``""`` (not every target needs one — see the reference example's
              HEK3/EMX1/RNF2 rows, all ``logistic_regression`` with no whitelist).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("name,method,file,whitelist\n")
        for name, method, ref_file, whitelist in targets:
            fh.write(f"{name},{method},{ref_file},{whitelist}\n")
    return output_path


# ── Folder skeleton ──────────────────────────────────────────────────────────

_STATIC_SCRIPTS_SUBDIR = Path("scripts_static")


def create_fishtank_folder_skeleton(fishtank_dir: Path, merci_dir: Path) -> None:
    """
    Pre-create ``fishtank_dir``'s ``params/``, ``reference/``, ``scripts/``,
    ``output/``, ``log/`` subfolders (the slurm scripts' ``#SBATCH --output
    ../log/...`` needs ``log/`` to exist), and copy the shared static utility
    scripts (``MERci/data/configs/fishtank/scripts_static/*`` — currently
    ``plot_drift.py``, ``slurm_stats.sh``, ``check_segmentation.ipynb``) into
    ``fishtank_dir/scripts/``.

    Parameters
    ----------
    fishtank_dir : ``SAMPLE_DIR/fishtank`` for this experiment
    merci_dir    : the ``MERci/`` clone directory (for locating the shared
                   static scripts under ``data/configs/fishtank/``)
    """
    fishtank_dir = Path(fishtank_dir)
    for sub in ("params", "reference", "scripts", "output", "log"):
        (fishtank_dir / sub).mkdir(parents=True, exist_ok=True)

    static_dir = Path(merci_dir) / "data" / "configs" / "fishtank" / _STATIC_SCRIPTS_SUBDIR
    for src in static_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, fishtank_dir / "scripts" / src.name)


def copy_fishtank_reference_files(lib_version: str, merci_dir: Path, fishtank_dir: Path) -> List[Path]:
    """
    Copy the shared reference files for *lib_version* (codebook/weights/
    whitelist, dispatched via :func:`resolve_fishtank_reference_dir`) from
    ``MERci/data/configs/fishtank/reference/`` into ``fishtank_dir/reference/``.

    Returns the list of destination paths.
    """
    src_dir = Path(merci_dir) / "data" / "configs" / "fishtank" / "reference" / resolve_fishtank_reference_dir(lib_version)
    dst_dir = Path(fishtank_dir) / "reference"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in src_dir.iterdir():
        if src.is_file():
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


# ── Fishtank run scripts ─────────────────────────────────────────────────────
# Every default below is copied verbatim from the reference experiment's
# scripts (cellpose_ft.slurm, cellpose_ft_mf.slurm, detect_spots_ft.slurm,
# decode_spots_ft.slurm, generate_mosaics.slurm).

_CELLPOSE_LINEAGE_DEFAULTS = {
    "job_name": "cellpose", "mem": "16gb", "time": "0-00:10:00",
    "gres": "gpu:1", "partition": "zhuang_gpu", "array_concurrency": 20,
    "output_subdir": "cellpose_polygons",
    "color_usage": None,   # filled in by create_fishtank_scripts
    "model": "nuclei", "channels": "DAPI,DAPI", "downsample": 4,
    "min_size": 5000, "do_3D": "True", "gpu": "True",
}
_CELLPOSE_MERFISH_DEFAULTS = {
    **_CELLPOSE_LINEAGE_DEFAULTS,
    "job_name": "cellpose_mf", "mem": "12gb",
    "output_subdir": "cellpose_polygons_mf",
}
_DETECT_SPOTS_DEFAULTS = {
    "mem": "75gb", "time": "0-01:20:00", "cpus_per_task": 4,
    "partition": "zhuang,sapphire,shared",
    "common_bits": "r52,r53", "reg_bit": "beads", "reg_z_slice": 0,
    "reg_min_intensity": 10, "filter": "unsharp_mask", "exclude_bits": "empty",
    "filter_args": "sigma=10", "spot_min_sigma": 2, "spot_max_sigma": 10,
    "spot_threshold": 100, "spot_radius": 5, "scale_factor": 0.09,
}
_AGGREGATE_POLYGONS_DEFAULTS = {"min_size": 100, "z_column": "global_z", "save_union": "True"}
_DECODE_SPOTS_DEFAULTS = {
    "mem": "128gb", "time": "00-08:00:00", "cpus_per_task": 16,
    "partition": "zhuang,sapphire,shared",
    "normalize_colors": "True", "max_dist": 2, "save_intensities": "True",
    "filter_output": "True",
}
_ASSIGN_SPOTS_DEFAULTS = {"max_dist": 2, "z_column": "global_z", "cell_fill": 0}
_MOSAIC_DEFAULTS = {
    "colors": "405", "z": 3, "flip_vertical": "False", "flip_horizontal": "False",
    "downsample": 8, "scale_factor": 0.09,
}


@dataclass
class FishtankScriptsSpec:
    """
    Compact description of every fishtank run script's parameters — every
    field overridable, defaulting to the reference experiment's verified
    values (mirrors ``merlin_config.MerlinAnalysisSpec``).

    ``n_fovs_lineage``/``n_fovs_merfish`` set each per-FOV array job's
    ``--array=0-{n_fovs-1}%{array_concurrency}`` range (the two acquisitions
    are independent imaging sessions and can have different FOV counts —
    ``cellpose_lineage``/``detect_spots`` use ``n_fovs_lineage``,
    ``cellpose_merfish`` uses ``n_fovs_merfish``); the per-job dicts below are
    merged OVER the verified defaults (only override what differs for this
    experiment).
    """
    n_fovs_lineage:       int
    n_fovs_merfish:       int
    cellpose_lineage:     Dict[str, Any] = field(default_factory=dict)
    cellpose_merfish:     Dict[str, Any] = field(default_factory=dict)
    detect_spots:         Dict[str, Any] = field(default_factory=dict)
    aggregate_polygons:   Dict[str, Any] = field(default_factory=dict)
    decode_spots:         Dict[str, Any] = field(default_factory=dict)
    assign_spots:         Dict[str, Any] = field(default_factory=dict)
    mosaic_lineage:       Dict[str, Any] = field(default_factory=dict)
    mosaic_merfish:       Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> "FishtankScriptsSpec":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(**data)


def _sbatch_header(job_name: str, mem: str, time: str, output_log: str,
                    partition: str, cpus_per_task: int = 1,
                    gres: Optional[str] = None, array: Optional[str] = None) -> str:
    lines = [
        "#!/bin/bash",
        "# Configuration values for SLURM job submission.",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
    ]
    if gres is not None:
        lines.append(f"#SBATCH --gres={gres}")
    lines.append(f"#SBATCH --partition={partition}")
    lines.append(f"#SBATCH --output {output_log}")
    if array is not None:
        lines.append(f"#SBATCH --array={array}")
    lines.append("")
    lines.append("module load python")
    lines.append("source activate fishtank_env")
    lines.append("")
    return "\n".join(lines)


def _write_script(output_path: Path, text: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text.rstrip("\n") + "\n")


def _create_cellpose_script(output_path: Path, params: dict, input_dir: str,
                             color_usage: str, n_fovs: int) -> Path:
    p = {**_CELLPOSE_LINEAGE_DEFAULTS, **params}
    header = _sbatch_header(
        job_name=p["job_name"], mem=p["mem"], time=p["time"],
        output_log="../log/%x_%j.out", partition=p["partition"],
        gres=p["gres"], array=f"0-{n_fovs - 1}%{p['array_concurrency']}",
    )
    body = (
        "fishtank cellpose \\\n"
        "        --fov $SLURM_ARRAY_TASK_ID \\\n"
        f"        --input {input_dir} \\\n"
        f"        --output ../output/{p['output_subdir']}/ \\\n"
        "        --file_pattern {series} \\\n"
        f"        --color_usage {color_usage} \\\n"
        f"        --model {p['model']} \\\n"
        f"        --channels {p['channels']} \\\n"
        f"        --downsample {p['downsample']} \\\n"
        f"        --min_size {p['min_size']} \\\n"
        f"        --do_3D {p['do_3D']} \\\n"
        f"        --gpu {p['gpu']}\n"
    )
    _write_script(output_path, header + "\n" + body)
    return Path(output_path)


def _create_detect_spots_script(output_path: Path, params: dict, input_dir: str,
                                 color_usage: str, ref_series: str, n_fovs: int,
                                 experiment_label: str) -> Path:
    p = {**_DETECT_SPOTS_DEFAULTS, **params}
    array_concurrency = p.get("array_concurrency", 20)
    header = _sbatch_header(
        job_name="detect_spots", mem=p["mem"], time=p["time"],
        output_log="../log/%x_%j.out", partition=p["partition"],
        cpus_per_task=p.get("cpus_per_task", 4),
        array=f"0-{n_fovs - 1}%{array_concurrency} ## Adjust this range based on the number of FOVs to process.",
    )
    body = (
        "fishtank detect-spots \\\n"
        "    --fov  ${SLURM_ARRAY_TASK_ID} \\\n"
        f"    --input {input_dir} \\\n"
        "    --output ../output/spots/ \\\n"
        f"    --color_usage {color_usage} \\\n"
        f"    --common_bits {p['common_bits']} \\\n"
        f"    --reg_bit {p['reg_bit']} \\\n"
        f"    --reg_z_slice {p['reg_z_slice']} \\\n"
        f"    --reg_min_intensity {p['reg_min_intensity']} \\\n"
        f"    --ref_series {ref_series} \\\n"
        "    --file_pattern {series} \\\n"
        f"    --filter {p['filter']} \\\n"
        f"    --exclude_bits {p['exclude_bits']} \\\n"
        f"    --filter_args {p['filter_args']} \\\n"
        f"    --spot_min_sigma {p['spot_min_sigma']} \\\n"
        f"    --spot_max_sigma {p['spot_max_sigma']} \\\n"
        f"    --spot_threshold {p['spot_threshold']} \\\n"
        f"    --spot_radius {p['spot_radius']} \\\n"
        f"    --scale_factor {p['scale_factor']}\n"
    )
    qc_guard = (
        "\n"
        "# --- Drift quality diagnostics -------------------------------------------------\n"
        "# detect-spots is a SLURM array (one task per FOV), so this guard runs the QC plot\n"
        "# exactly once, when the LAST FOV finishes: it fires only after every FOV has written\n"
        "# its output, and an atomic mkdir lock (keyed to this array job) prevents duplicates.\n"
        "EXPECTED=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))\n"
        "DONE=$(ls ../output/spots/channels_*.csv 2>/dev/null | wc -l)\n"
        'LOCK="../output/spots/.drift_qc_${SLURM_ARRAY_JOB_ID:-single}.lock"\n'
        'if [ "$DONE" -ge "$EXPECTED" ] && mkdir "$LOCK" 2>/dev/null; then\n'
        '    echo "All $DONE/$EXPECTED FOVs complete -> generating drift QC figure"\n'
        "    # --out defaults to <input>/drift_qc.png (../output/spots/drift_qc.png)\n"
        "    python ./plot_drift.py \\\n"
        "        --input ../output/spots \\\n"
        f'        --title "{experiment_label} detect-spots drift QC"\n'
        "fi\n"
    )
    _write_script(output_path, header + "\n" + body + qc_guard)
    return Path(output_path)


def _create_decode_spots_script(output_path: Path, decode_params: dict,
                                 aggregate_params: dict, assign_params: dict,
                                 color_usage: str, decoding_strategy: str) -> Path:
    dp = {**_DECODE_SPOTS_DEFAULTS, **decode_params}
    ap = {**_AGGREGATE_POLYGONS_DEFAULTS, **aggregate_params}
    sp = {**_ASSIGN_SPOTS_DEFAULTS, **assign_params}
    header = _sbatch_header(
        job_name="decode_spots", mem=dp["mem"], time=dp["time"],
        output_log="../log/%x_%j.out", partition=dp["partition"],
        cpus_per_task=dp.get("cpus_per_task", 16),
    )
    body = (
        'echo "Aggregating polygons"\n'
        "fishtank aggregate-polygons \\\n"
        "    -i ../output/cellpose_polygons/ \\\n"
        "    -o ../output/cellpose_polygons/all_polygons.json \\\n"
        f"    --min_size {ap['min_size']} \\\n"
        f'    --z_column "{ap["z_column"]}" \\\n'
        f"    --save_union {ap['save_union']}\n"
        "\n"
        'echo "Decoding spots"\n'
        "fishtank decode-spots \\\n"
        "    --input ../output/spots/ \\\n"
        "    --output ../output/decoded/decoded_spots.csv \\\n"
        f"    --color_usage {color_usage} \\\n"
        f"    --normalize_colors {dp['normalize_colors']} \\\n"
        f"    --max_dist {dp['max_dist']} \\\n"
        f"    --save_intensities {dp['save_intensities']} \\\n"
        f"    --filter_output {dp['filter_output']} \\\n"
        f"    --strategy {decoding_strategy}\n"
        "\n"
        'echo "Assigning spots to polygons"\n'
        "fishtank assign-spots \\\n"
        "    -i ../output/decoded/decoded_spots.csv \\\n"
        "    -p ../output/cellpose_polygons/all_polygons.json \\\n"
        "    -o ../output/decoded/decoded_spots.csv \\\n"
        f"    --max_dist {sp['max_dist']} \\\n"
        f"    --z_column {sp['z_column']} \\\n"
        f"    --cell_fill {sp['cell_fill']}\n"
    )
    _write_script(output_path, header + "\n" + body)
    return Path(output_path)


def _create_mosaic_script(output_path: Path, lineage_params: dict, merfish_params: dict,
                           mosaic_lineage_input: str, mosaic_merfish_input: str,
                           lineage_positions: str, merfish_positions: str) -> Path:
    lp = {**_MOSAIC_DEFAULTS, **lineage_params}
    mp = {**_MOSAIC_DEFAULTS, **merfish_params}
    header = _sbatch_header(
        job_name="mosaics", mem="64gb", time="0-04:00:00",
        output_log="../log/%x_%j.out", partition="zhuang,sapphire,shared",
        cpus_per_task=4,
    )

    def _mosaic_call(input_dir, out_name, positions, p):
        file_pattern = p.get("file_pattern", "disk_650f141_560f141_488f141_405f141_5micron_{fov:03d}.tif")
        return (
            "fishtank mosaic \\\n"
            f"    --i {input_dir} \\\n"
            f"    --o ../output/mosaics/{out_name} \\\n"
            f"    --file_pattern {file_pattern} \\\n"
            f"    --positions {positions} \\\n"
            f"    --colors {p['colors']} \\\n"
            f"    --z {p['z']} \\\n"
            f"    --flip_vertical {p['flip_vertical']} \\\n"
            f"    --flip_horizontal {p['flip_horizontal']} \\\n"
            f"    --downsample {p['downsample']} \\\n"
            f"    --scale_factor {p['scale_factor']}\n"
        )

    body = (
        _mosaic_call(mosaic_lineage_input, lp.get("output_name", "mosaic_60x_dapi_z30_lt.tif"),
                     lineage_positions, lp)
        + "\n"
        + _mosaic_call(mosaic_merfish_input, mp.get("output_name", "mosaic_60x_dapi_z30_mf.tif"),
                       merfish_positions, mp)
    )
    _write_script(output_path, header + "\n" + body)
    return Path(output_path)


def create_fishtank_scripts(
    spec:                FishtankScriptsSpec,
    fishtank_dir:        Path,
    experiment_label:    str,
    lineage_input:       str,
    merfish_input:       str,
    mosaic_lineage_input: str,
    mosaic_merfish_input: str,
    lineage_positions:   str,
    merfish_positions:   str,
    color_usage_lineage: str,
    color_usage_merfish: str,
    decoding_strategy:   str,
    ref_series:          str,
) -> Dict[str, Path]:
    """
    Write every fishtank run script into ``fishtank_dir/scripts/``:
    ``cellpose_ft.slurm``, ``cellpose_ft_mf.slurm``, ``detect_spots_ft.slurm``
    (incl. the drift-QC trigger calling ``plot_drift.py``), ``decode_spots_ft.slurm``,
    ``generate_mosaics.slurm`` — each verified field-for-field against the
    reference experiment's real scripts.

    Parameters
    ----------
    experiment_label     : used in the drift-QC plot title (e.g. ``"LT056s04"``)
    lineage_input/merfish_input : ``--input`` roots passed to ``cellpose``/
                            ``detect-spots`` (raw data directories for each
                            acquisition; ``--file_pattern {series}`` supplies
                            the subfolder+filename pattern from color_usage)
    mosaic_lineage_input/mosaic_merfish_input : ``--i`` values for
                            ``fishtank mosaic`` — unlike above, this must be
                            the FULL directory containing the actual image
                            files (mosaic has no ``--file_pattern {series}``
                            templating), e.g. ``f"{lineage_input}/data"``
    lineage_positions/merfish_positions : ``--positions`` values for the two
                            ``fishtank mosaic`` calls (plain strings — e.g.
                            ``"../../positions/positions_lineage.txt"``; the
                            caller resolves what these should point to)
    color_usage_lineage/color_usage_merfish : ``--color_usage`` values (paths
                            relative to ``fishtank/scripts/``, e.g.
                            ``"../params/color_usage_{sample}.csv"``)
    decoding_strategy    : ``--strategy`` value for ``decode-spots``
    ref_series            : ``--ref_series`` value for ``detect-spots`` — the
                            series pattern used as the fiducial/registration
                            reference (the reference example's "cells"-like row)

    Returns
    -------
    dict mapping script name -> written Path
    """
    fishtank_dir = Path(fishtank_dir)
    scripts_dir = fishtank_dir / "scripts"
    written = {}

    written["cellpose_ft.slurm"] = _create_cellpose_script(
        scripts_dir / "cellpose_ft.slurm", spec.cellpose_lineage,
        lineage_input, color_usage_lineage, spec.n_fovs_lineage,
    )
    written["cellpose_ft_mf.slurm"] = _create_cellpose_script(
        scripts_dir / "cellpose_ft_mf.slurm",
        {**_CELLPOSE_MERFISH_DEFAULTS, **spec.cellpose_merfish},
        merfish_input, color_usage_merfish, spec.n_fovs_merfish,
    )
    written["detect_spots_ft.slurm"] = _create_detect_spots_script(
        scripts_dir / "detect_spots_ft.slurm", spec.detect_spots,
        lineage_input, color_usage_lineage, ref_series, spec.n_fovs_lineage, experiment_label,
    )
    written["decode_spots_ft.slurm"] = _create_decode_spots_script(
        scripts_dir / "decode_spots_ft.slurm", spec.decode_spots,
        spec.aggregate_polygons, spec.assign_spots,
        color_usage_lineage, decoding_strategy,
    )
    written["generate_mosaics.slurm"] = _create_mosaic_script(
        scripts_dir / "generate_mosaics.slurm", spec.mosaic_lineage, spec.mosaic_merfish,
        mosaic_lineage_input, mosaic_merfish_input, lineage_positions, merfish_positions,
    )
    return written
