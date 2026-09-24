# MERci/acquisition/cluster_submit.py
"""
Build and submit SLURM array jobs that run MERci's own FOV/round QC analysis
(``analyze_file`` / ``build_round_mosaics``) on a cluster, reading data that
has already landed on cluster storage (see
``07_cluster_submit_analysis.ipynb``).

Follows the same sbatch conventions as ``acquisition/fishtank_config.py``'s
``_sbatch_header`` (this is FOV-parallel array work, like fishtank's
cellpose/detect-spots jobs -- not MERlin's single-orchestrator-job
convention, see ``acquisition/merlin_config.py``): ``module load python`` +
``source activate <env>``, a real ``#SBATCH --array=0-{n-1}%{concurrency}``.

No ``pip install`` needed on the cluster -- the generated scripts invoke the
standalone CLI scripts (``analysis/cli_analyze_fov.py`` /
``cli_build_round_mosaic.py``) by their absolute path under the cluster's
own ``MERci/`` clone (this module runs from that same clone, so it knows its
own path via ``__file__``), never ``python -m MERci...``.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# .../MERci/src/MERci/acquisition/cluster_submit.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
_CLI_ANALYZE_FOV            = _MERCI_SRC / "MERci" / "analysis" / "cli_analyze_fov.py"
_CLI_BUILD_ROUND_MOSAIC     = _MERCI_SRC / "MERci" / "analysis" / "cli_build_round_mosaic.py"
_CLI_COMPUTE_TEXTURE_STATS  = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_texture_stats.py"
_CLI_TPC_MARGIN_THUMBNAILS  = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_tpc_margin_thumbnails.py"
_CLI_GIF_FRAME_THUMBNAILS   = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_gif_frame_thumbnails.py"
_CLI_CHANNEL_COUNTERS       = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_channel_counters.py"
_CLI_FOV_PROJECTIONS        = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_fov_projections.py"
_CLI_FOV_ELEVATION          = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_fov_elevation.py"
_CLI_FOV_COMPLETENESS       = _MERCI_SRC / "MERci" / "analysis" / "cli_check_fov_completeness.py"
_CLI_INTENSITY_PERCENTILES  = _MERCI_SRC / "MERci" / "analysis" / "cli_measure_intensity_percentiles.py"

_DEFAULT_PARTITION = "zhuang,sapphire,shared"
_DEFAULT_CONDA_ENV = "merci_env"


def _sbatch_header(
    job_name:      str,
    mem:           str,
    time:          str,
    output_log:    str,
    partition:     str = _DEFAULT_PARTITION,
    cpus_per_task: int = 1,
    array:         Optional[str] = None,
    conda_env:     str = _DEFAULT_CONDA_ENV,
    gres:          Optional[str] = None,
) -> str:
    """``#SBATCH`` lines plus ``module load python`` / ``source activate
    <conda_env>`` (also used by ``fishtank_config``)."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
    ]
    if gres is not None:
        lines.append(f"#SBATCH --gres={gres}")
    lines += [
        f"#SBATCH --partition={partition}",
        f"#SBATCH --output={output_log}",
    ]
    if array is not None:
        lines.append(f"#SBATCH --array={array}")
    lines += ["", "module load python", f"source activate {conda_env}", ""]
    return "\n".join(lines)


def _write_script(output_path: Path, text: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text.rstrip("\n") + "\n")
    return output_path


def _csv(values) -> str:
    return ",".join(str(v) for v in values)


def _orientation_flags(orientation: dict) -> str:
    """``--flip-horizontal --transpose``-style flags for the True entries."""
    return " ".join(f"--{flag.replace('_', '-')}" for flag, on in orientation.items() if on)


def _job_script(
    cli_path: Path, arg_lines: List[str], sample_dir: Path, output_path: Path, array: Optional[str],
    mem: str, time: str, partition: str, conda_env: str, job_name: str,
) -> Path:
    """Write an sbatch script running ``python <cli_path> <arg_lines...>``
    (one line per argument group; empty lines are dropped), logging to
    ``<sample_dir>/analysis/logs``."""
    header = _sbatch_header(
        job_name=job_name, mem=mem, time=time,
        output_log=str(Path(sample_dir) / "analysis" / "logs" / "%x_%A_%a.out"),
        partition=partition, array=array, conda_env=conda_env,
    )
    body = f"python {cli_path}" + "".join(f" \\\n    {line}" for line in arg_lines if line) + "\n"
    return _write_script(output_path, header + "\n" + body)


def _array(n_pending: int, concurrency: int) -> str:
    return f"0-{n_pending - 1}%{concurrency}"


# Each build_*_script below writes an sbatch array job running one
# cli_*.py script once per manifest entry ($SLURM_ARRAY_TASK_ID selects the
# entry); see that script's own docstring for its manifest format.

def build_fov_array_script(
    sample_dir:         Path,
    manifest_path:      Path,
    n_pending:          int,
    output_path:        Path,
    array_concurrency:  int = 50,
    mem:                str = "8gb",
    time:               str = "02:00:00",
    partition:          str = _DEFAULT_PARTITION,
    conda_env:          str = _DEFAULT_CONDA_ENV,
    job_name:           str = "merci_fov",
) -> Path:
    """``cli_analyze_fov.py`` per pending image file (manifest: one path per line)."""
    return _job_script(
        _CLI_ANALYZE_FOV, [f"--sample-dir {sample_dir}", f"--manifest {manifest_path}"],
        sample_dir, output_path, _array(n_pending, array_concurrency),
        mem, time, partition, conda_env, job_name,
    )


def build_round_mosaic_script(
    sample_dir:    Path,
    manifest_path: Path,
    n_pending:     int,
    output_path:   Path,
    mem:           str = "8gb",
    time:          str = "00:30:00",
    partition:     str = _DEFAULT_PARTITION,
    conda_env:     str = _DEFAULT_CONDA_ENV,
    job_name:      str = "merci_mosaic",
) -> Path:
    """
    ``cli_build_round_mosaic.py`` for the round(s) in *manifest_path* (one
    round id per line): an array job for several rounds, a plain job
    (``--round-id``) for exactly one.
    """
    if n_pending > 1:
        round_arg, array = f"--manifest {manifest_path}", f"0-{n_pending - 1}"
    else:
        round_id = int(Path(manifest_path).read_text(encoding="utf-8").split()[0])
        round_arg, array = f"--round-id {round_id}", None
    return _job_script(
        _CLI_BUILD_ROUND_MOSAIC, [f"--sample-dir {sample_dir}", round_arg],
        sample_dir, output_path, array, mem, time, partition, conda_env, job_name,
    )


def build_texture_stats_array_script(
    sample_dir:         Path,
    manifest_path:      Path,
    output_dir:         Path,
    frame_indices,
    n_pending:          int,
    output_path:        Path,
    sigma:              float = 1.0,
    array_concurrency:  int = 50,
    mem:                str = "4gb",
    time:               str = "00:30:00",
    partition:          str = _DEFAULT_PARTITION,
    conda_env:          str = _DEFAULT_CONDA_ENV,
    job_name:           str = "merci_texture",
) -> Path:
    """``cli_compute_texture_stats.py`` per pending FOV (re-reads each FOV's z-stack)."""
    return _job_script(
        _CLI_COMPUTE_TEXTURE_STATS,
        [f"--manifest {manifest_path}", f"--output-dir {output_dir}",
         f"--frame-indices {_csv(frame_indices)}", f"--sigma {sigma}"],
        sample_dir, output_path, _array(n_pending, array_concurrency),
        mem, time, partition, conda_env, job_name,
    )


def build_tpc_margin_array_script(
    sample_dir:         Path,
    manifest_path:      Path,
    output_dir:         Path,
    frame_indices,
    z_um_values,
    margins,
    thumbnail_size,
    orientation:        dict,
    n_pending:          int,
    output_path:        Path,
    array_concurrency:  int = 50,
    mem:                str = "4gb",
    time:               str = "00:15:00",
    partition:          str = _DEFAULT_PARTITION,
    conda_env:          str = _DEFAULT_CONDA_ENV,
    job_name:           str = "merci_tpc_margin",
) -> Path:
    """``cli_compute_tpc_margin_thumbnails.py`` per pending FOV (one read covers every margin)."""
    tw, th = thumbnail_size
    return _job_script(
        _CLI_TPC_MARGIN_THUMBNAILS,
        [f"--manifest {manifest_path}", f"--output-dir {output_dir}",
         f"--frame-indices {_csv(frame_indices)}", f"--z-um-values {_csv(z_um_values)}",
         f"--margins {_csv(margins)}", f"--thumbnail-width {tw} --thumbnail-height {th}",
         _orientation_flags(orientation)],
        sample_dir, output_path, _array(n_pending, array_concurrency),
        mem, time, partition, conda_env, job_name,
    )


def build_fov_projections_array_script(
    sample_dir:         Path,
    manifest_path:      Path,
    output_dir:         Path,
    frame_indices,
    statistics,
    orientation:        dict,
    n_pending:          int,
    output_path:        Path,
    array_concurrency:  int = 50,
    mem:                str = "8gb",
    time:               str = "00:15:00",
    partition:          str = _DEFAULT_PARTITION,
    conda_env:          str = _DEFAULT_CONDA_ENV,
    job_name:           str = "merci_fov_proj",
) -> Path:
    """``cli_compute_fov_projections.py`` per pending FOV: every requested
    per-pixel z projection from one read of the stack."""
    return _job_script(
        _CLI_FOV_PROJECTIONS,
        [f"--manifest {manifest_path}", f"--output-dir {output_dir}",
         f"--frame-indices {_csv(frame_indices)}", f"--statistics {_csv(statistics)}",
         _orientation_flags(orientation)],
        sample_dir, output_path, _array(n_pending, array_concurrency),
        mem, time, partition, conda_env, job_name,
    )


def build_gif_frames_array_script(
    sample_dir:         Path,
    manifest_path:      Path,
    output_dir:         Path,
    z_positions,
    frame_indices,
    thumbnail_size,
    orientation:        dict,
    n_pending:          int,
    output_path:        Path,
    array_concurrency:  int = 50,
    mem:                str = "4gb",
    time:               str = "00:20:00",
    partition:          str = _DEFAULT_PARTITION,
    conda_env:          str = _DEFAULT_CONDA_ENV,
    job_name:           str = "merci_gif_frames",
) -> Path:
    """``cli_compute_gif_frame_thumbnails.py`` per pending FOV (the selected z-steps only)."""
    tw, th = thumbnail_size
    return _job_script(
        _CLI_GIF_FRAME_THUMBNAILS,
        [f"--manifest {manifest_path}", f"--output-dir {output_dir}",
         f"--z-positions {_csv(z_positions)}", f"--frame-indices {_csv(frame_indices)}",
         f"--thumbnail-width {tw} --thumbnail-height {th}", _orientation_flags(orientation)],
        sample_dir, output_path, _array(n_pending, array_concurrency),
        mem, time, partition, conda_env, job_name,
    )


def build_fov_elevation_array_script(
    sample_dir:         Path,
    manifest_path:      Path,
    output_dir:         Path,
    frame_indices,
    z_um_values,
    ffc_field_path:     Path,
    threshold:          float,
    downsample_factor:  int,
    orientation:        dict,
    n_pending:          int,
    output_path:        Path,
    array_concurrency:  int = 50,
    mem:                str = "8gb",
    time:               str = "00:15:00",
    partition:          str = _DEFAULT_PARTITION,
    conda_env:          str = _DEFAULT_CONDA_ENV,
    job_name:           str = "merci_elevation",
) -> Path:
    """``cli_compute_fov_elevation.py`` per pending FOV
    (``analysis.elevation.compute_fov_elevation`` at full-grid scale)."""
    return _job_script(
        _CLI_FOV_ELEVATION,
        [f"--manifest {manifest_path}", f"--output-dir {output_dir}",
         f"--ffc-field-path {ffc_field_path}", f"--threshold {threshold}",
         f"--downsample-factor {downsample_factor}", f"--frame-indices {_csv(frame_indices)}",
         f"--z-um-values {_csv(z_um_values)}", _orientation_flags(orientation)],
        sample_dir, output_path, _array(n_pending, array_concurrency),
        mem, time, partition, conda_env, job_name,
    )


def build_channel_counters_array_script(
    sample_dir:         Path,
    manifest_path:      Path,
    output_dir:         Path,
    frame_indices,
    z_um_values,
    n_pending:          int,
    output_path:        Path,
    array_concurrency:  int = 50,
    mem:                str = "4gb",
    time:               str = "00:20:00",
    partition:          str = _DEFAULT_PARTITION,
    conda_env:          str = _DEFAULT_CONDA_ENV,
    job_name:           str = "merci_channel_counters",
) -> Path:
    """``cli_compute_channel_counters.py`` per pending FOV (one channel z-sweep each)."""
    return _job_script(
        _CLI_CHANNEL_COUNTERS,
        [f"--manifest {manifest_path}", f"--output-dir {output_dir}",
         f"--frame-indices {_csv(frame_indices)}", f"--z-um-values {_csv(z_um_values)}"],
        sample_dir, output_path, _array(n_pending, array_concurrency),
        mem, time, partition, conda_env, job_name,
    )


def build_fov_completeness_array_script(
    sample_dir:         Path,
    manifest_path:      Path,
    output_dir:         Path,
    round_info_csv:     Path,
    positions_txt:      Path,
    data_dir:           Path,
    image_suffix:       str,
    round_ids,
    n_pending:          int,
    output_path:        Path,
    array_concurrency:  int = 50,
    mem:                str = "2gb",
    time:               str = "00:15:00",
    partition:          str = _DEFAULT_PARTITION,
    conda_env:          str = _DEFAULT_CONDA_ENV,
    job_name:           str = "merci_completeness",
) -> Path:
    """``cli_check_fov_completeness.py`` per pending FOV, checking all of
    *round_ids* (``after_imaging/10_check_fov_completeness.ipynb``)."""
    return _job_script(
        _CLI_FOV_COMPLETENESS,
        [f"--round-info-csv {round_info_csv}", f"--positions-txt {positions_txt}",
         f"--data-dir {data_dir}", f"--image-suffix {image_suffix}",
         f"--round-ids {_csv(round_ids)}", f"--manifest {manifest_path}",
         f"--output-dir {output_dir}"],
        sample_dir, output_path, _array(n_pending, array_concurrency),
        mem, time, partition, conda_env, job_name,
    )


def build_intensity_percentiles_array_script(
    sample_dir:         Path,
    manifest_path:      Path,
    n_pending:          int,
    output_path:        Path,
    percentiles:        tuple      = (25, 50, 75, 95),
    array_concurrency:  int = 50,
    mem:                str = "1gb",
    time:               str = "00:10:00",
    partition:          str = _DEFAULT_PARTITION,
    conda_env:          str = _DEFAULT_CONDA_ENV,
    job_name:           str = "merci_intensity_pctl",
) -> Path:
    """
    ``cli_measure_intensity_percentiles.py`` per image file
    (``after_imaging/12_measure_intensity_percentiles.ipynb``). Defaults are
    sized from a real run on 215-frame 2304x2304 FOVs: about 50 s and 260 MB
    per file.
    """
    return _job_script(
        _CLI_INTENSITY_PERCENTILES,
        [f"--manifest {manifest_path}", f"--percentiles {_csv(percentiles)}"],
        sample_dir, output_path, _array(n_pending, array_concurrency),
        mem, time, partition, conda_env, job_name,
    )


_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


def submit_sbatch(script_path: Path) -> Optional[int]:
    """
    Submit *script_path* via ``sbatch``. Returns the job id, or ``None``
    (after logging the failure) if submission failed -- never raises, so a
    submission hiccup doesn't crash a polling loop.
    """
    try:
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    except FileNotFoundError:
        log.error("sbatch not found on PATH -- are you running this on a cluster login node?")
        return None
    if result.returncode != 0:
        log.error("sbatch failed for %s (exit %d):\n%s", script_path, result.returncode, result.stderr)
        return None
    m = _SBATCH_JOB_ID_RE.search(result.stdout)
    if not m:
        log.error("sbatch succeeded but job id not found in output: %r", result.stdout)
        return None
    job_id = int(m.group(1))
    log.info("Submitted %s as job %d.", script_path, job_id)
    return job_id


# States in which a job is still occupying the queue/running -- not worth
# resubmitting work for yet. Anything else (COMPLETED, FAILED, TIMEOUT,
# CANCELLED, NODE_FAIL, ...) means the queue slot is free again.
_ACTIVE_STATES = {"PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING", "COMPLETING"}


def job_state(job_id: int) -> Optional[str]:
    """
    Return the SLURM state of *job_id* (e.g. ``"PENDING"``, ``"RUNNING"``,
    ``"COMPLETED"``, ``"FAILED"``), via the same ``sacct`` tool
    ``data/configs/fishtank/scripts_static/slurm_stats.sh`` already uses for
    job-resource auditing. Returns ``None`` if ``sacct`` couldn't be run or
    the job isn't known (yet).
    """
    try:
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=State", "--noheader", "--parsable2"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        log.error("sacct not found on PATH -- are you running this on a cluster login node?")
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first_line = result.stdout.strip().splitlines()[0].strip()
    return first_line.split()[0] if first_line else None


def is_job_active(job_id: int) -> bool:
    """True if *job_id* is still pending/running (i.e. not worth resubmitting
    work for yet)."""
    state = job_state(job_id)
    return state is not None and state.upper() in _ACTIVE_STATES


# ── Notebook-level driving functions (07_cluster_submit_analysis.ipynb) ──────

def submit_pending_fov_analysis(
    config, meta, tracker, sample_dir: Path, manifests_dir: Path,
    array_concurrency: int, mem: str, time: str, partition: str, conda_env: str,
    dry_run: bool = False,
) -> list:
    """
    Submit one FOV-analysis array job (:func:`build_fov_array_script`) per
    round that has pending FOVs and no still-active previous submission.
    Returns ``[(round_id, job_id, n_fovs), ...]``.
    """
    pending_all = set(tracker.pending_fov_files(meta.all_expected_files()))
    submitted = []
    for rid in meta.valid_round_ids():
        round_files = sorted(f for f in meta.files_for_round(rid) if f in pending_all)
        if not round_files:
            continue

        prev_job = tracker.fov_analysis_submitted_job_id(rid)
        if prev_job is not None and is_job_active(prev_job):
            log.info("Round %d: FOV job %s still active -- skipping.", rid, prev_job)
            continue

        manifest_path = manifests_dir / f"pending_fovs_round{rid:03d}.txt"
        manifest_path.write_text("\n".join(str(f) for f in round_files) + "\n", encoding="utf-8", newline="\n")
        script_path = manifests_dir / f"fov_array_round{rid:03d}.sh"
        build_fov_array_script(
            sample_dir=sample_dir, manifest_path=manifest_path, n_pending=len(round_files),
            output_path=script_path, array_concurrency=array_concurrency,
            mem=mem, time=time, partition=partition, conda_env=conda_env,
        )
        if dry_run:
            log.info("[dry run] would submit %s  (%d FOV(s))", script_path, len(round_files))
            continue
        job_id = submit_sbatch(script_path)
        if job_id is not None:
            tracker.mark_fov_analysis_submitted(rid, job_id)
            submitted.append((rid, job_id, len(round_files)))
    return submitted


def submit_pending_round_mosaics(
    config, meta, tracker, sample_dir: Path, manifests_dir: Path,
    mem: str, time: str, partition: str, conda_env: str,
    dry_run: bool = False,
) -> list:
    """
    Submit one job (:func:`build_round_mosaic_script`; array if >1 round)
    building mosaics for every round whose FOVs are all done but has no
    mosaic yet and no still-active previous submission. Returns
    ``[(round_ids, job_id)]`` or ``[]``.
    """
    pending_rounds = [
        rid for rid in tracker.pending_rounds(meta.valid_round_ids(), meta, config.fov_subset)
        if not (
            tracker.is_round_mosaic_submitted(rid)
            and (job_id := tracker.round_mosaic_submitted_job_id(rid)) is not None
            and is_job_active(job_id)
        )
    ]
    if not pending_rounds:
        return []

    manifest_path = manifests_dir / "pending_rounds.txt"
    manifest_path.write_text("\n".join(str(r) for r in pending_rounds) + "\n", encoding="utf-8", newline="\n")
    script_path = manifests_dir / "round_mosaic.sh"
    build_round_mosaic_script(
        sample_dir=sample_dir, manifest_path=manifest_path, n_pending=len(pending_rounds),
        output_path=script_path, mem=mem, time=time,
        partition=partition, conda_env=conda_env,
    )
    if dry_run:
        log.info("[dry run] would submit %s  (rounds %s)", script_path, pending_rounds)
        return []
    job_id = submit_sbatch(script_path)
    if job_id is None:
        return []
    for rid in pending_rounds:
        tracker.mark_round_mosaic_submitted(rid, job_id)
    return [(pending_rounds, job_id)]


def run_submission_pass(
    config, meta, tracker, sample_dir: Path, manifests_dir: Path,
    array_concurrency: int, fov_mem: str, fov_time: str, mosaic_mem: str, mosaic_time: str,
    partition: str, conda_env: str, dry_run: bool,
) -> dict:
    """
    One pass of :func:`submit_pending_fov_analysis` +
    :func:`submit_pending_round_mosaics`, plus a fresh :meth:`tracker.summary`
    -- what ``07_cluster_submit_analysis.ipynb``'s manual and continuous-loop
    cells both call. Returns ``{"fov_submitted", "mosaic_submitted", "summary"}``.
    """
    fov_submitted = submit_pending_fov_analysis(
        config, meta, tracker, sample_dir, manifests_dir,
        array_concurrency, fov_mem, fov_time, partition, conda_env, dry_run=dry_run,
    )
    mosaic_submitted = submit_pending_round_mosaics(
        config, meta, tracker, sample_dir, manifests_dir,
        mosaic_mem, mosaic_time, partition, conda_env, dry_run=dry_run,
    )
    return {
        "fov_submitted": fov_submitted,
        "mosaic_submitted": mosaic_submitted,
        "summary": tracker.summary(meta),
    }
