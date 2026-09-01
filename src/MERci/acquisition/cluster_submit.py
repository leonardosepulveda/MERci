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
from typing import Optional

log = logging.getLogger(__name__)

# .../MERci/src/MERci/acquisition/cluster_submit.py -> .../MERci/src
_MERCI_SRC = Path(__file__).resolve().parents[2]
_CLI_ANALYZE_FOV            = _MERCI_SRC / "MERci" / "analysis" / "cli_analyze_fov.py"
_CLI_BUILD_ROUND_MOSAIC     = _MERCI_SRC / "MERci" / "analysis" / "cli_build_round_mosaic.py"
_CLI_COMPUTE_TEXTURE_STATS  = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_texture_stats.py"
_CLI_TPC_MARGIN_THUMBNAILS  = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_tpc_margin_thumbnails.py"
_CLI_GIF_FRAME_THUMBNAILS   = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_gif_frame_thumbnails.py"
_CLI_CHANNEL_COUNTERS       = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_channel_counters.py"

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
) -> str:
    """Same shape as ``fishtank_config.py``'s own ``_sbatch_header``,
    generalised with a configurable conda env (MERci's own analysis code
    needs only the scientific-stack env mirroring ``merci_env``, not
    ``fishtank_env``)."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
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
    """
    Write an sbatch array-job script that runs ``cli_analyze_fov.py`` once
    per pending FOV file listed in *manifest_path* (one path per line,
    ``$SLURM_ARRAY_TASK_ID`` selects the line).
    """
    log_dir = Path(sample_dir) / "analysis" / "logs"
    header = _sbatch_header(
        job_name=job_name, mem=mem, time=time,
        output_log=str(log_dir / "%x_%A_%a.out"),
        partition=partition, array=f"0-{n_pending - 1}%{array_concurrency}",
        conda_env=conda_env,
    )
    body = (
        f"python {_CLI_ANALYZE_FOV} \\\n"
        f"    --sample-dir {sample_dir} \\\n"
        f"    --manifest {manifest_path}\n"
    )
    return _write_script(output_path, header + "\n" + body)


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
    Write an sbatch job script that runs ``cli_build_round_mosaic.py`` for
    the round(s) listed in *manifest_path* (one round id per line) --
    submitted as an array job when there is more than one pending round,
    or a single plain job (``--round-id``) when there's exactly one.
    """
    log_dir = Path(sample_dir) / "analysis" / "logs"
    array = f"0-{n_pending - 1}" if n_pending > 1 else None
    header = _sbatch_header(
        job_name=job_name, mem=mem, time=time,
        output_log=str(log_dir / "%x_%A_%a.out"),
        partition=partition, array=array,
        conda_env=conda_env,
    )
    if n_pending > 1:
        body = (
            f"python {_CLI_BUILD_ROUND_MOSAIC} \\\n"
            f"    --sample-dir {sample_dir} \\\n"
            f"    --manifest {manifest_path}\n"
        )
    else:
        round_id = int(Path(manifest_path).read_text().strip().splitlines()[0])
        body = (
            f"python {_CLI_BUILD_ROUND_MOSAIC} \\\n"
            f"    --sample-dir {sample_dir} \\\n"
            f"    --round-id {round_id}\n"
        )
    return _write_script(output_path, header + "\n" + body)


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
    """
    Write an sbatch array-job script that runs ``cli_compute_texture_stats.py``
    once per pending FOV file listed in *manifest_path* -- the SLURM-array
    counterpart to ``measure_tissue_thickness.ipynb`` section 14's own
    sequential loop, for experiments where computing every FOV's texture
    profile locally would take too long (each task re-reads one FOV's own
    z-stack of full-resolution frames, unlike sections 4/11-13 which reuse
    already-cached intensity Counters).
    """
    log_dir = Path(sample_dir) / "analysis" / "logs"
    header = _sbatch_header(
        job_name=job_name, mem=mem, time=time,
        output_log=str(log_dir / "%x_%A_%a.out"),
        partition=partition, array=f"0-{n_pending - 1}%{array_concurrency}",
        conda_env=conda_env,
    )
    frame_indices_str = ",".join(str(i) for i in frame_indices)
    body = (
        f"python {_CLI_COMPUTE_TEXTURE_STATS} \\\n"
        f"    --manifest {manifest_path} \\\n"
        f"    --output-dir {output_dir} \\\n"
        f"    --frame-indices {frame_indices_str} \\\n"
        f"    --sigma {sigma}\n"
    )
    return _write_script(output_path, header + "\n" + body)


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
    """
    Write an sbatch array-job script that runs
    ``cli_compute_tpc_margin_thumbnails.py`` once per pending FOV listed in
    *manifest_path* -- the SLURM-array counterpart to
    ``measure_tissue_thickness.ipynb`` section 23's own sequential loop,
    for experiments where reading every FOV's bounded margin-sweep window
    locally would take too long (each task still only reads its own FOV's
    window once, covering every margin candidate).
    """
    log_dir = Path(sample_dir) / "analysis" / "logs"
    header = _sbatch_header(
        job_name=job_name, mem=mem, time=time,
        output_log=str(log_dir / "%x_%A_%a.out"),
        partition=partition, array=f"0-{n_pending - 1}%{array_concurrency}",
        conda_env=conda_env,
    )
    frame_idx_str = ",".join(str(i) for i in frame_indices)
    z_um_str      = ",".join(str(z) for z in z_um_values)
    margins_str   = ",".join(str(m) for m in margins)
    tw, th        = thumbnail_size
    orientation_flags = " ".join(
        f"--{flag.replace('_', '-')}" for flag, on in orientation.items() if on
    )
    body = (
        f"python {_CLI_TPC_MARGIN_THUMBNAILS} \\\n"
        f"    --manifest {manifest_path} \\\n"
        f"    --output-dir {output_dir} \\\n"
        f"    --frame-indices {frame_idx_str} \\\n"
        f"    --z-um-values {z_um_str} \\\n"
        f"    --margins {margins_str} \\\n"
        f"    --thumbnail-width {tw} --thumbnail-height {th} \\\n"
        f"    {orientation_flags}\n"
    )
    return _write_script(output_path, header + "\n" + body)


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
    """
    Write an sbatch array-job script that runs
    ``cli_compute_gif_frame_thumbnails.py`` once per pending FOV listed in
    *manifest_path* -- the SLURM-array counterpart to
    ``measure_tissue_thickness.ipynb`` section 24's own sequential loop,
    for experiments where reading every FOV at every GIF_Z_STRIDE-selected
    z-step locally would take too long (each task still reads its own
    FOV's selected z-steps in one batched call, not one read per step).
    """
    log_dir = Path(sample_dir) / "analysis" / "logs"
    header = _sbatch_header(
        job_name=job_name, mem=mem, time=time,
        output_log=str(log_dir / "%x_%A_%a.out"),
        partition=partition, array=f"0-{n_pending - 1}%{array_concurrency}",
        conda_env=conda_env,
    )
    z_pos_str     = ",".join(str(z) for z in z_positions)
    frame_idx_str = ",".join(str(i) for i in frame_indices)
    tw, th        = thumbnail_size
    orientation_flags = " ".join(
        f"--{flag.replace('_', '-')}" for flag, on in orientation.items() if on
    )
    body = (
        f"python {_CLI_GIF_FRAME_THUMBNAILS} \\\n"
        f"    --manifest {manifest_path} \\\n"
        f"    --output-dir {output_dir} \\\n"
        f"    --z-positions {z_pos_str} \\\n"
        f"    --frame-indices {frame_idx_str} \\\n"
        f"    --thumbnail-width {tw} --thumbnail-height {th} \\\n"
        f"    {orientation_flags}\n"
    )
    return _write_script(output_path, header + "\n" + body)


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
    """
    Write an sbatch array-job script that runs
    ``cli_compute_channel_counters.py`` once per pending FOV listed in
    *manifest_path* -- the SLURM-array counterpart to
    ``measure_tissue_thickness``-style notebooks' own section-4 sequential
    loop (the heaviest read step: one full channel z-sweep per FOV).
    """
    log_dir = Path(sample_dir) / "analysis" / "logs"
    header = _sbatch_header(
        job_name=job_name, mem=mem, time=time,
        output_log=str(log_dir / "%x_%A_%a.out"),
        partition=partition, array=f"0-{n_pending - 1}%{array_concurrency}",
        conda_env=conda_env,
    )
    frame_idx_str = ",".join(str(i) for i in frame_indices)
    z_um_str      = ",".join(str(z) for z in z_um_values)
    body = (
        f"python {_CLI_CHANNEL_COUNTERS} \\\n"
        f"    --manifest {manifest_path} \\\n"
        f"    --output-dir {output_dir} \\\n"
        f"    --frame-indices {frame_idx_str} \\\n"
        f"    --z-um-values {z_um_str}\n"
    )
    return _write_script(output_path, header + "\n" + body)


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
        manifest_path.write_text("\n".join(str(f) for f in round_files) + "\n")
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
    manifest_path.write_text("\n".join(str(r) for r in pending_rounds) + "\n")
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
