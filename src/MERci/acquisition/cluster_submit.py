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
_CLI_ANALYZE_FOV          = _MERCI_SRC / "MERci" / "analysis" / "cli_analyze_fov.py"
_CLI_BUILD_ROUND_MOSAIC   = _MERCI_SRC / "MERci" / "analysis" / "cli_build_round_mosaic.py"
_CLI_COMPUTE_TEXTURE_STATS = _MERCI_SRC / "MERci" / "analysis" / "cli_compute_texture_stats.py"

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
