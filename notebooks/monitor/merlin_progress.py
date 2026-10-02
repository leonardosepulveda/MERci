#!/usr/bin/env python3
"""Live, top-like progress table for one or more running MERlin runs.

One row per MERlin task (union across all runs, in Snakefile order), one
column per run, each cell ``done/total (pct%)``. Finished tasks stay in the
table, unlike snakemake's own ``Job stats`` (which only lists what is left).

Usage::

    python3 merlin_progress.py                 # auto-discover running runs
    python3 merlin_progress.py MERLIN_DIR ...  # or name SAMPLE_DIR/merlin dirs
    python3 merlin_progress.py -n 60           # refresh every 60 s (default 30)
    python3 merlin_progress.py --once          # print once and exit

Standard library only, Python >= 3.6, so it runs with the login node's
system ``python3`` -- no conda env needed.

Where each number comes from (all read from disk/SLURM, nothing cached):

- **Runs**: auto-discovery lists the user's SLURM jobs named
  ``merlin_slurm_<label>.sh`` (the script MERci's
  ``merlin_config.create_merlin_slurm_script`` writes to
  ``SAMPLE_DIR/merlin/slurm/submit/``). squeue's command field is that
  script's path, so the merlin dir is three levels up.
- **Tasks and totals**: the newest ``<merlin>/<analysis>/snakemake/*.Snakefile``
  (MERlin writes one per launch). A rule whose output is ``<Task>_{i}.done``
  is per-fragment; its total is the ``range(N)`` in the matching
  ``<Task>Done`` rule. Any other rule (except ``*Done``/``all``) is a
  single-step task with total 1.
- **done / started / error**: MERlin's own status markers in
  ``<merlin>/<analysis>/<Task>/tasks/`` -- ``<Task>_<i>.start``, ``.done``,
  ``.error`` (``merlin.core.dataset.DataSet._record_analysis_event``).
  "running" = started but neither done nor errored, so a fragment killed
  mid-run (e.g. a preempted job) also counts as running until it is retried.
- **Footer**: snakemake's last ``N of M steps (P%) done`` line and the err
  log's age (the ``#SBATCH -e`` path in the submit script), plus the run's
  child jobs in squeue (snakemake names them ``<label>_<uuid>``) with the
  most common pending reason.
"""
import argparse
import getpass
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, OrderedDict

RULE_RE = re.compile(r"^rule (\w+):\s*\n\tinput: (.*)\n\toutput: '([^']*)'", re.M)
RANGE_RE = re.compile(r"range\((\d+)\)")
STEPS_RE = re.compile(rb"(\d+) of (\d+) steps \((\d+)%\) done")
MARKER_RE = re.compile(r"^(\w+?)(?:_(\d+))?\.(start|done|error)$")

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m")


def squeue_jobs():
    """[(jobid, name, state, command, reason)] for the current user, or None."""
    try:
        proc = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T|%o|%r"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode:
        return None
    return [tuple(line.split("|", 4)) for line in proc.stdout.splitlines()
            if line.count("|") >= 4]


class QueuePoller(threading.Thread):
    """Re-runs squeue in the background: on a busy cluster one call can take
    a minute, which must not hold up the (disk-only) table refresh."""

    def __init__(self, jobs, interval):
        super().__init__(daemon=True)
        self.jobs, self.stamp, self.interval = jobs, time.time(), interval

    def run(self):
        while True:
            if self.jobs is not None:
                time.sleep(self.interval)
            jobs = squeue_jobs()
            if jobs is not None:
                self.jobs, self.stamp = jobs, time.time()


def discover_merlin_dirs(jobs):
    dirs = []
    for _, name, _, command, _ in jobs:
        if name.startswith("merlin_slurm_") and name.endswith(".sh"):
            d = os.path.dirname(os.path.dirname(os.path.dirname(command)))
            if d not in dirs:
                dirs.append(d)
    return dirs


class Run:
    """One MERlin run, located by its SAMPLE_DIR/merlin folder."""

    def __init__(self, merlin_dir):
        self.merlin_dir = os.path.abspath(merlin_dir)
        submit = sorted(
            (os.path.join(self.merlin_dir, "slurm", "submit", f)
             for f in os.listdir(os.path.join(self.merlin_dir, "slurm", "submit"))
             if f.startswith("merlin_slurm_") and f.endswith(".sh")),
            key=os.path.getmtime) if os.path.isdir(
                os.path.join(self.merlin_dir, "slurm", "submit")) else []
        self.submit_script = submit[-1] if submit else None
        self.label = (os.path.basename(self.submit_script)[len("merlin_slurm_"):-3]
                      if self.submit_script else os.path.basename(
                          os.path.dirname(self.merlin_dir)))
        self.err_log = None
        if self.submit_script:
            with open(self.submit_script) as fh:
                m = re.search(r"^#SBATCH -e (\S+)", fh.read(), re.M)
            self.err_log = m.group(1) if m else None
        self.tasks = OrderedDict()   # task -> total fragments (None = single)
        self.analysis_dir = None
        self.snakefile = None

    def load_snakefile(self):
        """Re-read the newest Snakefile (a relaunch writes a new one)."""
        candidates = []
        for sub in os.listdir(self.merlin_dir):
            sm = os.path.join(self.merlin_dir, sub, "snakemake")
            if os.path.isdir(sm):
                candidates += [os.path.join(sm, f) for f in os.listdir(sm)
                               if f.endswith(".Snakefile")]
        if not candidates:
            return
        newest = max(candidates, key=os.path.getmtime)
        if newest == self.snakefile:
            return
        self.snakefile = newest
        self.analysis_dir = os.path.dirname(os.path.dirname(newest))
        with open(newest) as fh:
            text = fh.read()
        rules = OrderedDict((name, (inp, out)) for name, inp, out
                            in RULE_RE.findall(text))
        self.tasks = OrderedDict()
        for name, (_, out) in rules.items():
            if name.endswith("Done") and name[:-4] in rules:
                continue
            if out.endswith("_{i}.done"):
                done_rule = rules.get(name + "Done")
                m = RANGE_RE.search(done_rule[0]) if done_rule else None
                self.tasks[name] = int(m.group(1)) if m else 0
            else:
                self.tasks[name] = None

    def task_status(self, task):
        """(done, running, error, total) for one task."""
        total = self.tasks[task]
        tdir = os.path.join(self.analysis_dir, task, "tasks")
        marks = {"start": set(), "done": set(), "error": set()}
        try:
            with os.scandir(tdir) as it:
                for entry in it:
                    m = MARKER_RE.match(entry.name)
                    if m and m.group(1) == task:
                        frag = m.group(2)
                        if (total is None) == (frag is None):
                            marks[m.group(3)].add(frag)
        except FileNotFoundError:
            pass
        done, error = marks["done"], marks["error"] - marks["done"]
        running = marks["start"] - done - error
        return len(done), len(running), len(error), total or 1

    def snakemake_steps(self):
        """(last 'N of M steps (P%) done' bytes or None, err-log age in s)."""
        if not self.err_log or not os.path.exists(self.err_log):
            return None, None
        age = time.time() - os.path.getmtime(self.err_log)
        with open(self.err_log, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 200000))
            hits = STEPS_RE.findall(fh.read())
        return (hits[-1] if hits else None), age


def fmt_age(seconds):
    if seconds is None:
        return "?"
    if seconds < 120:
        return "{:.0f}s".format(seconds)
    if seconds < 7200:
        return "{:.0f}m".format(seconds / 60)
    return "{:.1f}h".format(seconds / 3600)


def cell(status, color):
    """Plain text and colored text for one table cell."""
    if status is None:
        return "-", DIM + "-" + RESET if color else "-"
    done, running, error, total = status
    text = "{}/{} ({:.0f}%)".format(done, total, 100.0 * done / total)
    if running:
        text += " r{}".format(running)
    if error:
        text += " e{}".format(error)
    if not color:
        return text, text
    tone = (RED if error else GREEN if done == total
            else YELLOW if done or running else DIM)
    return text, tone + text + RESET


def render(runs, jobs, jobs_age, color):
    task_order = []
    for run in runs:
        for task in run.tasks:
            if task not in task_order:
                task_order.append(task)
    statuses = {(run.label, t): run.task_status(t)
                for run in runs for t in run.tasks}

    header = ["task"] + [run.label for run in runs]
    rows = []
    for task in task_order:
        cells = [cell(statuses.get((run.label, task)), color) for run in runs]
        rows.append([(task, task)] + cells)
    widths = [max([len(header[i])] + [len(r[i][0]) for r in rows])
              for i in range(len(header))]

    def line(pairs):
        out = []
        for i, (plain, shown) in enumerate(pairs):
            pad = " " * (widths[i] - len(plain))
            out.append(shown + pad if i == 0 else pad + shown)
        return "  ".join(out)

    lines = [time.strftime("%Y-%m-%d %H:%M:%S") + "  MERlin progress"
             "  (cell = done/total (%)  rN = running  eN = errored)", ""]
    lines.append(line([(h, BOLD + h + RESET if color else h) for h in header]))
    lines.append("  ".join("-" * w for w in widths))
    lines += [line(r) for r in rows]
    lines.append("")
    if jobs is None:
        lines.append("squeue: no answer yet")
    else:
        lines.append("squeue as of {} ago".format(fmt_age(jobs_age)))

    for run in runs:
        steps, age = run.snakemake_steps()
        children = [j for j in jobs or [] if j[1].startswith(run.label + "_")]
        states = Counter(j[2] for j in children)
        reasons = Counter(j[4] for j in children if j[2] == "PENDING")
        parent = [j for j in jobs or [] if j[1] == "merlin_slurm_{}.sh".format(run.label)]
        lines.append("{}: {}".format(BOLD + run.label + RESET if color else run.label,
                                     run.merlin_dir))
        lines.append("    snakemake: {}   err log updated {} ago   driver job: {}".format(
            "{} of {} steps ({}%)".format(*(s.decode() for s in steps)) if steps else "?",
            fmt_age(age),
            "?" if jobs is None else parent[0][2] if parent else "not in queue"))
        if jobs is None:
            continue
        lines.append("    child jobs: {} running, {} pending{}".format(
            states.get("RUNNING", 0), states.get("PENDING", 0),
            "  (top reason: {} x{})".format(*reasons.most_common(1)[0])
            if reasons else ""))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("merlin_dirs", nargs="*",
                    help="SAMPLE_DIR/merlin folders (default: discover from squeue)")
    ap.add_argument("-n", "--interval", type=float, default=30,
                    help="refresh interval in seconds (default 30)")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    color = sys.stdout.isatty() and not args.no_color

    jobs = None  # with explicit dirs (live mode), the poller fetches it instead
    if not args.merlin_dirs or args.once:
        print("Asking squeue (can take a minute on a busy cluster) ...",
              file=sys.stderr, flush=True)
        jobs = squeue_jobs()
    dirs = args.merlin_dirs or discover_merlin_dirs(jobs or [])
    if not dirs:
        sys.exit("No running merlin_slurm_*.sh jobs found; pass MERLIN_DIR(s).")
    runs = [Run(d) for d in dirs]
    poller = QueuePoller(jobs, args.interval)
    if not args.once:
        poller.start()

    try:
        while True:
            for run in runs:
                run.load_snakefile()
            text = render(runs, poller.jobs, time.time() - poller.stamp, color)
            if args.once:
                print(text)
                return
            sys.stdout.write("\033[H\033[2J" + text + "\n\n(refresh every {:g}s, "
                             "Ctrl-C to quit)\n".format(args.interval))
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
