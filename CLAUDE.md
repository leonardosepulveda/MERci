# CLAUDE.md

Guidance for Claude Code working in this repository.

**This file is a map, not a manual.** It stays short on purpose — it's read in
full every session. For any function/notebook: the *why* (subtle invariants,
past bugs, design rationale) lives in that file's own docstring/markdown
cells, and the *history* of how a decision was reached lives in
`prompt_history/`. Read those on demand when you touch the relevant code;
don't expect this file to carry that detail. (This file itself is
git-tracked — `git log -p -- CLAUDE.md` recovers any older, more verbose
version if something you need turns out to be missing here.)

## Overview

Python tooling for MERFISH spatial transcriptomics experiments — acquisition
planning and online QC during imaging. Primary package: `MERci/`.

## Environment setup

```bash
mamba env create -f environment.yml   # one-time per computer
mamba activate merci_env
jupyter lab
```

Open notebooks from their folders under `MERci/notebooks/` in the JupyterLab
file browser so `SAMPLE_DIR` auto-detects correctly.

## Deployment model

This repo is cloned into each experiment folder as `SAMPLE_DIR/MERci/`. No
`pip install` needed — deps come from `merci_env`. Notebooks resolve
`MERCI_DIR`/`SAMPLE_DIR` by counting parent dirs from their own location:

- `after_imaging/`, `during_imaging/`, `misc/`: 2 levels
  (`MERCI_DIR = Path(os.getcwd()).parent.parent`)
- `before_imaging/{regular,multi_z}/` (3 levels): `.parent.parent.parent`
- repo-root `tests/` (1 level, `.parent`) and `tests/<subfolder>/` (2 levels)

`SAMPLE_DIR = MERCI_DIR.parent`. Never hardcode absolute paths in notebooks.

**Exported notebooks (optional)**: `notebooks/before_imaging/00_select_pipeline.ipynb`
copies one pipeline's notebooks, flattened, plus the shared
`after_imaging/`/`during_imaging/` notebooks, into a standalone
`SAMPLE_DIR/notebooks/` folder that sits *alongside* `SAMPLE_DIR/MERci/`
instead of inside it (`MERci/acquisition/pipeline_export.py`). There, `MERci`
is a sibling rather than an ancestor, so every exported notebook resolves
`MERCI_DIR = Path(os.getcwd()).parent.parent / "MERci"` — one fixed formula
regardless of the original notebook's nesting depth. The export also copies
the chosen pipeline's `pipeline.yaml`+`round_bit_color.csv` (if it has one —
every pipeline except `multi_z`) to `SAMPLE_DIR/notebooks/`, and rewrites
every notebook that loads it to read *that* copy instead of the one under
`MERci/data/pipelines/` — so it can be hand-edited per experiment without
touching `MERci/`. The shared (not per-experiment) per-microscope power
table it also needs still comes from the live `MERci/` clone. The `MERci/`
clone itself is only ever read from by the export, never modified.

**`before_imaging/regular/`** — one shared notebook set for every pipeline
except `multi_z`: `tumor_epi`, `tumor_disk`, `lineage_tracing_merfish`,
`lineage_tracing_lineage`. What used to differ between per-pipeline notebook
copies (microscope, imaging recipe, fluidics, codebook/task menu or fishtank
targets) now lives entirely in that pipeline's own
`data/pipelines/<id>_pipeline.yaml` (`acquisition/pipeline_config.py`); every
notebook's second cell sets `PIPELINE_ID` and loads it into
`PIPELINE_CONFIG`. Steps 05/07 have two files each (`analysis_backend:
merlin` vs `fishtank`) living side by side — `pipeline_export.py` copies only
the matching pair. See `regular/README.md`.

**`before_imaging/multi_z/`** — a separate pipeline (no `pipeline.yaml` yet)
for a variable-z-per-FOV acquisition: images a full-depth DAPI (cells) round
first, measures each FOV's real tissue thickness
(`after_imaging/08_measure_tissue_thickness.ipynb`, since that step runs
mid-acquisition once the cells round exists), then generates one bits HAL
config per z-depth tier. Own 9-notebook sequence — see `multi_z/README.md`.

## Experiment folder layout

```
SAMPLE_DIR/
  MERci/             clone of this repo
  positions/         boundaries/{manual,from_mosaic}/ + positions_{SAMPLE_NAME}.txt
  metadata/          frame_table_*.csv, round_info.csv, round_bit_color_map.csv,
                     data_organization_*.csv, experiment_info.yaml
  settings/          hal-config-*.xml, shutter-*.xml, dave-*.xml
  data/              raw image files (subfolder structure from round_info.csv's `dir` column)
  analysis/          thumbnails/, stats/, histograms/, mosaics/, done/
  merlin/            per-experiment MERlin config/run files (or fishtank/ for lineage_tracing_lineage)
  figures/           MERlin's per-task verification figures (merlin.<taskName>.<figureName>.png),
                     written via the generated sbatch script's `-f "$SAMPLE_DIR/figures"` flag
```

## Package layout

One-line index only — see each file's own docstring for what it actually
does, its parameters, and any gotcha a caller must respect.

```
src/MERci/
  common/
    config.py            ExperimentConfig — all paths/tunables
    metadata.py           ExperimentMetadata — parses round_info.csv + positions
    io.py                 read/write dax/zarr/tiff, frame-selective reads
    experiment_info.py    ExperimentInfo, resolve_sample_identity, collect_experiment_info
  acquisition/
    configs.py            frame tables, HAL/shutter config generation, color/channel mapping,
                          camera geometry/orientation
    positions.py           FOV grid generation, scanning paths, multi-tissue boundaries
    mosaic.py              derive tissue boundaries from a Steve low-mag mosaic
    alignment.py           cross-microscope FOV transfer, bead-drift registration
    camera_rotation.py     camera-vs-stage rotation correction, microscope orientation flags
    dave.py                Dave experiment-recipe XML generation
    kilroy.py              Kilroy fluidics-protocol resolution/consistency checks
    data_organization.py   MERlin data-organization CSV
    merlin_config.py       MERlin input/config-file generation (SAMPLE_DIR/merlin/)
    fishtank_config.py     fishtank input/config-file generation (lineage_tracing_lineage only)
    display.py             print_frame_table, display_xml, display_file
    cluster_submit.py      sbatch script generation for cluster-side QC analysis
    pipeline_config.py     PipelineConfig/MerlinConfig/FishtankConfig -- loads data/pipelines/<id>_pipeline.yaml
    pipeline_export.py     export one pipeline's notebooks to SAMPLE_DIR/notebooks/ (sibling of MERci/)
  analysis/
    fov.py                 per-FOV thumbnails/stats/histograms/intensity-percentiles
    round.py               round-level mosaics (plain + flat-field-corrected)
    ffc.py                 flat-field correction for round mosaics
    stage_z.py             stage-z drift QC from HAL's .off focus-lock sidecars
    spot_localization.py   bead detection / 3D Gaussian fitting / PSF simulation
    completeness.py         per-FOV raw-file existence + zarr chunk-integrity checks (no decompression)
    elevation.py           per-pixel tissue elevation/thickness, FFC from interior FOVs, GIF/movie/z-mosaic
    cell_mapping.py        per-cell identity matching between two segmentations of one tissue
    fast_spot_quantification.py  per-bit hyb-reagent spot QC (during_imaging)
    imaged_fovs.py         which round a live acquisition-progress view watches
    view_intensity_stats.py  load annotated per-FOV stats (after_imaging/04)
    batch_sample_review.py backfill + combined stats across a batch of samples (after_imaging/05)
    cli_*.py               standalone SLURM-array-task scripts (self-locating, no pip install
                           needed), one per cluster_submit builder; shared args in _cli_common.py
  live_round_mosaic.py     LiveRoundMosaicBuilder -- live quick-look mosaic (during_imaging/round_mosaics)
  plots/                   all plotting: per-notebook plot modules, experiment_plots (shutter sequence,
                           FOV layout, stats-over-rounds, figures dir), mosaic_plots, spot_localization_plots
  state.py                 ExperimentStateMonitor — imaging vs. fluidics phase detection
  progress.py              ProgressTracker — sentinel-file completion tracking
  progress_display.py      ProgressReporter — live console/notebook progress+ETA
  scheduler.py             FOVScheduler, RoundScheduler
  transfer.py              transfer_round, mirror_tree
  visualization.py         old import path, re-exports plots/experiment_plots
  disk_audit.py            scan shared-drive sample folders for cleanup candidates
```

## Notebooks

```
notebooks/
  before_imaging/    Pre-experiment, run in order. Two pipelines:
                     regular/ (tumor_epi, tumor_disk, lineage_tracing_merfish,
                     lineage_tracing_lineage -- one shared notebook set, see its own
                     README.md), multi_z/ (own 9-notebook sequence, see its own README.md)
    00  select_pipeline (opt.)               pick a pipeline, export it + after/during_imaging
                                              to SAMPLE_DIR/notebooks/ (sibling of MERci/)
    01  create_hal_config_and_shutters       imaging sequence, HAL/shutter XML, transit config
    02a create_boundary_from_mosaic (opt.)   derive tissue boundary from a Steve mosaic
    02b create_positions_from_boundaries     FOV scanning positions
    03  create_round_info                    round-bit-color map, round_info.csv
    04  create_dave_config                   Dave experiment-recipe XML, incl. per-round bit/color XML comments
    05  create_data_organization             MERlin data-org CSV (analysis_backend: merlin)
        create_color_usage                   fishtank color_usage/decoding_strategy (analysis_backend: fishtank)
    06  create_experiment_info               metadata/experiment_info.yaml
    07  create_merlin_scripts                SAMPLE_DIR/merlin/ (analysis_backend: merlin)
        create_fishtank_scripts              SAMPLE_DIR/fishtank/ (analysis_backend: fishtank)
  after_imaging/     Online analysis, run during the experiment
    01  fov_scheduler              FOV-level scheduler (thumbnails, stats, histograms)
    02  round_scheduler            round-level scheduler (mosaics, optional transfer)
    03  view_mosaics               display per-color mosaics
    04  view_intensity_stats       per-frame intensity stats over rounds
    05  batch_sample_review        post-acquisition: verify/backfill a batch, compare across it
    06  map_cells_across_microscopes  cross-microscope cell-identity mapping between two experiments of the same sample (see its own intro cell for the staged plan)
    07  cluster_submit_analysis    submit SLURM array jobs for QC (alternative to local 01/02)
    08  measure_tissue_thickness   per-FOV tissue z-extent + thickness heatmap/mosaic/GIF (any pipeline)
    09  multi_z_margin_export      multi_z only: margin/savings + trimmed-depth verify + z-table export, feeds multi_z's own notebook 04 -- continues 08's own state (see its own intro cell)
    10  check_fov_completeness     per-FOV raw-file existence + zarr chunk-integrity check (catches truncated writes)
    11  compare_tissue_thickness_merfish_lineage  lineage_tracing only: 08's heatmap/mosaic/movie for a
                                              sample's merfish + lineage sibling acquisitions, side by
                                              side (single colorbar/scale bar/z, depth-matched movie)
    12  measure_intensity_percentiles  per-frame (frame, z, color, min, p25/p50/p75/p95, max)
                                              intensity table, one SLURM array task per FOV movie file,
                                              across every declared round plus any undeclared
                                              "<round>_old[_N]" test-reimage folder (see
                                              MERci.common.metadata.discover_ad_hoc_round_dirs);
                                              cached as parquet
  during_imaging/    Live QC meant to be watched in real time
    stage_z_drift          stage-z drift from .off sidecars, one line per round
    imaged_fovs             live acquisition-progress map
    round_mosaics            live quick-look mosaic (on-demand/catch-up/live modes)
    fast_spot_quantification per-bit hybridization-reagent QC
    hyb_spot_intensity_qc    experiment-wide spot-intensity overview, N FOVs evenly spaced across the tissue
    z_profile_spot_intensity per-z (no projection) foci detection for one hyb round + FOV, incl. custom image paths
    dave_timing_accuracy     actual vs. Dave-estimated block timing, real-data ETA for remaining blocks
  misc/              Ad-hoc utilities — see each notebook's own markdown cells for what it does
```

**Test notebooks** live in a repo-root `tests/` folder (not under `notebooks/`),
which is gitignored: local only, not shipped with the clone deployed into each
experiment. It holds diagnostic/recovery notebooks for one specific real
incident, kept as templates, and validation notebooks for a new feature
(synthetic and/or real-data checks) written before it's wired into a production
notebook. Tracked files must not point into it by path (describe the notebook
instead). Investigations that outgrow a test notebook move to a standalone
analysis folder outside this repo and come back as `prompt_history/` handoffs.

## Architecture

**Pre-experiment workflow**: run the 8 `regular/` notebooks (or the 9
`multi_z/` ones, plus `after_imaging/08_measure_tissue_thickness.ipynb`
mid-sequence) above in order for the acquisition being prepared. Each writes
inputs the next one reads (HAL/shutter → positions → round_info → Dave
config → data-organization/color-usage → experiment_info → merlin/fishtank
scripts). Naming convention: `{kind}-{name}` stems (`bits`/`cells`/`transit`)
shared across HAL config, shutter file, and frame table for one round.

**Online-analysis**: `ExperimentConfig` holds paths/tunables.
`ExperimentMetadata` cross-references round/FOV/series/paths.
`ExperimentStateMonitor` detects imaging vs. fluidics phase from file mtimes.
`ProgressTracker` tracks completion via sentinel files under
`analysis/done/`. `FOVScheduler`/`RoundScheduler` run
the continuous analysis loops (see `scheduler.py`'s own docstring for the
full contract). QC analysis can instead run on a SLURM cluster via
`07_cluster_submit_analysis.ipynb` + `cli_analyze_fov.py`/
`cli_build_round_mosaic.py` + `cluster_submit.py`.

**Key data files**: `round_info.csv` (`imaging_round`, `series` format
string; optional `imaging_type`/`hal_config`/`shutter_file`/`dir`), loaded
via `common.io.load_round_info`. `positions_{SAMPLE_NAME}.txt` (comma-sep
x,y, `#`-comments). Images: `.zarr`/`.dax`/`.tiff`, read via `read_image`.

**Microscope channel mapping**: `MF2`/`MF3`/`MF4`/`MF5` share
`{405→4, 488→3, 560→2, 650→1, 750→0}` (5 channels). `MFX`/`ST2`:
`{650→0, 560→1, 488→2, 405→3}` (4 channels, no 750). Extend
`_COLOUR_TO_CHANNEL` in `acquisition/configs.py` for other scopes. Camera
properties (frame size, pixel size, orientation) come only from each scope's
MERlin microscope JSON (`data/configs/merlin/microscope/`, via
`merlin_config.load_microscope_parameters`). There are no in-code defaults,
and an unknown scope raises. Never hard-code them in a notebook. Acquisition
type (orthogonal to the above):
`MF2`/`MFX`/`ST2` = spinning-disk (`"disk"`); `MF3`/`MF4`/`MF5` =
epifluorescence (`"epi"`) — `get_acquisition_type`.

## Notebook coding guidelines

Every notebook follows [`NOTEBOOK_GUIDELINES.md`](NOTEBOOK_GUIDELINES.md):
separate calculation cells from display/plot cells, cache under
`analysis/cache/<notebook_name>/`, skip recomputation when cache is valid,
report progress (n/total, elapsed, ETA) in nontrivial loops, explicit plot
font sizes.

**Diagnostic images**: save every diagnostic image meant for the user's own
eyes to a real path under the experiment tree or repo (never only the
scratchpad), and state the literal path. When a section is redesigned,
delete/rename its old diagnostic PNGs rather than leaving a stale
same-named file in place. Before calling a diagnostic output "correct" from
a rendered image, confirm the code path actually applies the transform
being claimed (e.g. grep for it) — a plausible picture isn't proof. (Learned
from a real multi-hour false alarm — see
`prompt_history/2026_07_31_1932_confirm_camera_rotation_orientation.md`.)

## Running notebooks

Notebooks auto-detect `SAMPLE_DIR` from their own location (see "Deployment
model" above for the exact parent-dir counts per variant). Do not hardcode
absolute paths.

## Test notebooks: stay portable

`tests/<subfolder>/` notebooks should read real data through a
local copy in a `data/` subfolder of that notebook's own cache folder (see
"Working / cache files" below), not directly from the live
experiment tree in every cell -- see
[`NOTEBOOK_GUIDELINES.md`](NOTEBOOK_GUIDELINES.md)'s "Test notebooks: stay
portable" section for the full rule (folder layout, provenance comments,
calculation/plotting split for data too big to copy, preferring `MERci`
library functions over inline logic). This makes a later export to a
standalone folder (`/save_test <subfolder> <destination>`) a copy, not a
rewrite.

## Scope constraint

All edits and analysis must stay within this repo. Do not modify sibling
folders (`image_acquisition/`, `imaging_with_storm_control/`, etc.) unless
explicitly requested.

## Version control

Commit and push as you go — do not leave finished work uncommitted.

- One focused commit per logical change, then `git push` to `origin`.
- Don't batch unrelated changes into one commit; don't let edits pile up locally.
- Standing authorization to commit and push without asking each time.
- Never commit transient files (`*.tmp.*`, `__pycache__/`, `*.egg-info/` — gitignored).
- `tests/` is gitignored, so test-notebook work is never committed. A
  change a test investigation makes outside `tests/` (e.g. a function
  pulled out into `src/MERci/`) follows the normal push/merge rules above,
  on its own branch.

## Working / cache files

Any working/intermediate file Claude generates (notebook-generator scripts,
diagnostic images, migration backups) goes under `cache/{prompt_date}/`
(repo root, gitignored) — never the session scratchpad or anywhere outside
this repo. `{prompt_date}` is the `YYYY_MM_DD_HHMM` timestamp of the
`prompt_history/` entry for the request that created the file (e.g. work
for an entry named `2026_09_21_1816_<description>.md` goes in a
`2026_09_21_1816` folder under `cache/`). Subfolders inside it are
free-form -- name them after the notebook or task. Distinct from
`analysis/cache/<notebook_name>/` (a per-*experiment* cache under
`SAMPLE_DIR/`, not this repo).

A `tests/` notebook's local data copy lives in the folder of the
request that created that notebook, at
`cache/{prompt_date}/<...>/<notebook_name>/data/`, and the notebook
hardcodes that path -- later requests reuse it instead of copying the data
again. See [`NOTEBOOK_GUIDELINES.md`](NOTEBOOK_GUIDELINES.md)'s "Test
notebooks: stay portable" section.

Two fixed exceptions directly under `cache/`: the verbatim-capture buffer
folder (written by the Stop hook, see below) and an `unsorted` folder
(files from before this convention whose creating request could not be
identified).

## Remembering task history

Two local-only, gitignored records:

1. **`prompt_history/`** — the log. One file per request:
   `{YYYY_MM_DD_HH_MM}_{short_description}.md`, frontmatter + `## Prompt`
   (verbatim, never paraphrased) / `## Plan` / `## Summary`, plus optional
   `## Learning` (a genuinely generalizable lesson, not invented to fill the
   slot) and `## Verbatim History` (folded in from
   `cache/verbatim_buffer/{date}_verbatim.md` at finalization).
2. **`FINDINGS.md`** — curated, deduplicated *current state*: what's true
   now, what was wrong and got fixed, the open next step. Read this first
   when resuming.

**For every request**: log it to `prompt_history/`. If it changes a
conclusion or project state, update `FINDINGS.md` too. Two ways a prompt
arrives:
- **Pre-written file** (`prompt_history/{YYYY_MM_DD_HH_MM}.txt`, user-authored):
  read it, act on it, then rewrite it into the standard format above and
  rename to add a short description.
- **Typed directly**: create a new file in the standard format immediately.

**Never fabricate the timestamp.** Use the `UserPromptSubmit` hook's injected
`Current local date/time: … (epoch N)` when present, or a direct `date`/
`Get-Date` call — never estimate or space entries at a suspiciously regular
interval. Compute `elapsed` as finish time minus that submit epoch; omit it
if no submit epoch is available.

**`SOMEDAY.md`** (repo root, gitignored): real-but-deferred work, one dated
entry per item, newest first. When picked up, do it as a normal logged
request and delete the entry.

**Optional rationale docs**: for investigation-heavy tasks (reverse-
engineering a format, debugging unfamiliar source, iterating through failed
approaches), also write `prompt_rationales/{same-basename}.html` (gitignored,
personal) via the `/rationale` command — a narrative walkthrough with real
code/dead-ends. Not needed for mechanical tasks; offer one proactively at
the end of an investigation-heavy task.
