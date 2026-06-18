# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains Python tooling for MERFISH (Multiplexed Error-Robust Fluorescence In Situ Hybridization) spatial transcriptomics experiments — specifically acquisition planning and online quality-control analysis during imaging. The primary package is `MERci/`.

## Environment setup

```bash
# One-time, per computer (uses mamba from Miniforge)
mamba env create -f environment.yml

# Activate and launch notebooks
mamba activate merci_env
jupyter lab
```

Open notebooks from their folders under `MERci/notebooks/` (`prepare_imaging/`, `analysis/`, `misc/`) in the JupyterLab file browser so that `SAMPLE_DIR` is auto-detected correctly.

## Deployment model

This repo is cloned directly into each new experiment folder (`SAMPLE_DIR`) as `SAMPLE_DIR/MERci/`. No `pip install` is needed — dependencies come from the `merci_env` conda environment. Notebooks live two levels under the repo root (e.g. `MERci/notebooks/prepare_imaging/`), so each notebook resolves paths as:

```python
MERCI_DIR  = Path(os.getcwd()).parent.parent   # MERci/
SAMPLE_DIR = MERCI_DIR.parent                   # experiment root
sys.path.insert(0, str(MERCI_DIR / "src"))      # MERci/src
```

## Experiment folder layout

```
SAMPLE_DIR/          (the experiment root, e.g. D:\experiments\my_sample\)
  MERci/             ← clone of this repo
  positions/         ← boundary_positions.txt, hole*.txt (from operator),
                        positions_{SAMPLE_NAME}.txt (from prepare_imaging/02)
  metadata/          ← frame_table_*.csv, shutter_sequence_*.png (prepare_imaging/01),
                        round_info.csv (prepare_imaging/03),
                        round_bit_color_map.csv, data_organization_*.csv (prepare_imaging/04)
  settings/          ← hal-config-*.xml, shutter-*.xml (prepare_imaging/01), dave-*.xml (prepare_imaging/03)
  data/              ← raw image files; exact subfolder structure defined by the `dir`
                        column in round_info.csv (written by HAL during acquisition)
  analysis/          ← thumbnails/, stats/, histograms/, mosaics/, done/
                        (produced by the analysis/01 and analysis/02 schedulers)
```

## Package layout

```
src/MERci/
  common/
    config.py       # ExperimentConfig dataclass — all paths and tunable parameters
    metadata.py     # ExperimentMetadata — parses round_info.csv + positions.txt
    io.py           # read_dax/zarr/tiff/image, parse_inf, get_dax_shape, load_round_info, load_positions,
                    # save_positions_array, discover_image_files
  acquisition/
    configs.py      # get_frame_table, get_color_sequence_name, get_color_to_channel_dict, create_shutter_file,
                    # create_hal_config, format_z_offsets_from_frame_table
                    # + read_hal_flip_vertical, find_frame_table_for_hal_config, get_color_frame_indices
                    # + reconstruct_frame_table (inverse: hal+shutter XML -> frame table) and its parsers
                    #   read_shutter_reference, parse_z_offsets, parse_shutter_events
    positions.py    # create_grid_positions, generate_scanning_path, filter_scanning_path, close_scanning_path,
                    # load_hole_polygons, get_path_stats
    dave.py         # create_round_info, create_dave_config, annotate_dave_with_round_info,
                    # series_to_movie_name, get_hal_frame_count
    kilroy.py       # load_kilroy_protocols, find_kilroy_config (MF2 fallback),
                    # KilroyProtocolResolver — resolve dave fluidic steps to real Kilroy protocol names
    data_organization.py  # create_data_organization
    display.py      # print_frame_table, display_xml (Jupyter helpers)
  analysis/
    fov.py          # create_thumbnail(s), measure_stats, get_histogram, load_stats, load_histogram — FOV-level analysis
    round.py        # create_mosaic, load_thumbnails_for_round — round-level mosaic
    spot_localization.py  # bead detection / 3D Gaussian fitting + PSF simulation (detect_beads_2d,
                          # localize_beads_in_file, match_beads_across_colors, simulate_multicolor_stack, …)
  state.py          # ExperimentStateMonitor — detects imaging vs. fluidics phases by watching file mtimes
  progress.py       # ProgressTracker — sentinel files for fov_done, round_done, round_transferred
  scheduler.py      # FOVScheduler, RoundScheduler, ExperimentScheduler — main analysis loops
  transfer.py       # transfer_round — background robocopy/shutil to a network destination
  visualization.py  # visualize_shutter_sequence, plot_fov_layout, plot_stats_over_rounds, display_mosaic
notebooks/
  prepare_imaging/  # Pre-experiment notebooks (run in order)
    01_create_hal_config_and_shutters.ipynb        # define imaging sequence, write HAL XML and CSV
    02_create_positions_from_tissue_boundary.ipynb # generate FOV scanning grid
    03_create_dave_config.ipynb                    # generate round_info.csv and Dave recipe XML
    04_create_data_organization.ipynb              # MERlin data-organization setup
  analysis/         # Online-analysis notebooks (run during the experiment)
    01_fov_scheduler.ipynb                         # FOV-level scheduler (thumbnails, stats, histograms)
    02_round_scheduler.ipynb                       # round-level scheduler (mosaics, optional data transfer)
    03_view_mosaics.ipynb                          # display per-color mosaics as they are built
    04_view_intensity_stats.ipynb                  # plot per-frame intensity statistics over rounds
  misc/             # Ad-hoc utilities
    MF2_60XSil1.3_zcorrection.ipynb                # z-correction helper for the MF2 60x silicone objective
    reconstruct_frame_table_from_configs.ipynb     # inverse of prepare_imaging/01: hal+shutter XML -> frame_table CSV
data/
  configs/
    hal/            # hal-config-{mic}-epi.xml — HAL config templates (one per microscope)
    kilroy/         # kilroy-config-*-{mic}-*-{YYMMDD}.xml — Kilroy configs (one or more per microscope)
  positions/        # boundary_positions.txt, hole*.txt — example tissue boundary files
  readouts.csv      # default codebook readout table (bit number -> readout name), read by prepare_imaging/04
```

## Architecture

### Pre-experiment workflow

Run the four `prepare_imaging/` notebooks in order before starting the microscope.

**01** (`prepare_imaging/01_create_hal_config_and_shutters.ipynb`): defines the imaging sequence as a *frame table* (one row per camera frame, columns `color`, `channel`, `z`) using `get_frame_table`. Supports `scan_mode="interleaved"` (all colors per z-plane, AOTF) or `scan_mode="sequential"` (full z-sweep per color, boustrophedon, physical shutters). The objective's return to `bead_z` after the stack is controlled by `z_return_mode`: `"progressive"` (default) steps down with blank frames in increments of `return_step` (5 µm default); `"instant"` jumps straight back (the previous behaviour). Auto-generates a compact name via `get_color_sequence_name`. Sets `<filetype>` (`.zarr` default, or `.dax`/`.tiff`) and `<exposure_time>` in the HAL config. Writes:
- `SAMPLE_DIR/settings/hal-config-{microscope}-{name}.xml` — patched from `data/configs/hal/hal-config-{mic}-epi.xml`
- `SAMPLE_DIR/settings/shutter-{name}.xml` — shutter event XML
- `SAMPLE_DIR/metadata/frame_table_{name}.csv` — frame table
- `SAMPLE_DIR/metadata/shutter_sequence_{name}.png` — visualisation

Both XML files use Windows CRLF line endings and ISO-8859-1 encoding as required by HAL.

**02** (`prepare_imaging/02_create_positions_from_tissue_boundary.ipynb`): reads `boundary_positions.txt` and `hole*.txt` from `SAMPLE_DIR/positions/`, builds a regular boustrophedon FOV grid (`create_grid_positions` → `generate_scanning_path`), filters by polygon–polygon overlap (`filter_scanning_path`), reorders return points (`close_scanning_path`), and saves `SAMPLE_DIR/positions/positions_{SAMPLE_NAME}.txt`.

FOV grid rules: odd row and column count; centre FOV at bounding-box midpoint. A FOV is kept if its camera square overlaps the boundary polygon at all; excluded only if a hole polygon fully contains the FOV square.

**03** (`prepare_imaging/03_create_dave_config.ipynb`): generates `round_info.csv` and the Dave experiment recipe XML. HAL configs for bits vs. cells rounds are auto-detected by glob patterns (`blkf3*` for bits, `blkf1*` for cells). The Kilroy config for the microscope is resolved (via `find_kilroy_config`, falling back to MF2 when the microscope has no config) and passed to `create_dave_config` as the source of fluidic protocol names: every protocol written into the Dave recipe is resolved to — and required to exist as — a `<protocol>` in that Kilroy config, raising `ValueError` otherwise (e.g. adaptor mode on a Kilroy config lacking a readouts protocol). Writes `SAMPLE_DIR/metadata/round_info.csv` and `SAMPLE_DIR/settings/dave-{mic}-{N}bits-{SAMPLE_NAME}.xml`.

**04** (`prepare_imaging/04_create_data_organization.ipynb`): generates the MERlin data-organization CSV and annotates the Dave XML with per-round bit information. Requires `MERci/data/readouts.csv` (codebook mapping bit numbers to readout names; shipped in the repo). Frame tables and series patterns are auto-detected from `metadata/`. The user defines the `round_bit_color` mapping (round, bit, color_nm) to match the codebook. Writes:
- `SAMPLE_DIR/metadata/round_bit_color_map.csv`
- `SAMPLE_DIR/metadata/data_organization_{MICROSCOPE}_{SAMPLE_NAME}.csv`
- Annotates `SAMPLE_DIR/settings/dave-*.xml` with per-round bit comments

### Online-analysis architecture

`ExperimentConfig` holds all paths and tunable parameters. Notable fields:
- `image_suffix` — `.zarr` (default), `.dax`, or `.tiff`
- `fluidics_type` — `"adaptor"` (t_max = 100 min) or `"direct"` (t_max = 50 min); sets `t_max` automatically when left as `None`
- `settings_dir` — `SAMPLE_DIR/settings/`; needed for auto flip_y and per-color mosaic lookup
- `mosaic_flip_y` — `None` (auto-read from HAL config `<flip_vertical>`), `True`, or `False`
- `fov_subset` — list of FOV ids to restrict analysis; `None` = all FOVs
- `transfer_dest` — network path to copy raw data to during fluidics window; `None` = disabled
- `transfer_min_time` — minimum seconds remaining in the fluidics window before starting a transfer

`ExperimentMetadata` (loaded via `ExperimentMetadata.load(round_info_csv, positions_txt, data_dir)`) cross-references round IDs, FOV IDs, series patterns, and expected file paths. When a `dir`/`data_dir` column is present in `round_info.csv`, per-round file paths are resolved from that directory instead of the top-level `data_dir`. Each series carries an ordered list of **candidate directories** (`SeriesInfo.candidate_dirs`); `resolve_path(fov, suffix)` returns the first candidate that exists on disk, falling back to the primary one before acquisition. The **cells round** is treated as a bona fide imaging round (typically `imaging_round=1`) and its files are accepted in **either** `data/cells/` or the top-level `data/`, regardless of which the `data_dir` column records — so `all_fovs_done_for_round`, mosaics, and transfers all find the cells data wherever HAL actually wrote it.

`ExperimentStateMonitor` determines the microscope phase by watching the newest file mtime in `data_dir`:
- **IMAGING**: a new image file was written within `imaging_idle_threshold` seconds
- **FLUIDICS**: `t_min ≤ time_since_imaging ≤ t_max` → `should_analyze = True`

`ProgressTracker` tracks completeness via zero-byte sentinel files under `analysis_dir/done/`:
- `<stem>.fov_done` — FOV analysis complete
- `round_<r>.round_done` — mosaic(s) built for round r
- `round_<r>.round_transferred` — raw data for round r copied to `transfer_dest`

Multiple notebooks can run concurrently — no shared state.

`FOVScheduler.run_loop()` polls the phase, discovers stable image files (zarr/dax/tiff), and for each pending file generates thumbnails (PNG), per-frame stats (CSV), and histograms (`.npz`). Respects `fov_subset`. `RoundScheduler.run_loop()` assembles **one mosaic per imaging color** (`round_{r:03d}_{color}nm_mosaic.png`) once all FOV sentinels exist; auto-resolves `flip_y` from the HAL config; optionally launches background transfers via `transfer.transfer_round`. `ExperimentScheduler.wait_and_run()` calls a user callback after all rounds complete.

Typical scheduler setup in a notebook:

```python
from MERci.common.config   import ExperimentConfig
from MERci.common.metadata import ExperimentMetadata
from MERci.progress        import ProgressTracker
from MERci.state           import ExperimentStateMonitor
from MERci.scheduler       import FOVScheduler, RoundScheduler

config = ExperimentConfig(
    data_dir       = SAMPLE_DIR / "data",
    metadata_dir   = SAMPLE_DIR / "metadata",
    analysis_dir   = SAMPLE_DIR / "analysis",
    settings_dir   = SAMPLE_DIR / "settings",
    round_info_csv = SAMPLE_DIR / "metadata" / "round_info.csv",
    positions_txt  = SAMPLE_DIR / "positions" / f"positions_{SAMPLE_NAME}.txt",
    image_suffix   = ".zarr",          # or ".dax" / ".tiff"
    fluidics_type  = "adaptor",        # sets t_max = 100 min
)
meta    = ExperimentMetadata.load(config.round_info_csv, config.positions_txt, config.data_dir,
                                   image_suffix=config.image_suffix)
tracker = ProgressTracker(config.analysis_dir)
monitor = ExperimentStateMonitor(config)

FOVScheduler(config, meta, tracker, monitor).run_loop()
```

### Key data files

- `round_info.csv` — required columns: `imaging_round` (or legacy `round_id`), `series` (Python format string like `hal-mf3-epi_{fov:03d}_01`); optional: `imaging_type`, `hal_config`, `shutter_file`, `dir`. Loaded via `common.io.load_round_info`.
- `positions_{SAMPLE_NAME}.txt` — comma-separated `x,y` per line, one FOV per line; `#`-prefixed lines ignored
- Image files — HAL writes `.zarr` (directory store, default), `.dax` (raw uint16 binary + `.inf` sidecar), or `.tiff` (multi-page). Use `read_image(path)` to load any format. `discover_image_files` handles both flat files and zarr directory stores.
- HAL config templates — `data/configs/hal/hal-config-{mic}-epi.xml`; auto-detected in `prepare_imaging/01` by microscope name; patched by `create_hal_config` (sets frames, shutters, z_offsets, filetype, exposure_time)
- Kilroy config files — `data/configs/kilroy/kilroy-config-*-{mic}-*-{YYMMDD}.xml`; resolved in `prepare_imaging/03` by `find_kilroy_config` (newest by YYMMDD; falls back to MF2 when the microscope has no config), copied to `SAMPLE_DIR/settings/`, and used as the source of fluidic protocol names for the Dave recipe. Protocol names are **not** standardised across microscopes (e.g. `Cleave Adaptors` vs `Cleave Adaptor`), so `KilroyProtocolResolver` token-matches each logical dave step (cleave / hybridize k / readouts / image buffer) to the real `<protocol>` name in the chosen config.

### Microscope channel mapping

`MF2`, `MF3`, `MF4`, and `MF5` share the same 5-channel mapping: `{405→4, 488→3, 560→2, 650→1, 750→0}`. `MFX` has only 4 channels with a distinct ordering: `{650→0, 560→1, 488→2, 405→3}` (no 750). `NaN` = blank frame (no laser). Extend `_COLOUR_TO_CHANNEL` in `acquisition/configs.py` for other microscopes.

## Running notebooks

Notebooks auto-detect `SAMPLE_DIR` from their own location: `MERCI_DIR = Path(os.getcwd()).parent.parent` (the `MERci/` clone), then `SAMPLE_DIR = MERCI_DIR.parent`. This assumes notebooks are run from a second-level subfolder (`MERci/notebooks/<group>/`). Do not hardcode absolute paths in notebooks.

## Scope constraint

All edits and analysis must stay within this repo. Do not modify sibling folders (`image_acquisition/`, `imaging_with_storm_control/`, etc.) unless explicitly requested.

## Version control

Commit and push as you go — do not leave finished work uncommitted.

- After completing each individual modification, create its own focused git commit
  (one logical change per commit) with a clear message, then `git push` to `origin`.
- Do not batch many unrelated changes into a single commit, and do not let edits
  pile up locally — the remote (`github.com/leonardosepulveda/MERci`) should reflect
  progress as it happens.
- This is standing authorization to commit and push without asking each time.
- Never commit transient files: atomic-write leftovers (`*.tmp.*`), `__pycache__/`,
  or `*.egg-info/` (all gitignored).

## Remembering task history

For **every user question/request**, log it to `prompt_history/` (a gitignored,
local-only folder). This lets project context be reconstructed if the
conversation history is lost. There are **two methods** depending on how the
prompt arrives:

### Method 1 — user pre-writes the prompt as a file

The user creates a file in `prompt_history/` whose name is **only the date/time**
(e.g. `2026_06_18_1002.txt`) and writes their request inside it. When asked to read
and act on it:

1. Read the file and carry out the request.
2. **Append** the response record to the *same file* (the YAML frontmatter +
   `## Plan` + `## Summary` sections below; the `## Prompt` is already the file's
   existing content). Keep the original prompt text intact — append, never rewrite.
3. **Rename** the file to add a short description, converting it to the standard
   form `{YYYY_MM_DD_HH_MM}_{short_description}.md`
   (e.g. `2026_06_18_1002.txt` → `2026_06_18_1002_update_logging_rules.md`).

### Method 2 — user types the prompt in the Claude Code window

Create a **new** file per request, named `{YYYY_MM_DD_HH_MM}_{short_description}.md`
(e.g. `2026_06_04_1432_add_prompt_history_convention.md`), with the full template
below including a verbatim `## Prompt` copy.

### Shared format

Use YAML frontmatter for queryable metadata, then prose sections. Template:

```markdown
---
date: YYYY-MM-DD HH:MM
title: <short description>
files_modified:
  - path/relative/to/repo
status: completed | in-progress | abandoned
---

## Prompt
<verbatim copy of the user's request>

## Plan
<Claude's plan of action before executing>

## Summary
<what was actually done, including any deviations from the plan>
```

Format rationale: Markdown + YAML frontmatter is Claude-native, human-readable,
and lets all entries be scanned/grepped by metadata without reading every body.

