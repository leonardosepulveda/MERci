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

Open notebooks from `MERci/notebooks/` in the JupyterLab file browser so that `SAMPLE_DIR` is auto-detected correctly.

## Deployment model

This repo is cloned directly into each new experiment folder (`SAMPLE_DIR`) as `SAMPLE_DIR/MERci/`. No `pip install` is needed — dependencies come from the `merci_env` conda environment. Notebooks add `src/` to `sys.path` automatically (`Path(os.getcwd()).parent / "src"` resolves to `MERci/src` when the notebook CWD is `MERci/notebooks/`). `SAMPLE_DIR` is `Path(os.getcwd()).parent.parent`.

## Experiment folder layout

```
SAMPLE_DIR/          (e.g. G:\LT048_sample_18\)
  MERci/             ← clone of this repo
  positions/         ← boundary_positions.txt, hole*.txt (from operator), positions_{SAMPLE_NAME}.txt (from nb 02)
  metadata/          ← frame_table_*.csv, shutter_sequence_*.png (nb 01), round_info.csv (nb 03)
  settings/          ← hal-config-*.xml, shutter-*.xml (nb 01), dave-*.xml (nb 03)
  data/              ← raw .dax files (written by HAL during acquisition)
    cells/           ← cells-round images
    H01/ … H0N/      ← bits-round images, one folder per round
  analysis/          ← thumbnails, stats, mosaics produced by MERci schedulers
```

## Package layout

```
src/MERci/
  common/
    config.py       # ExperimentConfig dataclass — all paths and tunable parameters
    metadata.py     # ExperimentMetadata — parses round_info.csv + positions.txt
    io.py           # read_dax, parse_inf, save_positions_array, discover_image_files
  acquisition/
    configs.py      # get_frame_table, get_color_sequence_name, create_shutter_file, create_hal_config
    positions.py    # create_grid_positions, generate_scanning_path, filter_scanning_path, close_scanning_path
    dave.py         # create_round_info, create_dave_config, series_to_movie_name
    data_organization.py  # data organization helpers
    display.py      # print_frame_table, display_xml (Jupyter helpers)
  analysis/
    fov.py          # create_thumbnail(s), measure_stats, get_histogram — FOV-level analysis
    round.py        # create_mosaic, load_thumbnails_for_round — round-level mosaic
  state.py          # ExperimentStateMonitor — detects imaging vs. fluidics phases by watching file mtimes
  progress.py       # ProgressTracker — file-system sentinel-based progress (no shared database)
  scheduler.py      # FOVScheduler, RoundScheduler, ExperimentScheduler — main analysis loops
  visualization.py  # visualize_shutter_sequence, plot_fov_layout, plot_stats_over_rounds, display_mosaic
notebooks/
  01_create_hal_config_and_shutters.ipynb  # Pre-experiment: define imaging sequence, write HAL XML and CSV
  02_create_positions_from_tissue_boundary.ipynb  # Pre-experiment: generate FOV scanning grid
  03_create_dave_config.ipynb             # Pre-experiment: generate round_info.csv and Dave recipe XML
  04_create_data_organization.ipynb       # Pre-experiment: data organization setup
data/
  templates/        # hal-config-mf3-epi.xml — HAL config template (microscope-specific)
  examples/         # round_info_example.csv — reference schema for round_info.csv
```

## Architecture

### Pre-experiment workflow

Run notebooks 01–03 in order before starting the microscope.

**Notebook 01** (`01_create_hal_config_and_shutters.ipynb`): defines the imaging sequence as a *frame table* (one row per camera frame, columns `color`, `channel`, `z`) using `get_frame_table`. Auto-generates a compact name via `get_color_sequence_name` (e.g. `blkf3-488f1-560f49-650f49`). Writes:
- `SAMPLE_DIR/settings/hal-config-{microscope}-{name}.xml` — patched from `data/templates/hal-config-mf3-epi.xml`
- `SAMPLE_DIR/settings/shutter-{name}.xml` — shutter event XML
- `SAMPLE_DIR/metadata/frame_table_{name}.csv` — frame table
- `SAMPLE_DIR/metadata/shutter_sequence_{name}.png` — visualisation

Both XML files use Windows CRLF line endings and ISO-8859-1 encoding as required by HAL.

**Notebook 02** (`02_create_positions_from_tissue_boundary.ipynb`): reads `boundary_positions.txt` and `hole*.txt` from `SAMPLE_DIR/positions/`, builds a regular boustrophedon FOV grid (`create_grid_positions` → `generate_scanning_path`), filters by polygon–polygon overlap (`filter_scanning_path`), reorders return points (`close_scanning_path`), and saves `SAMPLE_DIR/positions/positions_{SAMPLE_NAME}.txt`.

FOV grid rules: odd row and column count; centre FOV at bounding-box midpoint. A FOV is kept if its camera square overlaps the boundary polygon at all; excluded only if a hole polygon fully contains the FOV square.

**Notebook 03** (`03_create_dave_config.ipynb`): generates `round_info.csv` and the Dave experiment recipe XML. HAL configs for bits vs. cells rounds are auto-detected by glob patterns (`blkf3*` for bits, `blkf1*` for cells). Writes `SAMPLE_DIR/metadata/round_info.csv` and `SAMPLE_DIR/settings/dave-{mic}-{N}bits-{SAMPLE_NAME}.xml`.

### Online-analysis architecture

`ExperimentConfig` holds all paths and tunable parameters. `ExperimentMetadata` (loaded via `ExperimentMetadata.load(round_info_csv, positions_txt, data_dir)`) cross-references round IDs, FOV IDs, series patterns, and expected file paths. When a `dir` column is present in `round_info.csv`, per-round file paths are resolved from that directory instead of the top-level `data_dir`.

`ExperimentStateMonitor` determines the microscope phase by watching the newest file mtime in `data_dir`:
- **IMAGING**: a new `.dax` file was written within `imaging_idle_threshold` seconds
- **FLUIDICS**: `t_min ≤ time_since_imaging ≤ t_max` → `should_analyze = True`

`ProgressTracker` tracks completeness via zero-byte sentinel files (`<stem>.fov_done`, `round_<r>.round_done`) under `analysis_dir/done/`. Multiple notebooks can run concurrently — no shared state.

`FOVScheduler.run_loop()` polls the phase, discovers stable `.dax` files, and for each pending file generates thumbnails (PNG), per-frame stats (CSV), and histograms (`.npz`). `RoundScheduler.run_loop()` assembles a spatial mosaic once all FOV sentinels exist. `ExperimentScheduler.wait_and_run()` calls a user callback after all rounds complete.

Typical scheduler setup in a notebook:

```python
from MERci.common.config   import ExperimentConfig
from MERci.common.metadata import ExperimentMetadata
from MERci.progress        import ProgressTracker
from MERci.state           import ExperimentStateMonitor
from MERci.scheduler       import FOVScheduler, RoundScheduler

config  = ExperimentConfig(data_dir=..., metadata_dir=..., analysis_dir=...,
                           round_info_csv=..., positions_txt=...)
meta    = ExperimentMetadata.load(config.round_info_csv, config.positions_txt, config.data_dir)
tracker = ProgressTracker(config.analysis_dir)
monitor = ExperimentStateMonitor(config)

FOVScheduler(config, meta, tracker, monitor).run_loop()
```

### Key data files

- `round_info.csv` — required columns: `imaging_round` (or legacy `round_id`), `series` (Python format string like `hal-mf3-epi_{fov:03d}_01`); optional: `imaging_type`, `hal_config`, `shutter_file`, `dir`. See `data/examples/round_info_example.csv`.
- `positions_{SAMPLE_NAME}.txt` — comma-separated `x,y` per line, one FOV per line; `#`-prefixed lines ignored
- `.dax` files — raw uint16 binary image stacks; dimensions resolved from `.inf` sidecar (same stem)
- `hal-config-mf3-epi.xml` — HAL config template in `data/templates/`; patched by `create_hal_config`

### Microscope channel mapping

Both `MF3` and `MF5`: `{405→4, 488→3, 560→2, 650→1, 750→0}`. `NaN` = blank frame (no laser). Extend in `acquisition/configs.py` for other microscopes.

## Running notebooks

Notebooks auto-detect `SAMPLE_DIR` via `Path(os.getcwd()).parent.parent`. Do not hardcode absolute paths in notebooks.

## Scope constraint

All edits and analysis must stay within this repo. Do not modify sibling folders (`image_acquisition/`, `imaging_with_storm_control/`, etc.) unless explicitly requested.
