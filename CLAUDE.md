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
  positions/         ← boundary_positions.txt, hole*.txt (from operator),
                        positions_{SAMPLE_NAME}.txt (from setup_02)
  metadata/          ← frame_table_*.csv, shutter_sequence_*.png (setup_01),
                        round_info.csv (setup_03),
                        round_bit_color_map.csv, data_organization_*.csv (setup_04)
  settings/          ← hal-config-*.xml, shutter-*.xml (setup_01), dave-*.xml (setup_03)
  readouts.csv       ← codebook readout table (user-provided, required by setup_04)
  data/              ← raw image files; exact subfolder structure defined by the `dir`
                        column in round_info.csv (written by HAL during acquisition)
  analysis/          ← thumbnails/, stats/, histograms/, mosaics/, done/
                        (produced by analysis_01 and analysis_02 schedulers)
```

## Package layout

```
src/MERci/
  common/
    config.py       # ExperimentConfig dataclass — all paths and tunable parameters
    metadata.py     # ExperimentMetadata — parses round_info.csv + positions.txt
    io.py           # read_dax/zarr/tiff/image, parse_inf, save_positions_array, discover_image_files
  acquisition/
    configs.py      # get_frame_table, get_color_sequence_name, create_shutter_file, create_hal_config
                    # + read_hal_flip_vertical, find_frame_table_for_hal_config, get_color_frame_indices
    positions.py    # create_grid_positions, generate_scanning_path, filter_scanning_path, close_scanning_path
    dave.py         # create_round_info, create_dave_config, series_to_movie_name
    data_organization.py  # data organization helpers
    display.py      # print_frame_table, display_xml (Jupyter helpers)
  analysis/
    fov.py          # create_thumbnail(s), measure_stats, get_histogram — FOV-level analysis
    round.py        # create_mosaic, load_thumbnails_for_round — round-level mosaic
  state.py          # ExperimentStateMonitor — detects imaging vs. fluidics phases by watching file mtimes
  progress.py       # ProgressTracker — sentinel files for fov_done, round_done, round_transferred
  scheduler.py      # FOVScheduler, RoundScheduler, ExperimentScheduler — main analysis loops
  transfer.py       # transfer_round — background robocopy/shutil to a network destination
  visualization.py  # visualize_shutter_sequence, plot_fov_layout, plot_stats_over_rounds, display_mosaic
notebooks/
  setup_01_create_hal_config_and_shutters.ipynb        # Pre-experiment: define imaging sequence, write HAL XML and CSV
  setup_02_create_positions_from_tissue_boundary.ipynb # Pre-experiment: generate FOV scanning grid
  setup_03_create_dave_config.ipynb                    # Pre-experiment: generate round_info.csv and Dave recipe XML
  setup_04_create_data_organization.ipynb              # Pre-experiment: data organization setup
  analysis_01_fov_scheduler.ipynb                      # Online: FOV-level scheduler (thumbnails, stats, histograms)
  analysis_02_round_scheduler.ipynb                    # Online: round-level scheduler (mosaics, optional data transfer)
  analysis_03_view_mosaics.ipynb                       # Online: display per-color mosaics as they are built
  analysis_04_view_intensity_stats.ipynb               # Online: plot per-frame intensity statistics over rounds
data/
  configs/
    hal/            # hal-config-{mic}-epi.xml — HAL config templates (one per microscope)
    kilroy/         # kilroy-config-*-{mic}-*-{YYMMDD}.xml — Kilroy configs (one or more per microscope)
  positions/        # boundary_positions.txt, hole*.txt — example tissue boundary files
```

## Architecture

### Pre-experiment workflow

Run the four setup notebooks in order before starting the microscope.

**setup_01** (`setup_01_create_hal_config_and_shutters.ipynb`): defines the imaging sequence as a *frame table* (one row per camera frame, columns `color`, `channel`, `z`) using `get_frame_table`. Supports `scan_mode="interleaved"` (all colors per z-plane, AOTF) or `scan_mode="sequential"` (full z-sweep per color, boustrophedon, physical shutters). Auto-generates a compact name via `get_color_sequence_name`. Sets `<filetype>` (`.zarr` default, or `.dax`/`.tiff`) and `<exposure_time>` in the HAL config. Writes:
- `SAMPLE_DIR/settings/hal-config-{microscope}-{name}.xml` — patched from `data/templates/hal-config-mf3-epi.xml`
- `SAMPLE_DIR/settings/shutter-{name}.xml` — shutter event XML
- `SAMPLE_DIR/metadata/frame_table_{name}.csv` — frame table
- `SAMPLE_DIR/metadata/shutter_sequence_{name}.png` — visualisation

Both XML files use Windows CRLF line endings and ISO-8859-1 encoding as required by HAL.

**setup_02** (`setup_02_create_positions_from_tissue_boundary.ipynb`): reads `boundary_positions.txt` and `hole*.txt` from `SAMPLE_DIR/positions/`, builds a regular boustrophedon FOV grid (`create_grid_positions` → `generate_scanning_path`), filters by polygon–polygon overlap (`filter_scanning_path`), reorders return points (`close_scanning_path`), and saves `SAMPLE_DIR/positions/positions_{SAMPLE_NAME}.txt`.

FOV grid rules: odd row and column count; centre FOV at bounding-box midpoint. A FOV is kept if its camera square overlaps the boundary polygon at all; excluded only if a hole polygon fully contains the FOV square.

**setup_03** (`setup_03_create_dave_config.ipynb`): generates `round_info.csv` and the Dave experiment recipe XML. HAL configs for bits vs. cells rounds are auto-detected by glob patterns (`blkf3*` for bits, `blkf1*` for cells). Writes `SAMPLE_DIR/metadata/round_info.csv` and `SAMPLE_DIR/settings/dave-{mic}-{N}bits-{SAMPLE_NAME}.xml`.

**setup_04** (`setup_04_create_data_organization.ipynb`): generates the MERlin data-organization CSV and annotates the Dave XML with per-round bit information. Requires `SAMPLE_DIR/readouts.csv` (codebook mapping bit numbers to readout names). Frame tables and series patterns are auto-detected from `metadata/`. The user defines the `round_bit_color` mapping (round, bit, color_nm) to match the codebook. Writes:
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

`ExperimentMetadata` (loaded via `ExperimentMetadata.load(round_info_csv, positions_txt, data_dir)`) cross-references round IDs, FOV IDs, series patterns, and expected file paths. When a `dir` column is present in `round_info.csv`, per-round file paths are resolved from that directory instead of the top-level `data_dir`.

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

- `round_info.csv` — required columns: `imaging_round` (or legacy `round_id`), `series` (Python format string like `hal-mf3-epi_{fov:03d}_01`); optional: `imaging_type`, `hal_config`, `shutter_file`, `dir`. See `data/examples/round_info_example.csv`.
- `positions_{SAMPLE_NAME}.txt` — comma-separated `x,y` per line, one FOV per line; `#`-prefixed lines ignored
- Image files — HAL writes `.zarr` (directory store, default), `.dax` (raw uint16 binary + `.inf` sidecar), or `.tiff` (multi-page). Use `read_image(path)` to load any format. `discover_image_files` handles both flat files and zarr directory stores.
- HAL config templates — `data/configs/hal/hal-config-{mic}-epi.xml`; auto-detected in setup_01 by microscope name; patched by `create_hal_config` (sets frames, shutters, z_offsets, filetype, exposure_time)
- Kilroy config files — `data/configs/kilroy/kilroy-config-*-{mic}-*-{YYMMDD}.xml`; auto-detected in setup_03 by microscope name; newest file (by YYMMDD) is copied to `SAMPLE_DIR/settings/`

### Microscope channel mapping

`MF2`, `MF3`, and `MF5` all share the same mapping: `{405→4, 488→3, 560→2, 650→1, 750→0}`. `NaN` = blank frame (no laser). Extend `_COLOUR_TO_CHANNEL` in `acquisition/configs.py` for other microscopes.

## Running notebooks

Notebooks auto-detect `SAMPLE_DIR` via `Path(os.getcwd()).parent.parent`. Do not hardcode absolute paths in notebooks.

## Scope constraint

All edits and analysis must stay within this repo. Do not modify sibling folders (`image_acquisition/`, `imaging_with_storm_control/`, etc.) unless explicitly requested.
