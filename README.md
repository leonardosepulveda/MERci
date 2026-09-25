# MERci

**MERci** (MERFISH acquisition + quality control) is a Python toolkit for planning and monitoring MERFISH spatial transcriptomics experiments on any microscope running the HAL/Dave/Kilroy/Steve software stack.

It generates the configuration files consumed by HAL (imaging), Kilroy (fluidics), and Dave (experiment orchestration), and runs a lightweight online quality-control analysis while the experiment is running.

---

## Setup

MERci is cloned directly into the experiment folder — no package installation is needed.

**Install Miniforge** (one-time, per computer): download and run the installer from
`https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe`

Miniforge provides `mamba`, a fast drop-in replacement for `conda`.

**Create the environment and register the kernel** (one-time, per computer):

```bash
mamba env create -f environment.yml
mamba activate merci_env
python -m ipykernel install --user --name merci_env --display-name "Python (merci_env)"
```

The kernel registration step makes `merci_env` visible in JupyterLab's kernel selector.  It only needs to be run once — JupyterLab can be launched from any environment afterwards.

**Open the notebooks:**

```bash
mamba activate merci_env
jupyter lab
```

Then navigate to `MERci/notebooks/` in the JupyterLab file browser. The notebooks are grouped into `before_imaging/` (pre-experiment setup, in `regular/` and `multi_z/` pipelines), `during_imaging/` (live QC), `after_imaging/` (online and post-acquisition analysis), `misc/` (ad-hoc utilities) and `tests/` (diagnostic and validation notebooks). Open notebooks from their own subfolder so that `SAMPLE_DIR` is auto-detected: each resolves `MERCI_DIR` by counting parent folders from its own location (2 levels for `after_imaging/`, `during_imaging/`, `misc/`, `tests/`; 3 for `before_imaging/<pipeline>/`) and `SAMPLE_DIR = MERCI_DIR.parent`.

### Updating an existing clone without overwriting your notebooks

Because MERci is cloned into each experiment folder, a clone where you have already run notebooks will have local changes (notebook outputs and any parameter edits). To pull the latest **package code** while leaving your notebooks exactly as you ran them, update only the source paths instead of doing a full `git pull`:

```bash
cd <experiment>/MERci
git fetch origin
git checkout origin/master -- src/ data/ environment.yml README.md CLAUDE.md
```

`git checkout origin/master -- <paths>` overwrites only the listed paths with the upstream version; `notebooks/` is untouched, so your runs and edits are preserved. (Caveat: any local edits you made *inside* the listed paths — e.g. to `src/` — would be overwritten, so check `git status` first.) Gitignored files such as `prompt_history/` and `settings.local.json` are never affected.

If you instead want to merge everything and review conflicts yourself, commit your local work and run `git pull origin master`; only files changed both locally and upstream (typically notebooks you ran) will conflict, and you can keep your version with `git checkout --ours <notebook>`.

---

## Experiment folder layout

```
SAMPLE_DIR/
  MERci/             clone of this repo
  positions/         boundaries/{manual,from_mosaic}/ + positions_{SAMPLE_NAME}.txt
  metadata/          frame_table_*.csv, round_info.csv, round_bit_color_map.csv,
                     data_organization_*.csv, experiment_info.yaml
  settings/          hal-config-*.xml, shutter-*.xml, dave-*.xml
  data/              raw image files (subfolders from round_info.csv's `dir` column)
  analysis/          thumbnails/, stats/, histograms/, mosaics/, done/
  merlin/            MERlin config/run files (or fishtank/ for lineage_tracing_lineage)
  figures/           MERlin's per-task verification figures
```

---

## Pre-experiment workflow

Run one pipeline's `before_imaging/` notebooks in order before starting the
microscope. Each writes inputs the next one reads (HAL/shutter configs →
positions → round_info → Dave recipe → data organization or color usage →
experiment_info → MERlin or fishtank scripts).

- `before_imaging/regular/`: the shared notebook set for `tumor_epi`,
  `tumor_disk`, `lineage_tracing_merfish` and `lineage_tracing_lineage`.
  Each pipeline's settings live in `data/pipelines/<id>_pipeline.yaml`.
  See [`notebooks/before_imaging/regular/README.md`](notebooks/before_imaging/regular/README.md).

- `before_imaging/multi_z/`: a variable-z-per-FOV acquisition with its own
  9-notebook sequence. See
  [`notebooks/before_imaging/multi_z/README.md`](notebooks/before_imaging/multi_z/README.md).

- `before_imaging/00_select_pipeline.ipynb` (optional) exports one
  pipeline's notebooks to a standalone `SAMPLE_DIR/notebooks/` folder.

---

## Online analysis

During the experiment, run the analysis notebooks in separate JupyterLab tabs to monitor quality in real time:

- `notebooks/after_imaging/01_fov_scheduler.ipynb` — FOV-level scheduler: thumbnails, per-frame stats, histograms
- `notebooks/after_imaging/02_round_scheduler.ipynb` — round-level scheduler: spatial mosaics, optional data transfer
- `notebooks/after_imaging/03`–`12` — mosaics, intensity stats, batch review, cluster submission, tissue thickness, completeness checks and more (see `CLAUDE.md`'s notebook index)
- `notebooks/during_imaging/` — live QC watched in real time (stage-z drift, imaged FOVs, quick-look mosaics, spot-intensity QC, Dave timing)

Standalone utility notebooks are also provided under `notebooks/misc/`:
- `MF2_60XSil1.3_zcorrection.ipynb` — z-correction for the MF2 60× silicone objective.
- `reconstruct_frame_table_from_configs.ipynb` — inverse of `before_imaging/01`: rebuild a `frame_table_*.csv` from an existing HAL config + its shutter file (recover a lost frame table or verify HAL/shutter consistency).
- `align_fovs_across_microscopes.ipynb` — transfer FOV positions to a second microscope after moving the stage insert. **Part 1** overlaps the two tissue-boundary polygons to fit an isotropic transform (scale + translation + optional x/y flips, no rotation). **Part 2** (optional) refines per-FOV residual drift from fiducial-bead images. Two methods (`METHOD`): `"phase"` = image `phase_cross_correlation` (the coarse-alignment primitive from [fishtank](https://github.com/jweissmanlab/fishtank); needs the two images to look alike), or `"beads"` = detect bead centroids in each image and register the point sets by consensus voting (modality-robust; each FOV gets a `score` = inlier fraction, low ⇒ the two images share no common beads, e.g. different focal planes). It writes drift-corrected positions plus diagnostic figures: a spatial per-FOV vector (quiver) plot, a drift x/y scatter + distance histogram, a drift-corrected FOV-layout plot (boundary + positions, à la `before_imaging/02`), and source-vs-target bead overlays (full FOV + center zoom) for FOVs sampled near the 10/25/50/75/90th drift percentiles.
- `extract_source_bead_frames.ipynb` — run at the **source** microscope before Part 2: writes a compact per-FOV `.tiff` containing only the full-resolution fiducial-bead frames (≈2 of ~30 frames) plus a matching compact frame table, so only the small bead files need to cross the NAS to the target microscope.

### How it works

Analysis runs **continuously** — during both acquisition and fluidics — and FOVs are processed **in parallel** across a pool of worker processes, so the analysis keeps up with long, data-heavy experiments instead of being limited to the fluidics gaps.

`ExperimentStateMonitor` still watches `data/` to tell imaging from fluidics, but only to time drive-bound background work (the mirror and the NAS transfer), not to gate analysis. `t_max` is set automatically from `fluidics_type` (`"adaptor"` → 100 min, `"direct"` → 50 min; override explicitly for a custom value).

**Analysis modes** (`config.analysis_mode`) control where analysis reads image data from, so its I/O need not fight the microscope's writes on a slow HDD:

| Mode | `analysis_mode` | Reads from | Best when |
|---|---|---|---|
| **B** | `"same_drive"` (default) | the acquisition drive (`data/`) | one fast drive, or contention is acceptable |
| **A** | `"mirror_drive"` | a 2nd drive (`analysis_source_dir`), mirrored from `data/` during fluidics | the acquisition drive is a slow HDD — analysis reads never touch it during acquisition |

**Parallelism** — `config.n_analysis_workers` worker processes (default `cpu_count − 2`) each handle one FOV: read the stack once, run all analyses, write outputs. Each worker holds one stack (~200 MB) in RAM, so lower it if memory is tight; set `1` to run serially in-process.

```
            Acquisition (hours)            Fluidics (~60–100 min)
═══════════════════════════════════════╪══════════════════════════╪═══ ...
  analysis runs the whole time            mirror (mode A) + NAS transfer
  (mode A: from 2nd-drive mirror;          run here, while the drive is idle
   mode B: from data/)
```

### Image format support

MERci reads `.zarr` (default), `.dax`, and `.tiff` image stacks.  The format is selected via `config.image_suffix` and must match what HAL is configured to write.

### FOV-level analysis (`FOVScheduler`)

Continuously discovers pending image files and dispatches one worker process per FOV (each reads the file once and runs every analysis). For each image file, produces:
- `analysis/thumbnails/{stem}_frame{n:03d}.png` — contrast-stretched thumbnails
- `analysis/stats/{stem}_stats.csv` — per-frame min/mean/median/max/std/p01/p99
- `analysis/histograms/{stem}_histograms.npz` — per-frame intensity histograms

### Round-level analysis (`RoundScheduler`)

Once all FOV sentinels exist for a round, assembles one spatial mosaic per imaging color (read from the frame table):
- `analysis/mosaics/round_{r:03d}_{color}nm_mosaic.png`

The `flip_y` orientation is read automatically from the `<flip_vertical>` field in the round's HAL config (override with `config.mosaic_flip_y`).

Progress is tracked via zero-byte sentinel files in `analysis/done/`.  Multiple schedulers can run concurrently without coordination.

### Data transfer (optional)

Set `transfer_dest` in `ExperimentConfig` to copy each round's raw data directory to a network destination (e.g. a NAS) during the fluidics window, using robocopy on Windows.  Transfer starts only when at least `transfer_min_time` seconds remain in the analysis window.

### FOV subset filtering (optional)

Set `fov_subset` to a list of FOV ids to restrict both the FOV scheduler and mosaic assembly to a subset of positions — useful for quick diagnostics or re-running a partial experiment.

### Typical notebook setup

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
    settings_dir   = SAMPLE_DIR / "settings",   # needed for auto flip_y and per-color mosaics
    round_info_csv = SAMPLE_DIR / "metadata" / "round_info.csv",
    positions_txt  = SAMPLE_DIR / "positions"  / f"positions_{SAMPLE_NAME}.txt",
    image_suffix   = ".zarr",                   # or ".dax" / ".tiff"
    fluidics_type  = "adaptor",                 # sets t_max = 100 min; use "direct" for 50 min
    # analysis_mode       = "mirror_drive",       # mode A: analyse from a 2nd drive (default "same_drive")
    # analysis_source_dir = Path(r"E:\merci_mirror\LT027\data"),  # required for mirror_drive
    # n_analysis_workers  = 6,                     # FOV worker processes (default cpu_count - 2; 1 = serial)
    # transfer_dest = Path(r"\\NAS\experiments"), # optional: copy data to NAS during fluidics window
    # fov_subset    = [0, 1, 2],                  # optional: restrict to a subset of FOVs
)
meta    = ExperimentMetadata.load(config.round_info_csv, config.positions_txt, config.data_dir,
                                   image_suffix=config.image_suffix)
tracker = ProgressTracker(config.analysis_dir)
monitor = ExperimentStateMonitor(config)

FOVScheduler(config, meta, tracker, monitor).run_loop()
```

---

## Package API

| Module | Key exports |
|---|---|
| `acquisition.configs` | `get_frame_table`, `get_color_sequence_name`, `get_color_to_channel_dict`, `create_shutter_file`, `create_hal_config`, `format_z_offsets_from_frame_table`, `read_hal_flip_vertical`, `find_frame_table_for_hal_config`, `get_color_frame_indices`, `reconstruct_frame_table`, `read_shutter_reference`, `parse_z_offsets`, `parse_shutter_events` |
| `acquisition.positions` | `create_grid_positions`, `generate_scanning_path`, `filter_scanning_path`, `close_scanning_path`, `load_hole_polygons`, `get_path_stats` |
| `acquisition.alignment` | `load_boundary_polygon`, `fit_isotropic_alignment`, `polygon_iou`, `AlignmentResult`, `bead_frame_indices`, `select_bead_frame`, `extract_bead_frames`, `apply_orientation`, `phase_drift`, `compute_fov_drifts`, `detect_beads`, `register_point_translation`, `compute_fov_drifts_beads` |
| `acquisition.dave` | `create_round_info`, `create_dave_config`, `annotate_dave_with_round_info`, `series_to_movie_name`, `get_hal_frame_count` |
| `acquisition.data_organization` | `create_data_organization` |
| `acquisition.display` | `print_frame_table`, `display_xml` |
| `common.config` | `ExperimentConfig` |
| `common.metadata` | `ExperimentMetadata`, `SeriesInfo`, `FOVInfo`, `RoundInfo` |
| `common.io` | `read_dax`, `read_zarr`, `read_tiff`, `read_image`, `parse_inf`, `get_dax_shape`, `load_round_info`, `load_positions`, `save_positions_array`, `discover_image_files` |
| `analysis.fov` | `create_thumbnail`, `create_thumbnails_for_stack`, `measure_stats`, `get_histogram`, `load_stats`, `load_histogram` |
| `analysis.round` | `create_mosaic`, `load_thumbnails_for_round` |
| `analysis.spot_localization` | `detect_beads_2d`, `fit_bead_3d`, `localize_beads_in_volume`, `localize_beads_in_file`, `match_beads_across_colors`, `compute_max_projection`, `plot_max_projections`, `simulate_multicolor_stack` (PSF/bead simulation + localization helpers) |
| `state` | `ExperimentStateMonitor`, `ExperimentPhase` |
| `progress` | `ProgressTracker` |
| `scheduler` | `FOVScheduler`, `RoundScheduler` |
| `transfer` | `transfer_round` |
| `visualization` | `visualize_shutter_sequence`, `plot_fov_layout`, `plot_stats_over_rounds`, `plot_spatial_uniformity`, `display_mosaic` |

---

## Key data files

### `round_info.csv`

Required columns: `imaging_round`, `series`  
Optional columns: `hal_config`, `dir`, `imaging_type`, `shutter_file`

```
imaging_round,series,hal_config,dir
1,hal-mf3_01_{fov:03d},hal-config-mf3-blkf3-488f1-560f49-650f49.xml,D:\experiments\my_sample\data\H01
1,hal-mf3-cells_{fov:03d},hal-config-mf3-blkf1-405f49-488f1.xml,D:\experiments\my_sample\data\cells
2,hal-mf3_02_{fov:03d},hal-config-mf3-blkf3-488f1-560f49-650f49.xml,D:\experiments\my_sample\data\H02
```

See the `round_info.csv` section above for column descriptions.

### `positions_{SAMPLE_NAME}.txt`

One `x,y` coordinate pair per line (stage units, µm).  Lines beginning with `#` are ignored.

### Image files

HAL can write images in three formats, selected by `<filetype>` in the HAL config:

| Format | Extension | Notes |
|---|---|---|
| Zarr | `.zarr/` | Directory store; default for new experiments |
| DAX | `.dax` | Raw uint16 binary; requires `.inf` sidecar |
| TIFF | `.tiff` | Multi-page TIFF |

Use `read_image(path)` to load any of the three formats without knowing the type in advance.

### Microscope channel mapping

**MF2, MF3, MF4, MF5** (5 channels):

| Wavelength (nm) | Channel index |
|---|---|
| 750 | 0 |
| 650 | 1 |
| 560 | 2 |
| 488 | 3 |
| 405 | 4 |
| blank | NaN |

**MFX, ST2** (4 channels, no 750, distinct ordering):

| Wavelength (nm) | Channel index |
|---|---|
| 650 | 0 |
| 560 | 1 |
| 488 | 2 |
| 405 | 3 |
| blank | NaN |

Add new microscopes to `_COLOUR_TO_CHANNEL` in `acquisition/configs.py`.
