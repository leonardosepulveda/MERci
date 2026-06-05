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

Then navigate to `MERci/notebooks/` in the JupyterLab file browser. The notebooks are grouped into `prepare_imaging/` (pre-experiment setup), `analysis/` (online monitoring), and `misc/` (ad-hoc utilities). Open notebooks from their subfolder so that `SAMPLE_DIR` is auto-detected correctly — each notebook resolves `MERCI_DIR = Path(os.getcwd()).parent.parent` and `SAMPLE_DIR = MERCI_DIR.parent`, which assumes it is run from a second-level subfolder (`MERci/notebooks/<group>/`).

---

## Experiment folder layout

```
SAMPLE_DIR/                          e.g.  D:\experiments\my_sample\
  MERci/                             ← clone of this repo
  positions/
    boundary_positions.txt           ← tissue boundary (from microscope operator)
    hole*.txt                        ← exclusion regions (from microscope operator)
    positions_{SAMPLE_NAME}.txt      ← FOV grid (output of prepare_imaging/02)
  metadata/
    frame_table_{name}.csv           ← frame sequence table (output of prepare_imaging/01)
    shutter_sequence_{name}.png      ← visual summary (output of prepare_imaging/01)
    round_info.csv                   ← per-round imaging metadata (output of prepare_imaging/03)
    round_bit_color_map.csv          ← round → bit → color mapping (output of prepare_imaging/04)
    data_organization_{mic}_{name}.csv  ← MERlin data-organization file (output of prepare_imaging/04)
  settings/
    hal-config-{mic}-{name}.xml      ← HAL imaging config (output of prepare_imaging/01)
    shutter-{name}.xml               ← HAL shutter config (output of prepare_imaging/01)
    dave-{mic}-{N}bits-{name}.xml    ← Dave recipe (output of prepare_imaging/03)
  readouts.csv                       ← codebook readout table (user-provided, required by prepare_imaging/04)
  data/                              ← raw image files; exact subfolder structure is defined in
                                        round_info.csv via the `dir` column (written by HAL)
  analysis/                          ← outputs of the online-analysis schedulers
    thumbnails/                      ← per-frame PNG thumbnails
    stats/                           ← per-frame intensity stats (CSV)
    histograms/                      ← per-frame intensity histograms (.npz)
    mosaics/                         ← spatial mosaics per round and color
    done/                            ← zero-byte sentinel files tracking analysis progress
```

---

## Pre-experiment workflow

Run the four `prepare_imaging/` notebooks in order before starting the microscope.  Each notebook auto-detects `SAMPLE_DIR` from its own location (`MERci/notebooks/prepare_imaging/`), so no paths need to be changed.

### Notebook 01 — HAL configs and shutter files

`notebooks/prepare_imaging/01_create_hal_config_and_shutters.ipynb`

Defines the per-frame imaging sequence and writes the HAL configuration files.

Key parameters to set:

| Variable | Description | Example |
|---|---|---|
| `MICROSCOPE` | Microscope identifier | `"my_scope"` |
| `z_bead` | z position for fiducial bead frames (µm) | `0` |
| `bead_seq` | Laser colours for bead frames | `[488, np.nan]` |
| `color_seq` | Laser colours for data frames | `[560, 650]` |
| `end_seq` | Blank frames at end of stack | `[np.nan, np.nan]` |
| `z_pos` | z positions for the data stack (µm) | `np.arange(1, 20.5, 0.5)` |
| `FILE_TYPE` | Image format written by HAL | `".zarr"` (default), `".dax"`, `".tiff"` |
| `EXPOSURE_TIME` | Camera exposure time (seconds) | `0.25` |

**Outputs** (written to `SAMPLE_DIR/settings/` and `SAMPLE_DIR/metadata/`):
- `hal-config-{mic}-{name}.xml` — HAL imaging config, patched from the template in `MERci/data/configs/hal/`
- `shutter-{name}.xml` — HAL shutter event sequence
- `frame_table_{name}.csv` — frame table used by the analysis modules
- `shutter_sequence_{name}.png` — visual summary for verification

The compact config name (e.g. `blkf3-488f1-560f49-650f49`) is auto-generated from the frame sequence.

The HAL template is auto-detected from `MERci/data/configs/hal/` by matching the `MICROSCOPE` name (e.g. `hal-config-mf3-epi.xml` for `MICROSCOPE = "MF3"`). A `FileNotFoundError` is raised if no matching template exists.

---

### Notebook 02 — FOV positions

`notebooks/prepare_imaging/02_create_positions_from_tissue_boundary.ipynb`

Builds a regular boustrophedon FOV grid within the tissue boundary.

Key parameters to set:

| Variable | Description | Default |
|---|---|---|
| `pixel_size_um` | Camera pixel size (µm) | `0.109` |
| `image_size_px` | Frame size (pixels, one side) | `2048` |
| `non_overlap_fraction` | Fractional FOV covered per step | `0.9` |
| `return_side` | Which grid edge to put last | `"top"` |

Reads `boundary_positions.txt` and any `hole*.txt` files from `SAMPLE_DIR/positions/`.  A FOV is included if its camera square overlaps the tissue boundary at all, and excluded only if a hole polygon fully contains it.

**Output:** `SAMPLE_DIR/positions/positions_{SAMPLE_NAME}.txt`

---

### Notebook 03 — Dave recipe

`notebooks/prepare_imaging/03_create_dave_config.ipynb`

Generates the `round_info.csv` table and the Dave experiment recipe XML.

Key parameters to set:

| Variable | Description | Example |
|---|---|---|
| `MICROSCOPE` | Microscope identifier | `"my_scope"` |
| `N_BITS` | Number of hybridisation rounds | `8` |
| `BITS_HAL_CONFIG` | HAL config for bits rounds | auto-detected |
| `CELLS_HAL_CONFIG` | HAL config for cells round | auto-detected |

HAL configs are auto-detected by glob patterns (`blkf3*` for bits, `blkf1*` for cells).  Override manually if needed.

**Experiment structure encoded in the Dave recipe:**

```
Round 1 imaging:   bits (all FOVs) → cells (all FOVs)
Fluidics 1:        Cleave direct → Hybridize 1 → Wash and Imaging Buffers
Round 2 imaging:   bits (all FOVs)
Fluidics 2:        Cleave direct → Hybridize 2 → Wash and Imaging Buffers
…
Round N imaging:   bits (all FOVs)
Final fluidics:    Cleave direct
```

**Outputs:**
- `SAMPLE_DIR/metadata/round_info.csv`
- `SAMPLE_DIR/settings/dave-{mic}-{N}bits-{SAMPLE_NAME}.xml`
- `SAMPLE_DIR/settings/kilroy-config-*-{mic}-*-{YYMMDD}.xml` — copied from `MERci/data/configs/kilroy/` (newest file matching the microscope name, if present)

---

### Notebook 04 — MERlin data organization

`notebooks/prepare_imaging/04_create_data_organization.ipynb`

Generates the MERlin `data_organization_*.csv` file that maps each bit to its images, z-positions, and fiducial frames.  Also annotates the Dave XML produced by notebook 03 with per-round bit information.

**Required input:** `SAMPLE_DIR/readouts.csv` — a table mapping bit numbers to readout names, with columns `Bit number` and `Name`.

Key parameters to set:

| Variable | Description |
|---|---|
| `round_bit_color` | List of `(round, bit, color_nm)` tuples matching the experiment codebook |

Frame tables and series patterns are auto-detected from `metadata/frame_table_*.csv` and `round_info.csv`.

**Outputs:**
- `SAMPLE_DIR/metadata/round_bit_color_map.csv` — round → bit → color mapping table
- `SAMPLE_DIR/metadata/data_organization_{MICROSCOPE}_{SAMPLE_NAME}.csv` — MERlin data-organization file
- `SAMPLE_DIR/settings/dave-*.xml` annotated with per-round bit comments

---

## Online analysis

During the experiment, run the analysis notebooks in separate JupyterLab tabs to monitor quality in real time:

- `notebooks/analysis/01_fov_scheduler.ipynb` — FOV-level scheduler: thumbnails, per-frame stats, histograms
- `notebooks/analysis/02_round_scheduler.ipynb` — round-level scheduler: spatial mosaics, optional data transfer
- `notebooks/analysis/03_view_mosaics.ipynb` — displays per-color mosaics as they are built
- `notebooks/analysis/04_view_intensity_stats.ipynb` — plots per-frame intensity statistics over rounds

Standalone utility notebooks are also provided under `notebooks/misc/`:
- `MF2_60XSil1.3_zcorrection.ipynb` — z-correction for the MF2 60× silicone objective.
- `reconstruct_frame_table_from_configs.ipynb` — inverse of `prepare_imaging/01`: rebuild a `frame_table_*.csv` from an existing HAL config + its shutter file (recover a lost frame table or verify HAL/shutter consistency).

### How it works

`ExperimentStateMonitor` watches `data/` for new image files.  When imaging finishes and the microscope enters the fluidics step, the analysis window `[t_min, t_max]` opens and the schedulers process all pending files.

```
Imaging ends          t_min          t_max     Next round starts
     │                  │              │              │
─────┼──────────────────┼──────────────┼──────────────┼────
                        └── analysis window ──┘
```

`t_max` is set automatically from `fluidics_type`:
- `"adaptor"` (default) → 100 min
- `"direct"` → 50 min

Override `t_max` explicitly to use a custom value.

### Image format support

MERci reads `.zarr` (default), `.dax`, and `.tiff` image stacks.  The format is selected via `config.image_suffix` and must match what HAL is configured to write.

### FOV-level analysis (`FOVScheduler`)

For each image file, produces:
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
    # transfer_dest = Path(r"\\NAS\experiments"), # optional: copy data during fluidics window
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
| `scheduler` | `FOVScheduler`, `RoundScheduler`, `ExperimentScheduler` |
| `transfer` | `transfer_round` |
| `visualization` | `visualize_shutter_sequence`, `plot_fov_layout`, `plot_stats_over_rounds`, `plot_spatial_uniformity`, `display_mosaic` |

---

## Key data files

### `round_info.csv`

Required columns: `imaging_round`, `series`  
Optional columns: `hal_config`, `dir`, `imaging_type`, `shutter_file`

```
imaging_round,series,hal_config,dir
1,hal-mf3-epi_01_{fov:03d},hal-config-mf3-blkf3-488f1-560f49-650f49.xml,D:\experiments\my_sample\data\H01
1,hal-mf3-epi_cells_{fov:03d},hal-config-mf3-blkf1-405f49-488f1.xml,D:\experiments\my_sample\data\cells
2,hal-mf3-epi_02_{fov:03d},hal-config-mf3-blkf3-488f1-560f49-650f49.xml,D:\experiments\my_sample\data\H02
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

**MF2, MF3, MF5** (5 channels):

| Wavelength (nm) | Channel index |
|---|---|
| 750 | 0 |
| 650 | 1 |
| 560 | 2 |
| 488 | 3 |
| 405 | 4 |
| blank | NaN |

**MFX** (4 channels, no 750, distinct ordering):

| Wavelength (nm) | Channel index |
|---|---|
| 650 | 0 |
| 560 | 1 |
| 488 | 2 |
| 405 | 3 |
| blank | NaN |

Add new microscopes to `_COLOUR_TO_CHANNEL` in `acquisition/configs.py`.
