# MERci

**MERci** (MERFISH acquisition + quality control) is a Python toolkit for planning and monitoring MERFISH spatial transcriptomics experiments on any microscope running the HAL/Dave/Kilroy/Steve software stack.

It generates the configuration files consumed by HAL (imaging), Kilroy (fluidics), and Dave (experiment orchestration), and runs a lightweight online quality-control analysis while the experiment is running.

---

## Setup

MERci is cloned directly into the experiment folder — no package installation is needed.

**Install Miniforge** (one-time, per computer): download and run the installer from
`https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe`

Miniforge provides `mamba`, a fast drop-in replacement for `conda`.

**Create the environment** (one-time, per computer):

```bash
mamba env create -f environment.yml
```

**Open the notebooks:**

```bash
mamba activate merci_env
jupyter lab
```

Then navigate to `MERci/notebooks/` in the JupyterLab file browser. Open notebooks from that folder so that `SAMPLE_DIR` is auto-detected correctly.

---

## Experiment folder layout

```
SAMPLE_DIR/                          e.g.  G:\LT048_sample_18\
  MERci/                             ← clone of this repo
  positions/
    boundary_positions.txt           ← tissue boundary (from microscope operator)
    hole*.txt                        ← exclusion regions (from microscope operator)
    positions_{SAMPLE_NAME}.txt      ← FOV grid (output of notebook 02)
  metadata/
    frame_table_{name}.csv           ← frame sequence table (output of notebook 01)
    shutter_sequence_{name}.png      ← visual summary (output of notebook 01)
    round_info.csv                   ← per-round imaging metadata (output of notebook 03)
  settings/
    hal-config-{mic}-{name}.xml      ← HAL imaging config (output of notebook 01)
    shutter-{name}.xml               ← HAL shutter config (output of notebook 01)
    dave-{mic}-{N}bits-{name}.xml    ← Dave recipe (output of notebook 03)
  data/
    cells/                           ← cells-round .dax files (written by HAL)
    H01/ … H0N/                      ← bits-round .dax files (written by HAL)
  analysis/                          ← thumbnails, stats, mosaics (output of schedulers)
```

---

## Pre-experiment workflow

Run the three notebooks in order before starting the microscope.  Each notebook auto-detects `SAMPLE_DIR` from its own location (`MERci/notebooks/`), so no paths need to be changed.

### Notebook 01 — HAL configs and shutter files

`notebooks/01_create_hal_config_and_shutters.ipynb`

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

**Outputs** (written to `SAMPLE_DIR/settings/` and `SAMPLE_DIR/metadata/`):
- `hal-config-{mic}-{name}.xml` — HAL imaging config, patched from the template in `MERci/data/templates/`
- `shutter-{name}.xml` — HAL shutter event sequence
- `frame_table_{name}.csv` — frame table used by the analysis modules
- `shutter_sequence_{name}.png` — visual summary for verification

The compact config name (e.g. `blkf3-488f1-560f49-650f49`) is auto-generated from the frame sequence.

---

### Notebook 02 — FOV positions

`notebooks/02_create_positions_from_tissue_boundary.ipynb`

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

`notebooks/03_create_dave_config.ipynb`

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

---

## Online analysis

During the experiment, run the analysis schedulers in separate notebook cells (or notebooks) to monitor quality in real time.

### How it works

`ExperimentStateMonitor` watches `data/` for new `.dax` files.  When imaging finishes and the microscope enters the fluidics step, the analysis window `[t_min, t_max]` opens and the schedulers process all pending files.

```
Imaging ends          t_min          t_max     Next round starts
     │                  │              │              │
─────┼──────────────────┼──────────────┼──────────────┼────
                        └── analysis window ──┘
```

### FOV-level analysis (`FOVScheduler`)

For each `.dax` file, produces:
- `analysis/thumbnails/{stem}_frame{n:03d}.png` — contrast-stretched thumbnails
- `analysis/stats/{stem}_stats.csv` — per-frame min/mean/median/max/std/p01/p99
- `analysis/histograms/{stem}_histograms.npz` — per-frame intensity histograms

### Round-level analysis (`RoundScheduler`)

Once all FOV sentinels exist for a round, assembles:
- `analysis/mosaics/round_{r:03d}_mosaic.png` — spatial mosaic of all FOV thumbnails

Progress is tracked via zero-byte sentinel files in `analysis/done/`.  Multiple schedulers can run concurrently without coordination.

### Typical notebook setup

```python
from MERci.common.config   import ExperimentConfig
from MERci.common.metadata import ExperimentMetadata
from MERci.progress        import ProgressTracker
from MERci.state           import ExperimentStateMonitor
from MERci.scheduler       import FOVScheduler, RoundScheduler

config  = ExperimentConfig(data_dir=..., metadata_dir=..., analysis_dir=...,
                           round_info_csv=..., positions_txt=...)
meta    = ExperimentMetadata.load(config.round_info_csv, config.positions_txt,
                                  config.data_dir)
tracker = ProgressTracker(config.analysis_dir)
monitor = ExperimentStateMonitor(config)

FOVScheduler(config, meta, tracker, monitor).run_loop()
```

---

## Package API

| Module | Key exports |
|---|---|
| `acquisition.configs` | `get_frame_table`, `get_color_sequence_name`, `create_shutter_file`, `create_hal_config` |
| `acquisition.positions` | `create_grid_positions`, `generate_scanning_path`, `filter_scanning_path`, `close_scanning_path`, `load_hole_polygons`, `get_path_stats` |
| `acquisition.dave` | `create_round_info`, `create_dave_config`, `series_to_movie_name` |
| `common.config` | `ExperimentConfig` |
| `common.metadata` | `ExperimentMetadata`, `SeriesInfo`, `FOVInfo`, `RoundInfo` |
| `common.io` | `read_dax`, `parse_inf`, `get_dax_shape`, `save_positions_array`, `discover_image_files` |
| `analysis.fov` | `create_thumbnail`, `create_thumbnails_for_stack`, `measure_stats`, `get_histogram` |
| `analysis.round` | `create_mosaic`, `load_thumbnails_for_round` |
| `state` | `ExperimentStateMonitor`, `ExperimentPhase` |
| `progress` | `ProgressTracker` |
| `scheduler` | `FOVScheduler`, `RoundScheduler`, `ExperimentScheduler` |
| `visualization` | `visualize_shutter_sequence`, `plot_fov_layout`, `plot_stats_over_rounds`, `display_mosaic` |

---

## Key data files

### `round_info.csv`

Required columns: `imaging_round`, `series`  
Optional columns: `hal_config`, `dir`, `imaging_type`, `shutter_file`

```
imaging_round,series,hal_config,dir
1,hal-mf3-epi_01_{fov:03d},hal-config-mf3-blkf3-488f1-560f49-650f49.xml,G:\sample\data\H01
1,hal-mf3-epi_cells_{fov:03d},hal-config-mf3-blkf1-405f49-488f1.xml,G:\sample\data\cells
2,hal-mf3-epi_02_{fov:03d},hal-config-mf3-blkf3-488f1-560f49-650f49.xml,G:\sample\data\H02
```

See `data/examples/round_info_example.csv` for a complete example.

### `positions_{SAMPLE_NAME}.txt`

One `x,y` coordinate pair per line (stage units, µm).  Lines beginning with `#` are ignored.

### `.dax` / `.inf` files

Raw uint16 binary image stacks written by HAL.  Frame dimensions are read from the `.inf` sidecar (same stem, same directory).

### Microscope channel mapping

| Wavelength (nm) | Channel index |
|---|---|
| 750 | 0 |
| 650 | 1 |
| 560 | 2 |
| 488 | 3 |
| 405 | 4 |
| blank | NaN |

This is the default mapping; it can be extended in `acquisition/configs.py` for other microscopes.
