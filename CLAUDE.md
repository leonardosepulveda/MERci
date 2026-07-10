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

This repo is cloned directly into each new experiment folder (`SAMPLE_DIR`) as `SAMPLE_DIR/MERci/`. No `pip install` is needed — dependencies come from the `merci_env` conda environment. Most notebooks live two levels under the repo root (e.g. `MERci/notebooks/analysis/`), so they resolve paths as:

```python
MERCI_DIR  = Path(os.getcwd()).parent.parent   # MERci/
SAMPLE_DIR = MERCI_DIR.parent                   # experiment root
sys.path.insert(0, str(MERCI_DIR / "src"))      # MERci/src
```

The `prepare_imaging/` notebooks live **three** levels under the repo root
(`MERci/notebooks/prepare_imaging/<variant>/`, where `<variant>` is
`reference`), so they use
`MERCI_DIR = Path(os.getcwd()).parent.parent.parent`.

**Exceptions — `tumor` and `lineage_tracing`** are each split into two acquisition
types, one full copy of the four notebooks per type: `tumor/epi/` and `tumor/disk/`
(epifluorescence vs. spinning-disk confocal, same single-tissue-section layout);
`lineage_tracing/merfish/` and `lineage_tracing/lineage/` (MERFISH readout vs.
lineage-barcode readout, same multi-tissue layout). Those notebooks live **four**
levels deep, so they use `MERCI_DIR = Path(os.getcwd()).parent.parent.parent.parent`.

## Experiment folder layout

```
SAMPLE_DIR/          (the experiment root, e.g. D:\experiments\my_sample\)
  MERci/             ← clone of this repo
  positions/         ← boundary_positions.txt, hole*.txt (from operator),
                        positions_{SAMPLE_NAME}.txt (from prepare_imaging/02)
  metadata/          ← frame_table_*.csv, shutter_sequence_*.png (prepare_imaging/01),
                        round_info.csv, round_bit_color_map.csv (prepare_imaging/03),
                        data_organization_*.csv (prepare_imaging/04)
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
    configs.py      # get_frame_table, get_transit_frame_table (N blank frames), get_color_sequence_name
                    # (underscore-joined tokens, e.g. blkf5_488f2_560f25_650f25_750f25),
                    # get_color_to_channel_dict, create_shutter_file (per-colour power via power={nm:power}),
                    # create_hal_config, format_z_offsets_from_frame_table,
                    # resolve_power / power_dict_to_channel_list (colour->power, channel-ordered HAL default_power),
                    # naming rule: sequence_stem + hal_config_filename / shutter_filename / frame_table_filename
                    # (stem = "{kind}-{name}", kind in bits/cells/transit; hyphens delimit the prefix)
                    # + read_hal_flip_vertical, find_frame_table_for_hal_config, get_color_frame_indices
                    # + reconstruct_frame_table (inverse: hal+shutter XML -> frame table) and its parsers
                    #   read_shutter_reference, parse_z_offsets, parse_shutter_events
    positions.py    # create_grid_positions, generate_scanning_path, filter_scanning_path, close_scanning_path,
                    # load_hole_polygons, get_path_stats
                    # + multi-tissue: discover_boundary_files (auto multi/single/legacy), load_boundary_polygon,
                    #   create_transit_path (A->B, ~2x step), build_boundary_path (per-boundary pipeline), BoundarySpec,
                    #   has_boundary_files / resolve_boundary_dir (fall back to data/positions/examples when empty)
    alignment.py    # cross-microscope FOV transfer: load_boundary_polygon, fit_isotropic_alignment
                    # (centroid/area init + IoU refinement, optional x/y axis flips), polygon_iou,
                    # AlignmentResult (scale + translation + flip_x/flip_y);
                    # bead-drift refinement: bead_frame_indices, select_bead_frame, extract_bead_frames
                    # (write bead-only .tiff), apply_orientation (inter-scope flip/transpose/rot to
                    # reconcile camera handedness HAL flip flags miss), phase_drift (skimage
                    # phase_cross_correlation, à la fishtank), compute_fov_drifts (mov_orient param);
                    # modality-robust path: detect_beads, register_point_translation (consensus
                    # voting + inlier-fraction score), compute_fov_drifts_beads
    dave.py         # create_round_info, create_round_info_multitissue (per (round,segment) rows +
                    #   positions_file/tissue/segment cols; both take data_drives= to round-robin hyb
                    #   rounds across physical drives), create_data_drive_skeleton (pre-creates the
                    #   H01..H0N folder skeleton on every configured drive), create_dave_config
                    #   (positions_dir= enables per-segment loops), annotate_dave_with_round_info,
                    #   series_to_movie_name, get_hal_frame_count
    kilroy.py       # load_kilroy_protocols, find_kilroy_config (MF2 fallback),
                    # KilroyProtocolResolver — resolve dave fluidic steps to real Kilroy protocol names.
                    # + protocol/command consistency: load_kilroy_commands, iter_protocol_references,
                    #   check_kilroy_consistency (flag protocol steps naming an undefined valve/pump command,
                    #   with fuzzy-matched suggestions), format_consistency_report, fix_kilroy_consistency
                    #   (apply confirmed name fixes in place, backup *.bak, preserve CRLF + ISO-8859-1)
    data_organization.py  # create_data_organization
    display.py      # print_frame_table, display_xml (Jupyter helpers)
  analysis/
    fov.py          # create_thumbnail(s), measure_stats, get_histogram, load_stats, load_histogram,
                    # analyze_file (top-level per-FOV worker: read once + all analyses + sentinel) — FOV-level analysis
    round.py        # create_mosaic, load_thumbnails_for_round — round-level mosaic
    spot_localization.py  # bead detection / 3D Gaussian fitting + PSF simulation (detect_beads_2d,
                          # localize_beads_in_file, match_beads_across_colors, simulate_multicolor_stack, …)
  state.py          # ExperimentStateMonitor — detects imaging vs. fluidics phases by watching file mtimes
  progress.py       # ProgressTracker — sentinel files for fov_done, round_done, round_transferred
  scheduler.py      # FOVScheduler (continuous, parallel process-pool), RoundScheduler, ExperimentScheduler — main analysis loops
  transfer.py       # transfer_round (per-round → NAS), mirror_tree (incremental data_dir → 2nd drive) — background robocopy/shutil
  visualization.py  # visualize_shutter_sequence, plot_fov_layout, plot_stats_over_rounds, display_mosaic
notebooks/
  prepare_imaging/  # Pre-experiment notebooks (run in order), split into per-experiment variants:
    reference/       # canonical, fully-featured templates (keep up to date)
    tumor/           # single tissue section per coverslip; split by acquisition type:
      epi/           #   epifluorescence acquisition — full copy of the 4 notebooks
      disk/          #   spinning-disk confocal acquisition — full copy of the 4 notebooks
                     #   (both four levels deep -> MERCI_DIR = ...parent.parent.parent.parent)
    lineage_tracing/ # multiple tissue sections per coverslip; split by acquisition type:
      merfish/       #   MERFISH (codebook) acquisition — full copy of the 4 notebooks
      lineage/       #   lineage-barcode acquisition — full copy of the 4 notebooks
                     #   (both four levels deep -> MERCI_DIR = ...parent.parent.parent.parent)
      # each of tumor/{epi,disk}/ and lineage_tracing/{merfish,lineage}/ contains:
      #   01_create_hal_config_and_shutters.ipynb     # imaging sequence, per-channel POWER, HAL/shutter for
      #                                                #   bits+cells, and a transit HAL config (blank frames)
      #   02_create_positions_from_tissue_boundary.ipynb # multi-boundary FOV grids + transit segments; per-segment /
      #                                                #   per-tissue positions files; creates data/ subfolders
      #   03_create_dave_config.ipynb                 # round-bit-color map (+ derives N_HYBS), round_info.csv + Dave recipe
      #   04_create_data_organization.ipynb           # MERlin data-organization setup (transit-safe series pick)
  analysis/         # Online-analysis notebooks (run during the experiment)
    01_fov_scheduler.ipynb                         # FOV-level scheduler (thumbnails, stats, histograms)
    02_round_scheduler.ipynb                       # round-level scheduler (mosaics, optional data transfer)
    03_view_mosaics.ipynb                          # display per-color mosaics as they are built
    04_view_intensity_stats.ipynb                  # plot per-frame intensity statistics over rounds
  misc/             # Ad-hoc utilities
    MF2_60XSil1.3_zcorrection.ipynb                # z-correction helper for the MF2 60x silicone objective
    reconstruct_frame_table_from_configs.ipynb     # inverse of prepare_imaging/01: hal+shutter XML -> frame_table CSV
    align_fovs_across_microscopes.ipynb            # map FOV positions from one scope to another: Part 1 overlaps
                                                   #   tissue boundaries (isotropic scale+translation+optional
                                                   #   axis flips, no rotation); Part 2 refines per-FOV bead drift
                                                   #   via phase_cross_correlation. Inputs are explicit per-dir paths.
    extract_source_bead_frames.ipynb               # run at the source scope: write a compact per-FOV bead-only
                                                   #   .tiff (+ compact frame table) so only the bead frames move
                                                   #   to the NAS for align_fovs Part 2
    verify_kilroy_protocol_consistency.ipynb       # verify a Kilroy config's protocols only reference defined
                                                   #   valve/pump commands; fuzzy-suggest + (after confirm) rewrite
                                                   #   mismatches in place (backup to *.bak), via kilroy.py helpers
data/
  configs/
    hal/            # hal-config-{mic}-epi.xml — HAL config templates (one per microscope)
    kilroy/         # kilroy-config-*-{mic}-*-{YYMMDD}.xml — Kilroy configs (one or more per microscope)
  positions/        # boundary_positions.txt, hole*.txt — example tissue boundary files
    examples/       # ready-made boundary sets for each layout, used as the notebook-02
                    #   fallback when SAMPLE_DIR/positions is empty:
                    #   legacy/ (one boundary), single/ (1 tissue, 2 boundaries),
                    #   multi/ (2 tissues x 2 boundaries)
  readouts.csv      # default codebook readout table (bit number -> readout name), read by prepare_imaging/04
```

## Architecture

### Pre-experiment workflow

Run the four `prepare_imaging/<variant>/` notebooks (variant = `reference`) in
order before starting the microscope. For `tumor` and `lineage_tracing` the four
notebooks live one level deeper under an acquisition-type subfolder
(`prepare_imaging/tumor/{epi,disk}/`, `prepare_imaging/lineage_tracing/{merfish,lineage}/`);
run the set for the acquisition being prepared.

**01** (`prepare_imaging/<variant>/01_create_hal_config_and_shutters.ipynb`): defines the imaging sequence as a *frame table* (one row per camera frame, columns `color`, `channel`, `z`) using `get_frame_table`. Supports `scan_mode="interleaved"` (all colors per z-plane, AOTF) or `scan_mode="sequential"` (full z-sweep per color, boustrophedon, physical shutters). The objective's return to `bead_z` after the stack is controlled by `z_return_mode`: `"progressive"` (default) steps down with blank frames in increments of `return_step` (5 µm default); `"instant"` jumps straight back (the previous behaviour). A per-channel `POWER = {nm: power}` dict sets each shutter `<event>`'s `<power>` (actual acquisition power, by frame colour) and the HAL `<default_power>` (channel-ordered via `power_dict_to_channel_list`). Auto-generates a compact colour name via `get_color_sequence_name` (underscore-joined tokens, e.g. `blkf5_488f2_560f25_650f25_750f25`). Sets `<filetype>` (`.zarr` default, or `.dax`/`.tiff`) and `<exposure_time>` in the HAL config. Also writes a **transit** HAL config/shutter (`get_transit_frame_table`, `N_TRANSIT_BLANK` blank frames at bead z) for the between-boundary transit FOVs.

**Naming rule.** Each round's three artefacts share a stem `{kind}-{name}` (kind = `bits`/`cells`/`transit`), built by the `configs` helpers `hal_config_filename` / `shutter_filename` / `frame_table_filename`. Hyphens delimit the structural prefix; underscores live only inside `{name}`. So for the bits round with `{name}=blkf5_488f2_560f25_650f25_750f25`:
- `SAMPLE_DIR/settings/hal-config-{mic}-bits-{name}.xml` — patched from `data/configs/hal/hal-config-{mic}-epi.xml`
- `SAMPLE_DIR/settings/shutter-bits-{name}.xml` — shutter event XML
- `SAMPLE_DIR/metadata/frame-table-bits-{name}.csv` — frame table
- `SAMPLE_DIR/metadata/shutter_sequence_{name}.png` — visualisation

(cells and transit rounds follow the same pattern with their own `kind`.) The analysis-side `find_frame_table_for_hal_config` mirrors this — it reads `<shutters>`, rewrites the `shutter-` prefix to `frame-table-`, and finds the CSV (legacy `frame_table_{name}.csv` still accepted as a fallback).

Both XML files use Windows CRLF line endings and ISO-8859-1 encoding as required by HAL.

**02** (`prepare_imaging/<variant>/02_create_positions_from_tissue_boundary.ipynb`): builds the FOV scanning positions for one or more tissue sections. If `SAMPLE_DIR/positions/` has no boundary files yet, it falls back to a bundled example dataset under `MERci/data/positions/examples/{legacy,single,multi}` (chosen by the notebook's `EXAMPLE_LAYOUT`; per-variant default: tumor→`legacy`, lineage_tracing/reference→`multi`) via `resolve_boundary_dir`, and **copies that example's boundary + hole inputs into `positions/`** so the experiment folder is self-contained and notebooks 03/04 (which read `positions/` directly) find them. This lets the whole pipeline be run and tested before any real boundaries are drawn; the copy is idempotent (skipped once `positions/` has inputs). It **auto-detects the layout** from the boundary filenames in the resolved directory (`discover_boundary_files`): `tissue_{t}_boundary_positions_{b}.txt` → **multi** (several sections), `boundary_positions_{b}.txt` → **single** (one section, several boundaries), or a lone `boundary_positions.txt` → **legacy** (one boundary). For each boundary it builds a boustrophedon FOV path (`build_boundary_path` = `create_grid_positions` → `generate_scanning_path` → `filter_scanning_path`); between consecutive boundaries (wrapping the last back to the first) it inserts a **transit** segment (`create_transit_path`: FOVs on the A→B line spaced ~`TRANSIT_SPACING`×step). `hole*.txt` polygons are global (applied to every boundary). Writes per-segment files referenced by Dave (`positions_{SAMPLE_NAME}_{T#B#|B#}.txt`, `positions_{SAMPLE_NAME}_transit_{k}.txt`), per-tissue FOV-only files (`positions_{SAMPLE_NAME}_T{t}.txt`, or `positions_{SAMPLE_NAME}.txt` for single/legacy), and creates the `data/` subfolders for the layout (`mosaic10x`, and `tissue_{t}/{cells,hybs,transit}` or top-level `{cells,hybs,transit}`).

FOV grid rules: odd row and column count; centre FOV at bounding-box midpoint. A FOV is kept if its camera square overlaps the boundary polygon at all; excluded only if a hole polygon fully contains the FOV square.

**03** (`prepare_imaging/<variant>/03_create_dave_config.ipynb`): generates `round_info.csv` and the Dave experiment recipe XML. With a single boundary it uses the classic single-positions recipe (`create_round_info` + one `<loop>` per imaging round). With **multiple boundaries** it builds a **segment-aware** `round_info` (`create_round_info_multitissue`: one row per (round, segment) — boundary movies with the cells/bits config + transit movies with the transit config, plus `positions_file`, `tissue`, `segment` columns) and a **per-segment** Dave recipe (`create_dave_config(positions_dir=…)`: each boundary/transit segment is its own `<loop>` — "Imaging Round NN - <segment>" — with its own movie, HAL config and positions file, in order; fluidics loops stay between rounds). HAL configs for bits vs. cells rounds are auto-detected by glob patterns (`blkf3*` for bits, `blkf1*` for cells); the transit HAL config from notebook 01 is auto-detected too. The Kilroy config for the microscope is resolved (via `find_kilroy_config`, falling back to MF2 when the microscope has no config) and passed to `create_dave_config` as the source of fluidic protocol names: every protocol written into the Dave recipe is resolved to — and required to exist as — a `<protocol>` in that Kilroy config, raising `ValueError` otherwise. This notebook also **defines the round–bit–colour mapping** (`round_bit_color`, one `(round, bit, color_nm)` per bit) and **derives `N_HYBS` from it** (`N_HYBS = max(round)`) rather than hard-coding it, so the hyb count always matches the codebook; it saves the mapping to `SAMPLE_DIR/metadata/round_bit_color_map.csv` for notebook 04 to reuse. Writes `SAMPLE_DIR/metadata/round_info.csv`, `SAMPLE_DIR/metadata/round_bit_color_map.csv`, and `SAMPLE_DIR/settings/dave-{mic}-{N}bits-{SAMPLE_NAME}.xml`.

**Multi-drive round-robin (optional).** Setting `DATA_DRIVES = ["D:", "E:", "F:"]` (default `[]` = disabled) makes `create_round_info`/`create_round_info_multitissue` spread successive **hyb** rounds round-robin across those drives — round *i*'s `data_dir` becomes `<drive>/data/hybs/H{NN}` instead of always `SAMPLE_DIR/data/hybs/H{NN}` (cells/transit are unaffected, always under `SAMPLE_DIR/data/...`). `create_data_drive_skeleton` pre-creates the full `H01..H0N` folder skeleton on every configured drive so each disk has an identical structure before acquisition starts. This pairs with `analysis_mode="round_robin_drives"` (see Online-analysis architecture below) so analysis can read already-completed rounds on idle drives while HAL is still writing the current round to a different one.

**04** (`prepare_imaging/<variant>/04_create_data_organization.ipynb`): generates the MERlin data-organization CSV and annotates the Dave XML with per-round bit information. Picks the bits/cells series by `imaging_type` (so a multi-boundary `round_info`'s transit movies are never selected). Note: multi-tissue MERlin analysis is per tissue / per boundary — confirm the intended workflow before relying on the generated data-organization. Requires `MERci/data/readouts.csv` (codebook mapping bit numbers to readout names; shipped in the repo). Frame tables and series patterns are auto-detected from `metadata/`. The `round_bit_color` mapping is **defined in notebook 03**; notebook 04 reads it back from `SAMPLE_DIR/metadata/round_bit_color_map.csv` (raising `FileNotFoundError` if notebook 03 has not run). Writes:
- `SAMPLE_DIR/metadata/data_organization_{MICROSCOPE}_{SAMPLE_NAME}.csv`
- Annotates `SAMPLE_DIR/settings/dave-*.xml` with per-round bit comments

### Online-analysis architecture

`ExperimentConfig` holds all paths and tunable parameters. Notable fields:
- `image_suffix` — `.zarr` (default), `.dax`, or `.tiff`
- `fluidics_type` — `"adaptor"` (t_max = 100 min) or `"direct"` (t_max = 50 min); sets `t_max` automatically when left as `None`
- `settings_dir` — `SAMPLE_DIR/settings/`; needed for auto flip_y and per-color mosaic lookup
- `mosaic_flip_y` — `None` (auto-read from HAL config `<flip_vertical>`), `True`, or `False`
- `fov_subset` — list of FOV ids to restrict analysis; `None` = all FOVs
- `transfer_dest` — network path (e.g. a NAS) to copy completed round data to; `None` = disabled. In `same_drive`/`mirror_drive` mode this only runs during the fluidics window (see `transfer_min_time`); in `round_robin_drives` mode it runs **continuously**, skipping only the round on the drive HAL is actively writing (`RoundScheduler._process_pending_transfers`) — since each round already lives on its own physical drive, a completed round elsewhere is always safe to copy out.
- `transfer_min_time` — minimum seconds remaining in the fluidics window before starting a transfer (`same_drive`/`mirror_drive` modes only)
- `analysis_mode` — `"same_drive"` (default, mode B: analyse from `data_dir`), `"mirror_drive"` (mode A: mirror `data_dir` → `analysis_source_dir` during fluidics and analyse from that second-drive copy), or `"round_robin_drives"` (for experiments whose `round_info.csv` spreads hyb rounds across several physical drives via `create_round_info(data_drives=…)` — see prepare_imaging/03). Analysis runs **continuously** in all three modes (not only during fluidics). `config.analysis_data_dir` resolves to the directory the FOV scheduler reads from (not meaningful in `round_robin_drives` mode — see `all_data_roots` below).
- `analysis_source_dir` — second-drive mirror directory; **required** when `analysis_mode="mirror_drive"`
- `all_data_roots` — every directory referenced in `round_info_csv`'s `data_dir` column plus `data_dir` itself, read directly from the CSV; used by `FOVScheduler`/`ExperimentStateMonitor` in `round_robin_drives` mode to scan every physical drive instead of one root. No new config field is needed for the drive list itself — `round_info.csv` is the single source of truth.
- `n_analysis_workers` — FOV process-pool size; `None` = `cpu_count − 2` (`config.resolved_n_workers`). Each worker holds one image stack (~200 MB) in RAM.

`ExperimentMetadata` (loaded via `ExperimentMetadata.load(round_info_csv, positions_txt, data_dir)`) cross-references round IDs, FOV IDs, series patterns, and expected file paths. When a `dir`/`data_dir` column is present in `round_info.csv`, per-round file paths are resolved from that directory instead of the top-level `data_dir`. Each series carries an ordered list of **candidate directories** (`SeriesInfo.candidate_dirs`); `resolve_path(fov, suffix)` returns the first candidate that exists on disk, falling back to the primary one before acquisition. The **cells round** is treated as a bona fide imaging round (typically `imaging_round=1`) and its files are accepted in **either** `data/cells/` or the top-level `data/`, regardless of which the `data_dir` column records — so `all_fovs_done_for_round`, mosaics, and transfers all find the cells data wherever HAL actually wrote it. Two more methods support `round_robin_drives` mode: `drive_of_round(round_id)` returns the drive letter of a round's `data_dir`, and `actively_writing_round()` returns the round HAL is most likely writing right now — the highest round id that has started (≥1 expected file exists) but isn't yet complete (not every expected file exists); it returns `None` once that round is fully on disk, so a just-finished round's drive isn't wrongly excluded for its whole following fluidics window.

`ExperimentStateMonitor` determines the microscope phase by watching the newest file mtime in `data_dir` (or, in `round_robin_drives` mode, across every drive in `config.all_data_roots`):
- **IMAGING**: a new image file was written within `imaging_idle_threshold` seconds
- **FLUIDICS**: `t_min ≤ time_since_imaging ≤ t_max` → `should_analyze = True`

`should_analyze` is no longer the analysis gate — FOV/round analysis runs continuously. The phase is still used to time the mode-A mirror and the NAS transfer (both read the acquisition drive, so both run only while the microscope is idle).

`ProgressTracker` tracks completeness via zero-byte sentinel files under `analysis_dir/done/`:
- `<stem>.fov_done` — FOV analysis complete
- `round_<r>.round_done` — mosaic(s) built for round r
- `round_<r>.round_transferred` — raw data for round r copied to `transfer_dest`

Multiple notebooks can run concurrently — no shared state.

`FOVScheduler.run_loop()` runs **continuously** (acquisition + fluidics): each tick it (in mirror mode) refreshes the second-drive mirror while idle, discovers stable image files (zarr/dax/tiff) under `config.analysis_data_dir` (or every drive in `config.all_data_roots` in `round_robin_drives` mode — each root's scan is wrapped so an unreachable/disconnected disk logs a warning instead of crashing the tick), and analyses pending files **in parallel across a process pool** (`config.resolved_n_workers`). In `round_robin_drives` mode, files on the drive of `meta.actively_writing_round()` are excluded from `pending` each tick — HAL's current write target is skipped, every other drive's completed rounds analyse immediately. Each worker runs the top-level `analysis.fov.analyze_file`, which reads the stack once and writes thumbnails (PNG) + per-frame stats (CSV) + histograms (`.npz`) + the FOV sentinel. With `n_analysis_workers=1` it runs serially in-process. Respects `fov_subset`; call `.close()` (done automatically when `run_loop` exits) to shut the pool down. `RoundScheduler.run_loop()` also runs continuously, assembling **one mosaic per imaging color** (`round_{r:03d}_{color}nm_mosaic.png`) once all FOV sentinels exist; auto-resolves `flip_y` from the HAL config; optional background transfers via `transfer.transfer_round` happen only during fluidics in `same_drive`/`mirror_drive` mode, or continuously (same active-round/drive exclusion as `FOVScheduler`) in `round_robin_drives` mode. `ExperimentScheduler.wait_and_run()` calls a user callback after all rounds complete.

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

`MF2`, `MF3`, `MF4`, and `MF5` share the same 5-channel mapping: `{405→4, 488→3, 560→2, 650→1, 750→0}`. `MFX` and `ST2` have only 4 channels with a distinct ordering: `{650→0, 560→1, 488→2, 405→3}` (no 750). `NaN` = blank frame (no laser). Extend `_COLOUR_TO_CHANNEL` in `acquisition/configs.py` for other microscopes.

Camera geometry also follows from the microscope: `MFX` and `ST2` have 2304×2304 sensors at 0.0878 µm/pixel; the MF-series (`MF2`–`MF5`) have 2048×2048 at 0.108 µm/pixel. `acquisition/configs.py` exposes `get_camera_frame_size(microscope)` (sensor pixels; mapping `_CAMERA_PIXELS`), `get_camera_pixel_size_um(microscope)` (mapping `_CAMERA_PIXEL_SIZE_UM`), and `get_fov_geometry(microscope) -> (pixel_size_um, image_size_px)` which bundles both. Frame size drives the storage figure in the Dave experiment estimate (`estimate_dave_experiment` / the summary printed by `create_dave_config`); `get_fov_geometry` gives `prepare_imaging/02` its scanning-grid geometry from the microscope alone (set `MICROSCOPE` there instead of hard-coding `pixel_size_um`/`image_size_px`).

## Running notebooks

Notebooks auto-detect `SAMPLE_DIR` from their own location. `analysis/` and `misc/` notebooks are two levels under the repo root, so `MERCI_DIR = Path(os.getcwd()).parent.parent` (the `MERci/` clone), then `SAMPLE_DIR = MERCI_DIR.parent`. The `prepare_imaging/<variant>/` notebooks (`reference`) are **three** levels deep, so they use `MERCI_DIR = Path(os.getcwd()).parent.parent.parent`; the `tumor/{epi,disk}/` and `lineage_tracing/{merfish,lineage}/` notebooks are **four** levels deep, so they use `MERCI_DIR = Path(os.getcwd()).parent.parent.parent.parent`. Do not hardcode absolute paths in notebooks.

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
elapsed: <e.g. 12m 30s — wall-clock from prompt submission to completion>
status: completed | in-progress | abandoned
---

## Prompt
<verbatim copy of the user's request>

## Plan
<Claude's plan of action before executing>

## Summary
<what was actually done, including any deviations from the plan>
```

The `UserPromptSubmit` date/time hook injects `Current local date/time: … (epoch N)`
on every prompt. Compute **`elapsed`** (written just before `status`) as the finish
time minus that submit epoch: run `date +%s` (bash) or
`[DateTimeOffset]::Now.ToUnixTimeSeconds()` (PowerShell) when done and subtract the
epoch from the message that began the request (the first turn's epoch for a
multi-turn request). Omit `elapsed` if no submit epoch is available — never guess it.

Format rationale: Markdown + YAML frontmatter is Claude-native, human-readable,
and lets all entries be scanned/grepped by metadata without reading every body.

