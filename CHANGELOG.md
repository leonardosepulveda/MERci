# Changelog

All notable changes to MERci. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/).

What counts as MERci's public API (a change to it is breaking): signatures of
`src/MERci/` functions/classes that notebooks call, the `pipeline.yaml`
schema, files MERci reads or writes (`round_info.csv`, `positions_*.txt`,
`experiment_info.yaml`, ...), and notebook names/locations plus the
`SAMPLE_DIR` layout they assume. While on `0.x`, a breaking change bumps
MINOR; from `1.0.0` on, it bumps MAJOR.

Each feature/fix branch adds a line under `[Unreleased]`. A release moves
those lines under a new version heading, bumps `version` in
`pyproject.toml`, and tags the merge commit `vX.Y.Z`.

## [Unreleased]

### Added

- fishtank scripts: detect-spots takes `on_the_fly` and `max_drift`;
  detect-spots, cellpose and mosaic take `microscope` (a settings YAML that
  overrides the image XML; `--scale_factor` is then left out). ST2 60X YAML
  shipped in `data/configs/fishtank/microscope/`. `create_fishtank_files`
  sets them (on-the-fly, max_drift 300, 64gb first pass). Default output
  unchanged.

### Fixed

- fishtank cellpose scripts default to `--model cpsam` instead of `nuclei`,
  which cellpose 4 does not have (it fell back to `cpsam`, so results don't
  change; the script now says what runs).
- fishtank cellpose scripts pass `--anisotropy` (full-resolution z step /
  pixel size, from each acquisition's DAPI frame table; new
  `cellpose_anisotropy`). Without it 3D cellpose treated z like xy (LT066:
  5.6 at full resolution, 1.4 after fishtank's /4 downsample).
- Cluster-resource template: `CellPoseEvalSAM` asks for 2 CPUs (`cpus: 2`,
  needs MERlin fc012d8), so reading the next fov overlaps the GPU eval.
- Cluster-resource template: `Decode`, `SmfishSignal`, `SumSignal`,
  `CleanCellBoundaries`, `RefineCellDatabases`, `AdaptiveFilterBarcodes`,
  `GenerateMosaicTile` and every `OptimizeNN` rule no longer pin `mem`/`time`
  (commented out), so MERlin's data-based estimates (MERlin 207196a) apply.
- Cluster-resource template: `CellPoseSegmentSAMFigures` 32 GB / 6 h ->
  4 GB with the default 3 h, `CombineCleanedBoundaries` 32 GB -> 8 GB
  (measured max 2.2 GB / 100 min and 2.6 GB on LT066/BC553 runs).
- ST2 40X pixel size 0.1317 -> 0.1463 um/px (MERlin `STORM2_40X.yaml`,
  both ST2 HAL configs). 0.1317 assumed a 60/40 ratio; the Olympus 60X and
  Nikon 40X (f = 3.0 / 5.0 mm) give 5/3. Still a placeholder until measured.
- Cluster-resource template: `CellPoseSegmentSAMDone` renamed to
  `CellPoseSegmentSAMFigures` (32 GB, 6 h), matching MERlin's new
  per-task `<Task>Figures` rule that now draws the figures.
- Cluster-resource template: `ExportBarcodes` 100 GB -> 32 GB with a 4 h
  limit, `CombineCleanedBoundaries` 100 GB -> 32 GB (MERlin now streams
  ExportBarcodes per fov and fixed CombineCleanedBoundaries' quadratic loops).
- `build_merlin_analysis_parameters` raises if `generate_mosaic` uses FFC
  (`use_ffc`, on by default) but `create_ffc` is not in the task list. That
  config used to fail only at MERlin run time.
- `notebooks/monitor/merlin_progress.py` is executable, so it runs
  directly via its shebang (was `Permission denied`).

### Added

- `MERci.analysis.tissue_quality` (BC561's tissue-quality model):
  P(damaged) of 103 µm DAPI windows, a FOV's damaged %, a 51 µm removal map
  and per-nucleus flags; image-only or v2 + nuclei. Fit at first use from
  `data/tissue_quality/training_windows.parquet` and cached. New
  dependencies: scikit-learn, joblib.
- `before_imaging/regular/02d_filter_positions_by_tissue_quality`: drops
  FOVs with damaged tissue (focus-test DAPI) before the full acquisition;
  the original positions are kept as `positions_<tag>_all.txt`.
- create_merlin_files sets `create_ffc`'s `dark_frame_indices` to the
  blank frames of the movie MERlin's CreateFfc reads its dark frame from
  (`merlin_config.ffc_dark_frame_indices`). Left out when there are none.
- Two-task CellPose (MERlin 3323ebc): new `cellpose_eval_sam` atom
  (`CellPoseEvalSAM`, batched on `gpu_requeue`, full GPU cards only) whose
  cellpose parameters come from `cellpose_segment_sam` (`parameters_from`);
  with it, `CellPoseSegmentSAM` gets `eval_task` and runs on a CPU. On in
  `default_with_segmentation` and the tumor_epi/tumor_disk/
  lineage_tracing_merfish menus. Set `cellpose_eval_sam: false` for the
  one-task form; `create_cluster_resource_allocation(cellpose_eval=False)`
  then puts the GPU back on `CellPoseSegmentSAM`.
- Dave recipes wrap every loop in `<log><loop> start</log>` /
  `<log><loop> end</log>` (a patched Dave writes each as a small file in the
  current folder; an unpatched one ignores the tag). Each round's
  `<change_directory>` now comes before the fluidics that precedes it, so
  those logs land in that round's folder. The round-info comments of
  `annotate_dave_with_round_info` head that whole block.
- `CHANGELOG.md` and versioning via git tags (`vX.Y.Z`).

### Changed

- The focus-lock test recipe moved from notebook 04 to a new
  `02c_create_focus_test_dave_config`, so it can run before 02d filters
  the positions; its movie length comes from pipeline.yaml's
  `focus_test_frames` (tumor_disk: 2, a 405 + blank movie). 04 builds the
  cells and hybs recipes only.
- **Breaking:** `LiveRoundMosaicBuilder` caches tiles as float32 `.npy`
  (oriented and resized, before FFC and the contrast stretch) instead of
  stretched `.png`, so a round re-renders with another contrast range or FFC
  setting without raw reads (LT066: 9 s vs 88 s for 30 FOVs). Old `.png`
  tiles are ignored and rebuilt. The contrast-range cache name now includes
  the percentiles and FFC setting. New `set_contrast_range` for a hand-set
  range; `stretch_to_uint8` now resizes before stretching (split into
  `resize_tile` + `stretch_tile`).
- `export_barcodes` atom sets `format: parquet` explicitly (MERlin's new
  default since af9e800: `ExportBarcodes/barcodes.parquet` instead of
  `barcodes.csv`). Override with `format: csv` for the old file.
- CLAUDE.md: when updating an experiment's clone, a task menu is not a
  carried-over parameter (keep the current default, list differences).
- **Breaking:** `create_snakemake_parameters` writes MERlin's `-k` file as
  `parameters_{SHORT_NAME}.yaml` (was `.json`), with a comment above `nodes`
  recording the Slurm launch-delay measurement behind its value. Needs a
  MERlin with YAML `-k` support (MERlin commit 43986a9).
- `create_snakemake_parameters` default `nodes` lowered 1000 -> 150, to cap
  per-run concurrent jobs (Slurm step-launch delays timed out short jobs).

## [0.1.0] - 2026-10-02

First tagged version: a baseline snapshot of `master`, not a curated
release. Earlier history is in `git log`.
