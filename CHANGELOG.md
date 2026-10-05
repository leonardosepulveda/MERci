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

### Fixed

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

- `CHANGELOG.md` and versioning via git tags (`vX.Y.Z`).

### Changed

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
