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

- `CHANGELOG.md` and versioning via git tags (`vX.Y.Z`).

## [0.1.0] - 2026-10-02

First tagged version: a baseline snapshot of `master`, not a curated
release. Earlier history is in `git log`.
