# prepare_imaging / lineage_tracing / merfish_multi_z

Pre-experiment notebooks for a **variable-z-per-FOV** MERFISH acquisition of a
lineage-tracing experiment. A variant of `../merfish/` for tissue whose depth
varies across the coverslip: instead of imaging every FOV to one fixed z-depth
(wasting time/disk on thin regions), this workflow images a full-depth DAPI
(cells) round first, measures each FOV's real tissue thickness from it, buckets
FOVs into a handful of z-depth tiers, and images each bits round to only the
depth its tier actually needs.

Same **multi-tissue** layout as `lineage_tracing` (multiple tissue sections per
coverslip, each possibly split across several boundaries, with transit FOVs
between them). These notebooks live **four** levels under the repo root
(`MERci/notebooks/prepare_imaging/lineage_tracing/merfish_multi_z/`), so they
resolve `MERCI_DIR = Path(os.getcwd()).parent.parent.parent.parent`.

## Why a separate notebook set instead of a flag on `../merfish/`

The variable-z workflow needs an extra imaging round (the full-depth DAPI
calibration pass) and an extra decision point (the per-FOV tissue-thickness
measurement) *between* the normal "generate positions" and "generate
round_info" steps — it isn't a parameter tweak on the existing 7-notebook
sequence, it changes the sequence itself. Notebook *logic* should still be kept
in sync with `../merfish/` wherever the two share a step (round_info, dave
config, data organization, experiment info, merlin scripts) — change only the
experiment parameters here.

## Notebook sequence (10 notebooks)

| # | Notebook | vs. `../merfish/` |
|---|----------|--------------------|
| 01 | `01_create_hal_config_and_shutters.ipynb` | Trimmed: **cells + transit only**. The bits HAL config is deliberately deferred to notebook 05 — there is no single bits depth to fix yet. |
| 02 | `02_create_boundary_from_mosaic.ipynb` | Unchanged — derive tissue boundaries from a Steve mosaic (optional). |
| 03 | `03_create_positions_from_boundaries.ipynb` | Unchanged (renumbered from `merfish`'s `02_create_positions_from_boundaries.ipynb`) — builds the FOV grid + transit segments. |
| 04 | `04_measure_tissue_thickness.ipynb` | **New** (adapted from `notebooks/misc/measure_tissue_thickness.ipynb`). Run after the **cells** round finishes: measures each FOV's real z-extent of tissue signal and exports a per-FOV z table. |
| 05 | `05_create_hal_config_and_shutters_multi_z.ipynb` | **New**. Reads notebook 04's z table, buckets FOVs into `N_TIERS` z-depth tiers (quantile binning), and generates one bits HAL config + shutter file per tier into `SAMPLE_DIR/multi_z/` (see below). Tags each FOV in the positions file(s) with its assigned tier via a 3rd column. |
| 06 | `06_create_round_info.ipynb` | Renumbered from `03_create_round_info.ipynb`. Uses the **deepest** tier's HAL config as the representative bits config, and tags bits rows `tissue_thickness="multi"` + `z_lengths` (every tier's frame count, JSON-encoded). |
| 07 | `07_create_dave_config.ipynb` | Renumbered from `04_create_dave_config.ipynb`. No functional changes — `create_dave_config` already knows to skip the static per-movie `<length>`/`<parameters>` for a `tissue_thickness="multi"` round, since the positions file's own 3rd column supplies the real per-FOV values (see the Dave patch below). |
| 08 | `08_create_data_organization.ipynb` | Renumbered from `05_create_data_organization.ipynb`. Picks the bits frame table with the **most frames** among all `frame-table-bits-*.csv` matches (the deepest tier), so MERlin's declared z-range covers every FOV. |
| 09 | `09_create_experiment_info.ipynb` | Renumbered from `06_create_experiment_info.ipynb`. Bits HAL config lookup (for exposure time) checks both `settings/` and `multi_z/`. |
| 10 | `10_create_merlin_scripts.ipynb` | Renumbered from `07_create_merlin_scripts.ipynb`. Passes `--allow-ragged-z-stacks` to the generated slurm submit script whenever `round_info.csv` has any `tissue_thickness="multi"` row. |

## The `multi_z/` folder

Notebook 05 writes each tier's HAL config **and** shutter XML into one combined
`SAMPLE_DIR/multi_z/` folder (not split into separate `hal_configs/`/`shutters/`
subfolders) — HAL resolves a `<shutters>` reference relative to its own
`xml_directory`, and keeping both file kinds in the same folder avoids relying
on an unverified cross-folder resolution path. `settings/` still holds only the
cells + transit HAL configs/shutters (notebook 01) and the Dave recipe
(notebook 07).

## Positions-file 3rd column + the Dave patch

Notebook 05 rewrites each positions file in place, adding a 3rd column: the
bare HAL-config stem (e.g. `hal-config-st2-bits-deep-560f25_650f25`) for the
tier that FOV was assigned to. A stock `storm_control` Dave cannot read this —
it requires the patched `v2Generator.py` in
`../../../../../misc/dave_multi_z/` (**outside** the MERci repo; see that
folder's own `README.md` for what it changes and how to deploy it to the
microscope computer). Both halves are required together: this patch alone
(without the `dave.py` changes in this repo) or this repo's changes alone
(without the patch) will not produce a working per-FOV z recipe.

## Expected layout

Same boundary-file conventions as `../merfish/`:

- Boundary files in `positions/`: `tissue_{t}_boundary_positions_{b}.txt`.
- Notebook 03 runs in **multi** mode: boundaries are visited in order (tissue,
  then boundary) with a transit segment between consecutive boundaries. It
  creates `data/tissue_{t}/{cells,hybs,transit}` and `data/mosaic10x`, and
  writes per-segment positions files plus per-tissue FOV-only files.
- If `positions/` is empty, notebook 03 falls back to the bundled example set
  selected by `EXAMPLE_LAYOUT` (default `"multi"` here) under
  `MERci/data/positions/examples/`.

## Parameters to set for a variable-z lineage-tracing MERFISH run

Set these in the notebooks to match the assay (values left at the `merfish`
defaults until confirmed):

- `01`: `MICROSCOPE`, `POWER`, `color_seq` for cells, z-range, `EXPOSURE_TIME`,
  `N_TRANSIT_BLANK`.
- `03`: `TRANSIT_SPACING`, `SCAN_DIRECTION`.
- `04`: `THRESHOLD` (tissue/background separator), `TPC_THRESHOLD`,
  `Z_MARGIN_UM` — review the histograms/heatmaps before trusting the
  exported z table.
- `05`: `N_TIERS`, `MICROSCOPE`, `POWER`, `color_seq` for bits, `EXPOSURE_TIME`
  (must match notebook 01's cells exposure), `z_bead`/`z_step`/etc. for the
  bits sweep.
- `06`: `round_bit_color` mapping to match the codebook (derives `N_HYBS`).
- `07`: `USE_ADAPTORS`, `FIRST_HYB_NO_CLEAVE`, `INCLUDE_FINAL_CLEAVE`.
- `08`: Note: multi-tissue MERlin analysis is per tissue / per boundary —
  confirm the intended workflow before relying on the generated
  data-organization.
- `10`: MERlin's ragged-z-stack decoding is still incomplete upstream as of
  this writing — confirm with whoever maintains your MERlin checkout before
  relying on `--allow-ragged-z-stacks` for a real decode.
