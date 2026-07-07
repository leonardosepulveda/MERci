# prepare_imaging / tumor

Pre-experiment notebooks tuned for a **tumor** experiment: a **single tissue
section** per coverslip.

Copied from `../reference/`. Keep the notebook logic in sync with `reference/`;
change only the experiment parameters here.

## Expected layout

- Boundary files in `positions/`: `boundary_positions_{b}.txt` for a section
  split across several boundaries, or a single `boundary_positions.txt`.
  (Do **not** use the `tissue_{t}_…` naming — that selects the multi-section
  path used by `lineage_tracing/`.)
- Notebook 02 therefore runs in **single**/**legacy** mode and creates top-level
  `data/{cells,hybs}` (+ `data/transit` only if there are multiple boundaries)
  and `data/mosaic10x`.

## Parameters to set for a tumor run

Set these in the notebooks to match the assay (values left at the `reference`
defaults until confirmed):

- `01`: `MICROSCOPE`, `POWER`, `color_seq`, z-range, `EXPOSURE_TIME`,
  `N_TRANSIT_BLANK`.
- `03`: `N_HYBS`, `USE_ADAPTORS`, `FIRST_HYB_NO_CLEAVE`, `INCLUDE_FINAL_CLEAVE`.
- `04`: `round_bit_color` mapping to match the codebook.
