# prepare_imaging / tumor / disk

Pre-experiment notebooks for the **spinning-disk confocal** acquisition of a
tumor experiment: a **single tissue section** per coverslip. One of the two
acquisition types under `tumor/`; the other is `../epi/` (epifluorescence).

Duplicated from the shared `tumor` notebook set — tune the parameters here for
the disk acquisition. Keep the notebook *logic* in sync with `../../reference/`;
change only the experiment parameters here.

These notebooks live **four** levels under the repo root
(`MERci/notebooks/prepare_imaging/tumor/disk/`), so they resolve
`MERCI_DIR = Path(os.getcwd()).parent.parent.parent.parent`.

## Expected layout

- Boundary files in `positions/`: `boundary_positions_{b}.txt` for a section
  split across several boundaries, or a single `boundary_positions.txt`.
  (Do **not** use the `tissue_{t}_…` naming — that selects the multi-section
  path used by `lineage_tracing/`.)
- Notebook 02 therefore runs in **single**/**legacy** mode and creates top-level
  `data/{cells,hybs}` (+ `data/transit` only if there are multiple boundaries)
  and `data/mosaic10x`.
- If `positions/` is empty, notebook 02 falls back to the bundled example set
  selected by `EXAMPLE_LAYOUT` (default `"legacy"` here; `"single"` also fits a
  single section) under `MERci/data/positions/examples/`, and copies that
  example's boundary + hole inputs into `positions/` so notebooks 03/04 find
  them. Lets you test the pipeline immediately.

## Parameters to set for a tumor disk run

Set these in the notebooks to match the assay (values left at the `reference`
defaults until confirmed):

- `01`: `MICROSCOPE`, `POWER`, `color_seq`, z-range, `EXPOSURE_TIME`,
  `N_TRANSIT_BLANK`.
- `03`: `N_HYBS`, `USE_ADAPTORS`, `FIRST_HYB_NO_CLEAVE`, `INCLUDE_FINAL_CLEAVE`.
- `04`: `round_bit_color` mapping to match the codebook.
