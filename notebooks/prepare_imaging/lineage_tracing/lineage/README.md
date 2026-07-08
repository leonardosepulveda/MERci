# prepare_imaging / lineage_tracing / lineage

Pre-experiment notebooks for the **lineage** acquisition of a lineage-tracing
experiment (the lineage-barcode readout). One of the two acquisition types under
`lineage_tracing/`; the other is `../merfish/`.

Same **multi-tissue** layout as `lineage_tracing` (multiple tissue sections per
coverslip, each possibly split across several boundaries, with transit FOVs between
them). Duplicated from the shared `lineage_tracing` notebook set — tune the
parameters here for the lineage acquisition. Keep the notebook *logic* in sync with
`../../reference/`; change only the experiment parameters here.

These notebooks live **four** levels under the repo root
(`MERci/notebooks/prepare_imaging/lineage_tracing/lineage/`), so they resolve
`MERCI_DIR = Path(os.getcwd()).parent.parent.parent.parent`.

## Expected layout

- Boundary files in `positions/`: `tissue_{t}_boundary_positions_{b}.txt`
  (e.g. `tissue_1_boundary_positions_1.txt`, `tissue_1_boundary_positions_2.txt`,
  `tissue_2_boundary_positions_1.txt`, …).
- Notebook 02 runs in **multi** mode: boundaries are visited in order
  (tissue, then boundary) with a transit segment between consecutive boundaries
  (wrapping the last back to the first). It creates
  `data/tissue_{t}/{cells,hybs,transit}` and `data/mosaic10x`, and writes
  per-segment positions files plus per-tissue FOV-only files.
- Notebook 03 emits a per-segment Dave recipe (a `<loop>` per boundary/transit
  movie) using the transit HAL config from notebook 01.
- If `positions/` is empty, notebook 02 falls back to the bundled example set
  selected by `EXAMPLE_LAYOUT` (default `"multi"` here) under
  `MERci/data/positions/examples/`, and copies that example's boundary + hole
  inputs into `positions/` so notebooks 03/04 find them. Lets you test the
  pipeline immediately.

## Parameters to set for a lineage-tracing lineage run

Set these in the notebooks to match the assay (values left at the `reference`
defaults until confirmed):

- `01`: `MICROSCOPE`, `POWER`, `color_seq`, z-range, `EXPOSURE_TIME`,
  `N_TRANSIT_BLANK`.
- `02`: `TRANSIT_SPACING`, `SCAN_DIRECTION`.
- `03`: `N_HYBS`, `USE_ADAPTORS`, `FIRST_HYB_NO_CLEAVE`, `INCLUDE_FINAL_CLEAVE`.
- `04`: `round_bit_color` mapping to match the codebook. Note: multi-tissue
  MERlin analysis is per tissue / per boundary — confirm the intended workflow
  before relying on the generated data-organization.
