# prepare_imaging / reference

Canonical, fully-featured copies of the four pre-experiment notebooks. The
`tumor/` and `lineage_tracing/` folders are copies of these with their parameters
tuned for a specific experiment type — edit those for real experiments and keep
this `reference/` set as the up-to-date template.

Run order (same for every variant):

1. `01_create_hal_config_and_shutters` — imaging sequence, per-channel `POWER`,
   HAL/shutter for bits + cells, and the **transit** HAL config (blank frames).
2. `02_create_positions_from_tissue_boundary` — FOV grid per boundary + transit
   segments; writes per-segment / per-tissue positions files and the `data/`
   subfolders. Auto-detects the layout from the boundary filenames.
3. `03_create_dave_config` — `round_info.csv` + Dave recipe (per-segment movies
   when >1 boundary, else single-positions).
4. `04_create_data_organization` — MERlin data organization + Dave bit annotation.

Boundary-file layouts recognised by notebook 02 (in `positions/`):

- **multi**  — `tissue_{t}_boundary_positions_{b}.txt` (several sections)
- **single** — `boundary_positions_{b}.txt` (one section, several boundaries)
- **legacy** — `boundary_positions.txt` (one boundary)

Each notebook resolves `MERCI_DIR = Path(os.getcwd()).parent.parent.parent`
because it lives three levels under the repo root
(`MERci/notebooks/prepare_imaging/<variant>/`).
