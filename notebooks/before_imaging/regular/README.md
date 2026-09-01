# before_imaging / regular

Pre-experiment notebooks shared by every pipeline **except** `multi_z`
(variable-z-per-FOV, its own sequence -- see `../multi_z/README.md`):
`tumor_epi`, `tumor_disk`, `lineage_tracing_merfish`, `lineage_tracing_lineage`.
One notebook set instead of one copy per pipeline -- what used to differ
between them (microscope, imaging recipe, fluidics, codebook/task menu or
fishtank targets) now lives entirely in each pipeline's own
`MERci/data/pipelines/<id>/pipeline.yaml`
(`MERci.acquisition.pipeline_config.load_pipeline_config`).

These notebooks live **three** levels under the repo root
(`MERci/notebooks/before_imaging/regular/`), so they resolve
`MERCI_DIR = Path(os.getcwd()).parent.parent.parent`.

## Choosing a pipeline

Every notebook's second cell sets `PIPELINE_ID` (one of the ids in
`MERci.acquisition.pipeline_export.PIPELINES`) and loads that pipeline's
`pipeline.yaml` into `PIPELINE_CONFIG` -- everything else in the notebook
reads from there. Set the same `PIPELINE_ID` in every notebook for one run.

## Notebook sequence (10 notebooks, two pairs)

| # | Notebook | Backend |
|---|----------|---------|
| 01 | `01_create_hal_config_and_shutters.ipynb` | either |
| 02 | `02_create_boundary_from_mosaic.ipynb` (optional) | either |
| 02 | `02_create_positions_from_boundaries.ipynb` | either |
| 03 | `03_create_round_info.ipynb` | either |
| 04 | `04_create_dave_config.ipynb` | either |
| 05 | `05_create_data_organization.ipynb` | `analysis_backend: merlin` |
| 05 | `05_create_color_usage.ipynb` | `analysis_backend: fishtank` |
| 06 | `06_create_experiment_info.ipynb` | either |
| 07 | `07_create_merlin_scripts.ipynb` | `analysis_backend: merlin` |
| 07 | `07_create_fishtank_scripts.ipynb` | `analysis_backend: fishtank` |

Run steps 01-04 and 06 regardless of pipeline; for 05/07, run the file
matching your pipeline's `analysis_backend` (`00_select_pipeline.ipynb`
exports the right one automatically once you pick a pipeline).

## What still isn't pipeline.yaml-driven

`02_create_boundary_from_mosaic.ipynb`/`02_create_positions_from_boundaries.ipynb`
are genuinely per-experiment (tissue-segmentation thresholds, boundary/grid
choices) -- they don't read `PIPELINE_CONFIG` and aren't expected to; set
`MICROSCOPE` there to match `PIPELINE_CONFIG.microscope` by hand (see each
notebook's own "Experiment parameters" cell). If `positions/` is empty, they
fall back to a bundled example dataset so the notebook chain can still be
run/tested before any real scan exists.

## History

Consolidated from `tumor/{epi,disk}/` and `lineage_tracing/{merfish,lineage}/`
(4 near-duplicate 7-8-notebook copies), reconciling the real differences that
had grown between them rather than just picking one copy. The most
significant: the old `tumor_epi/07_create_merlin_scripts.ipynb` re-added
`smfish_signal`/`sum_signal`/`export_sum_signals` to the generated MERlin
recipe on top of `pipeline.yaml` already enabling them statically -- a real
duplicate-task bug (MERlin's task-name list has no deduplication), fixed
here by only adding a task dynamically when pipeline.yaml hasn't already
enabled it, while still supplying the per-experiment channel-name override
`pipeline.yaml` can't know ahead of time.
