# Notebook coding guidelines

Architectural rules for every notebook under `notebooks/` -- apply these when
creating a new notebook, or modifying an existing one. This is not a prose/
comment style guide (see the root `CLAUDE.md`'s "Code clarity" section for
that); it's specifically about cell structure, caching, progress reporting,
and plot legibility.

Reference implementation: `notebooks/misc/measure_tissue_thickness.ipynb`.

## 1. Separate calculation from display

If a step involves both a nontrivial calculation and a plot or printed
result, put the calculation in its own cell and the plot/print in the next
cell. Tweaking a plot's colors, labels, or bin count should never require
re-running the calculation that feeds it.

Trivial, near-instant lookups (reading one CSV, resolving one round id)
don't need to be split out just to satisfy this rule -- the point is to
protect genuinely slow work (a loop over many FOVs/files/frames), not to
fragment every single-line print into its own cell.

## 2. Cache every nontrivial calculation cell's result

Save to `SAMPLE_DIR/analysis/cache/<notebook_name>/<step_name>.<ext>`:
`.npz` (via `np.savez_compressed`) for numeric arrays or ragged per-item
data, `.csv` for tabular per-FOV/per-round results, `.json` for a handful of
scalars. This makes calculation cells self-contained: rerunning the notebook
(a fresh kernel, a re-opened cluster session) never needs to recompute a
result that's already on disk.

`MERci.analysis.fov.save_channel_counters`/`load_channel_counters` (and the
underlying `_atomic_save`) are the reference pattern for the ragged-array
case; a plain `pd.DataFrame.to_csv`/`pd.read_csv` round-trip covers the
tabular case.

## 3. Skip recomputation when a valid cache already exists

At the top of a calculation cell, check whether its cache file exists and
still matches the current inputs, and load it instead of recomputing when it
does. Pick whatever invalidation signal is actually relevant to that
calculation -- matching FOV/frame count, matching threshold/parameter
values, or (for a per-item cache, like Counters) checking cache existence
file-by-file and only computing the missing ones.
`measure_tissue_thickness.ipynb` section 4 (`compute_channel_counters`) is
the canonical version: it checks every FOV's own cache file individually and
only computes what's actually missing, rather than an all-or-nothing cache
for the whole round.

## 4. Report progress in every nontrivial calculation cell

Any loop over more than a handful of items (FOVs, files, frames) should show
n/total, percent complete, elapsed time, and an ETA. Use
`MERci.progress_display.ProgressReporter`:

```python
from MERci.progress_display import ProgressReporter

reporter = ProgressReporter(total=len(items), label="Doing the thing")
for item in reporter.wrap(items):
    ...
```

(or drive `reporter.update()`/`reporter.done()` manually when `wrap()`
doesn't fit the loop shape). See `measure_tissue_thickness.ipynb` section 4
for the reference usage.

## 5. Keep plot fonts legible

Matplotlib's default font sizes shrink relative to `figsize`, so a
wide/short figure (e.g. a heatmap laid out across many FOV columns) ends up
with unreadably small titles/labels/ticks even though a squarer figure in
the same notebook looks fine. Set explicit font sizes for every title, axis
label, tick, and legend rather than relying on the default --
`measure_tissue_thickness.ipynb`'s `PLOT_TITLE_FONTSIZE` /
`PLOT_LABEL_FONTSIZE` / `PLOT_TICK_FONTSIZE` / `PLOT_LEGEND_FONTSIZE`
constants (defined once in its Parameters section, reused by every plotting
cell) are the reference values -- reuse the same sizes (or the same pattern)
in new notebooks rather than picking new numbers per plot.

## 6. Save every displayed figure to `analysis/figures/`

Every cell that calls `plt.show()` on a real figure (not a quick throwaway
diagnostic) should also `fig.savefig(...)` a copy to
`{figures_dir}/{NOTEBOOK_NAME}.{figure_name}.png`, where `figures_dir` comes
from `MERci.visualization.get_merci_figures_dir(SAMPLE_DIR, category,
NOTEBOOK_NAME, subfolder=...)` -- it resolves to
`SAMPLE_DIR/figures/MERci/<category>/[<subfolder>/]<notebook_name>/`,
outside the `MERci/` clone (sibling of it, alongside MERlin's own
`figures/`), and creates the directory if missing. `NOTEBOOK_NAME` is the
notebook's own filename stem (e.g. `stage_z_drift`, defined once in the
Parameters/Setup section) and `figure_name` is a short, descriptive slug for
that specific figure (e.g. `stage_z_drift`, `stage_z_heatmap`). `category`
is the notebook's top-level folder under `notebooks/` (`before_imaging`,
`after_imaging`, `during_imaging`, `misc`, `tests`); `subfolder` is an
organizational subfolder *within* that category that groups otherwise-
unrelated notebooks (e.g. `fov_stitching` under `tests/`) -- omit it for
before_imaging's own `regular`/`multi_z` pipeline subfolders, since only one
pipeline's notebooks exist in a given experiment folder at a time. Routing
every notebook through this one function (rather than each notebook
constructing its own path) means the convention only has to change in one
place if it ever needs to. This gives every notebook's output figures one
shared, predictable location and naming scheme, so a later batch step (or a
human skimming the experiment folder) can find any notebook's plots without
knowing that notebook's own internal cell structure.
`during_imaging/stage_z_drift.ipynb` is the reference implementation.
