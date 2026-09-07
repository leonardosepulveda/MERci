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

## 7. Test notebooks: stay portable

`notebooks/tests/<subfolder>/` notebooks investigate one specific question
against real data, and their findings sometimes need to move to a
standalone location later (a different machine, a paper's supplementary
material, a collaborator without access to the source experiment tree).
Write them so that move is a copy, not a rewrite.

**Start from a local data folder, not the live experiment tree.** At the
top of the notebook, resolve a `DATA_DIR` under that notebook's own cache
mirror (its `cache/` path, per the root `CLAUDE.md`'s "Working / cache
files") with a `data/` subfolder there, and copy in only the specific files
the notebook actually reads (a boundary/positions file, a pre-built
mosaic-canvas array, one CSV) the first time it needs them. Every later
cell reads from `DATA_DIR`, never from the original `SAMPLE_DIR`/experiment
path directly -- that keeps the notebook from silently growing more live
dependencies as it's extended, and means nothing else has to change when
the notebook is later copied elsewhere.

**Too big to copy?** Split the notebook into a calculation section (reads
the real, possibly-huge source once, computes/downsamples, writes the
result under `DATA_DIR`) and a plotting/analysis section that reads only
from `DATA_DIR` from then on -- same split as guideline 1, just with the
boundary drawn at "touches the live experiment tree" instead of "is slow".
`MERci.acquisition.mosaic.load_or_build_mosaic_canvas_cached`'s own
`mosaic_canvas.npz` cache (a downsampled array, orders of magnitude smaller
than the raw mosaic tiles it's built from) is the reference pattern: don't
depend on the raw tiles at all once that cache exists -- load it directly
with `load_mosaic_canvas`.

**Note provenance.** Wherever a file under `DATA_DIR` is loaded, a short
comment or markdown cell should say where it came from: the source
experiment (by name, not by the path it happened to sit at when copied --
see the root `CLAUDE.md`'s "No Local-Only References" rule, which applies
here even though this data itself is gitignored) and which
notebook/function produced it.

**Prefer library functions over inline logic.** Before writing a
calculation as bespoke notebook code, check whether `MERci` already has
something for it (`acquisition/positions.py`'s grid/offset/path functions
are the usual candidates for FOV-geometry work). If the notebook needs
something genuinely reusable that doesn't exist yet, add it to `MERci`
proper and import it, rather than growing the same logic independently in
several test notebooks -- each new test can then contribute back to the
shared toolkit instead of duplicating it.

**Porting to a standalone folder**: `/save_test <subfolder> <destination>`
automates all of the above into a `{destination}/{MERci,data,notebooks,
figures}` layout -- a fresh `MERci` clone, only the data actually used,
notebook copies rewired to the local clone/data, and figures saved as
`{prefix}_{description}.{ext}` (`{prefix}` = the notebook's own leading
number, e.g. `01`). See that command's own definition for the full
contract.
