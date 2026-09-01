# MERci/acquisition/pipeline_export.py
"""
Export one before_imaging pipeline, plus the shared during_imaging/
after_imaging notebooks, into a standalone ``SAMPLE_DIR/notebooks/`` tree
that sits *alongside* the MERci clone (``SAMPLE_DIR/MERci/``) instead of
inside it. Used by ``notebooks/before_imaging/00_select_pipeline.ipynb``. The MERci
clone itself is only ever read from, never modified.

If the chosen pipeline has a ``pipeline.yaml`` (every pipeline except
`multi_z` -- see ``pipeline_config.py``), it's copied (with its
round_bit_color CSV) to ``notebooks/pipeline.yaml``/``notebooks/
round_bit_color.csv``, and every notebook that loads it is rewritten to
read *that* copy instead of the one under ``MERci/data/pipelines/`` -- so
it can be hand-edited per experiment without touching the MERci clone. The
shared (not per-experiment) per-microscope power table it also needs still
comes from the MERci clone; see ``_rewrite_pipeline_config_line`` and
``load_pipeline_config``'s own ``data_dir`` parameter.

Path-detection change
----------------------
Every notebook's first cell resolves ``MERCI_DIR`` by counting parent
directories up from its own location (see the "Deployment model" section of
CLAUDE.md) -- inside ``MERci/notebooks/`` that count depends on how deeply
the notebook is nested (2-4 levels, per variant). Once exported, every copy
sits at the same depth -- ``SAMPLE_DIR/notebooks/<stage>/<name>.ipynb`` --
with ``MERci`` now a *sibling* of ``notebooks/`` instead of an ancestor, so a
single fixed ``MERCI_DIR`` line works for every exported notebook regardless
of its original nesting; see ``_rewrite_merci_dir_line``.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, NamedTuple

import yaml


# ── Pipeline registry ────────────────────────────────────────────────────────
# id -> before_imaging/ subpath, relative to MERci/notebooks/

PIPELINES: Dict[str, str] = {
    "tumor_epi":                       "before_imaging/regular",
    "tumor_disk":                      "before_imaging/regular",
    "lineage_tracing_merfish":         "before_imaging/regular",
    "lineage_tracing_lineage":         "before_imaging/regular",
    "multi_z":                         "before_imaging/multi_z",
}

# before_imaging/regular/ holds both backends' 05/07 side by side -- exactly
# one pair is copied per export, picked by the chosen pipeline's
# analysis_backend (see export_pipeline_notebooks).
_MERLIN_ONLY_NAMES   = {"05_create_data_organization.ipynb", "07_create_merlin_scripts.ipynb"}
_FISHTANK_ONLY_NAMES = {"05_create_color_usage.ipynb", "07_create_fishtank_scripts.ipynb"}


class PipelineInfo(NamedTuple):
    source: str        # before_imaging/ subpath, relative to MERci/notebooks/
    description: str    # first paragraph of that variant's README.md


def _first_paragraph(readme_path: Path) -> str:
    """First paragraph below the '# heading' of a README.md, or "" if missing."""
    if not readme_path.exists():
        return ""
    para = []
    for line in readme_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            if para:
                break
            continue
        para.append(line.strip())
    return " ".join(para)


def describe_pipelines(merci_dir: Path) -> Dict[str, PipelineInfo]:
    """Pipeline id -> (source subpath, description), for the notebook's
    "available pipelines" display cell. Several ids share one `source`
    folder (`before_imaging/regular/`), so the description comes from each
    pipeline's own `pipeline.yaml` `label` (falling back to that folder's
    README.md first paragraph for a pipeline with no pipeline.yaml, e.g.
    `multi_z`)."""
    notebooks_dir = merci_dir / "notebooks"
    out = {}
    for pid, src in PIPELINES.items():
        yaml_path = merci_dir / "data" / "pipelines" / pid / "pipeline.yaml"
        if yaml_path.exists():
            description = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["label"]
        else:
            description = _first_paragraph(notebooks_dir / src / "README.md")
        out[pid] = PipelineInfo(source=src, description=description)
    return out


# ── MERCI_DIR rewrite ─────────────────────────────────────────────────────────

_MERCI_DIR_RE = re.compile(
    r"MERCI_DIR(\s*)=(\s*)Path\(os\.getcwd\(\)\)(?:\.parent)+.*"
)
_MERCI_DIR_REPLACEMENT = (
    'MERCI_DIR  = Path(os.getcwd()).parent.parent / "MERci"  '
    "# MERci/ (sibling of this notebooks/ folder -- see notebooks/README.md)"
)


def _rewrite_merci_dir_line(notebook: dict) -> bool:
    """Rewrite MERCI_DIR's parent-counting line to the fixed sibling-MERci
    formula, in every code cell of `notebook` (in place). Returns whether a
    match was found -- every exported notebook is expected to have exactly
    one."""
    found = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        is_str = isinstance(src, str)
        text = src if is_str else "".join(src)
        new_text, n = _MERCI_DIR_RE.subn(_MERCI_DIR_REPLACEMENT, text)
        if n:
            found = True
            cell["source"] = new_text if is_str else new_text.splitlines(keepends=True)
    return found


# ── PIPELINE_CONFIG rewrite ──────────────────────────────────────────────────
# Only the MERlin-based pipelines' before_imaging notebooks load a
# pipeline.yaml (see PIPELINES/pipeline_config.py) -- a miss here is not an
# error, just a notebook (or whole pipeline) that doesn't use one.

_PIPELINE_CONFIG_RE = re.compile(
    r'PIPELINE_CONFIG(\s*)=(\s*)load_pipeline_config\('
    r'MERCI_DIR\s*/\s*"data"\s*/\s*"pipelines"\s*/\s*PIPELINE_ID\s*/\s*"pipeline\.yaml"'
    r'\)'
)
_PIPELINE_CONFIG_REPLACEMENT = (
    'PIPELINE_CONFIG = load_pipeline_config('
    'Path(os.getcwd()).parent / "pipeline.yaml", data_dir=MERCI_DIR / "data")  '
    "# edit pipeline.yaml here, not in MERci/"
)


def _rewrite_pipeline_config_line(notebook: dict) -> bool:
    """Rewrite `PIPELINE_CONFIG = load_pipeline_config(MERCI_DIR / ...)` to
    load the pipeline.yaml exported alongside this notebooks/ folder instead
    (still passing MERCI_DIR/data as data_dir, for the shared power table),
    in every code cell of `notebook` (in place). Returns whether a match was
    found."""
    found = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        is_str = isinstance(src, str)
        text = src if is_str else "".join(src)
        new_text, n = _PIPELINE_CONFIG_RE.subn(_PIPELINE_CONFIG_REPLACEMENT, text)
        if n:
            found = True
            cell["source"] = new_text if is_str else new_text.splitlines(keepends=True)
    return found


# ── README adaptation ─────────────────────────────────────────────────────────
# Both phrasings used across the six variant READMEs for the stale
# parent-counting explanation (see each variant's own README.md).
_STALE_LEVELS_RES = [
    re.compile(
        r"Each notebook resolves `MERCI_DIR[^`]*`\s+because it lives \w+ "
        r"levels under the repo root\s+\([^)]*\)\.\n?"
    ),
    re.compile(
        r"These notebooks live \*\*\w+\*\* levels under the repo root\s+"
        r"\([^)]*\),\s+so they\s+resolve\s+`MERCI_DIR[^`]*`\.\n?"
    ),
]

_MERCI_DIR_NOTE = (
    "Every notebook here resolves "
    '`MERCI_DIR = Path(os.getcwd()).parent.parent / "MERci"`, since `MERci/` '
    "sits alongside this `notebooks/` folder (`SAMPLE_DIR/MERci/`, "
    "`SAMPLE_DIR/notebooks/`).\n"
)


def _adapt_readme(pipeline_src: Path, pipeline_id: str, has_pipeline_yaml: bool) -> str:
    """Base the new notebooks/README.md on the pipeline's own README.md:
    drop the stale parent-counting explanation, note the new sibling-MERci
    path, and describe the exported folder layout."""
    readme_path = pipeline_src / "README.md"
    base = readme_path.read_text(encoding="utf-8") if readme_path.exists() else f"# {pipeline_id}\n"

    for pat in _STALE_LEVELS_RES:
        base, n = pat.subn(_MERCI_DIR_NOTE, base)
        if n:
            break

    pipeline_yaml_line = (
        "  pipeline.yaml     this pipeline's config -- edit here, not in MERci/\n"
        "  round_bit_color.csv  round/bit/color assignment referenced by pipeline.yaml\n"
        if has_pipeline_yaml else ""
    )
    layout_note = (
        "\n## Folder structure\n\n"
        f"This `notebooks/` folder was generated for the `{pipeline_id}` "
        "pipeline by `MERci/notebooks/before_imaging/00_select_pipeline.ipynb`.\n\n"
        "```\n"
        "notebooks/\n"
        "  before_imaging/   this pipeline's pre-experiment notebooks, run in order\n"
        "  during_imaging/   live QC notebooks, run during acquisition\n"
        "  after_imaging/    online-analysis notebooks, run during/after acquisition\n"
        f"{pipeline_yaml_line}"
        "```\n\n"
        "The `MERci/` clone this was exported from is untouched; re-run "
        "`00_select_pipeline.ipynb` there to regenerate this folder.\n"
    )
    return base.rstrip() + "\n" + layout_note


# ── Export ────────────────────────────────────────────────────────────────────

def _copy_notebooks(src_dir: Path, dst_dir: Path, exclude: set = frozenset()) -> None:
    for nb_path in sorted(src_dir.glob("*.ipynb")):
        if nb_path.name in exclude:
            continue
        notebook = json.loads(nb_path.read_text(encoding="utf-8"))
        if not _rewrite_merci_dir_line(notebook):
            raise ValueError(f"No MERCI_DIR line found in {nb_path}")
        _rewrite_pipeline_config_line(notebook)
        (dst_dir / nb_path.name).write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
        )


_ROUND_BIT_COLOR_CSV_RE = re.compile(r"(round_bit_color_csv:\s*)\S+")


def _copy_pipeline_config(merci_dir: Path, pipeline_id: str, out_dir: Path) -> bool:
    """Copy `pipeline_id`'s pipeline.yaml and its round_bit_color CSV
    (MERCI_DIR/data/pipelines/<id>/) to `out_dir/pipeline.yaml` /
    `out_dir/round_bit_color.csv`, rewriting the copied yaml's
    round_bit_color_csv line to point at the flat copy (text substitution,
    not a yaml round-trip, so pipeline.yaml's comments survive). This is
    the copy every exported notebook is rewritten to read and edit -- see
    `_rewrite_pipeline_config_line`.

    Not every pipeline has a pipeline.yaml (`multi_z` doesn't yet -- see
    pipeline_config.py); if missing, remove any stale copy left over from a
    previous `force=True` export of a different pipeline. Returns whether
    one was copied."""
    dst_yaml = out_dir / "pipeline.yaml"
    dst_csv  = out_dir / "round_bit_color.csv"
    src_dir  = merci_dir / "data" / "pipelines" / pipeline_id
    src_yaml = src_dir / "pipeline.yaml"
    if not src_yaml.exists():
        dst_yaml.unlink(missing_ok=True)
        dst_csv.unlink(missing_ok=True)
        return False

    text = src_yaml.read_text(encoding="utf-8")
    csv_relpath = yaml.safe_load(text)["dataorganization"]["round_bit_color_csv"]
    shutil.copy2(src_dir / csv_relpath, dst_csv)

    new_text, n = _ROUND_BIT_COLOR_CSV_RE.subn(rf"\g<1>{dst_csv.name}", text)
    if not n:
        raise ValueError(f"No round_bit_color_csv line found in {src_yaml}")
    dst_yaml.write_text(new_text, encoding="utf-8")
    return True


def export_pipeline_notebooks(
    merci_dir: Path,
    sample_dir: Path,
    pipeline_id: str,
    force: bool = False,
) -> Path:
    """
    Copy `pipeline_id`'s before_imaging notebooks (flattened, no variant
    subfolders) plus the shared during_imaging/after_imaging notebooks into
    `sample_dir/notebooks/`, rewriting each copy's MERCI_DIR line for the new
    sibling-MERci layout; copy `pipeline_id`'s pipeline.yaml + round_bit_color
    CSV (if it has one) to `sample_dir/notebooks/`, rewriting every notebook
    that loads it to read that copy instead of MERci's; and write
    notebooks/README.md (adapted from the pipeline's own README.md). Returns
    the new notebooks/ directory.

    Raises FileExistsError if `sample_dir/notebooks/` already exists, unless
    `force=True`. Never modifies `merci_dir`.
    """
    if pipeline_id not in PIPELINES:
        raise ValueError(
            f"Unknown pipeline {pipeline_id!r}; choices: {sorted(PIPELINES)}"
        )

    notebooks_dir = merci_dir / "notebooks"
    out_dir = sample_dir / "notebooks"
    if out_dir.exists() and not force:
        raise FileExistsError(
            f"{out_dir} already exists; pass force=True to overwrite"
        )

    out_before = out_dir / "before_imaging"
    out_during = out_dir / "during_imaging"
    out_after  = out_dir / "after_imaging"
    for d in (out_before, out_during, out_after):
        d.mkdir(parents=True, exist_ok=True)

    has_pipeline_yaml = _copy_pipeline_config(merci_dir, pipeline_id, out_dir)

    # before_imaging/regular/ holds both backends' 05/07 side by side --
    # copy only the pair matching this pipeline's analysis_backend.
    exclude = set()
    if has_pipeline_yaml:
        src_yaml_path = merci_dir / "data" / "pipelines" / pipeline_id / "pipeline.yaml"
        backend = yaml.safe_load(src_yaml_path.read_text(encoding="utf-8"))["analysis_backend"]
        exclude = _FISHTANK_ONLY_NAMES if backend == "merlin" else _MERLIN_ONLY_NAMES

    pipeline_src = notebooks_dir / PIPELINES[pipeline_id]
    _copy_notebooks(pipeline_src, out_before, exclude=exclude)
    _copy_notebooks(notebooks_dir / "during_imaging", out_during)
    _copy_notebooks(notebooks_dir / "after_imaging", out_after)

    (out_dir / "README.md").write_text(
        _adapt_readme(pipeline_src, pipeline_id, has_pipeline_yaml), encoding="utf-8"
    )

    return out_dir
