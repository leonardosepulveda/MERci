# MERci/acquisition/pipeline_export.py
"""
Export one before_imaging pipeline, plus the shared during_imaging/
after_imaging notebooks, into a standalone ``SAMPLE_DIR/notebooks/`` tree
that sits *alongside* the MERci clone (``SAMPLE_DIR/MERci/``) instead of
inside it. Used by ``notebooks/setup/00_select_pipeline.ipynb``. The MERci
clone itself is only ever read from, never modified.

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
from pathlib import Path
from typing import Dict, NamedTuple


# ── Pipeline registry ────────────────────────────────────────────────────────
# id -> before_imaging/ subpath, relative to MERci/notebooks/

PIPELINES: Dict[str, str] = {
    "reference":                       "before_imaging/reference",
    "tumor_epi":                       "before_imaging/tumor/epi",
    "tumor_disk":                      "before_imaging/tumor/disk",
    "lineage_tracing_merfish":         "before_imaging/lineage_tracing/merfish",
    "lineage_tracing_lineage":         "before_imaging/lineage_tracing/lineage",
    "lineage_tracing_merfish_multi_z": "before_imaging/lineage_tracing/merfish_multi_z",
}


class PipelineInfo(NamedTuple):
    source: str        # before_imaging/ subpath, relative to MERci/notebooks/
    description: str    # first paragraph of that variant's README.md


def _first_paragraph(readme_path: Path) -> str:
    """First paragraph below the '# heading' of a README.md, or "" if missing."""
    if not readme_path.exists():
        return ""
    para = []
    for line in readme_path.read_text().splitlines()[1:]:
        if not line.strip():
            if para:
                break
            continue
        para.append(line.strip())
    return " ".join(para)


def describe_pipelines(merci_dir: Path) -> Dict[str, PipelineInfo]:
    """Pipeline id -> (source subpath, description), for the notebook's
    "available pipelines" display cell."""
    notebooks_dir = merci_dir / "notebooks"
    return {
        pid: PipelineInfo(
            source=src,
            description=_first_paragraph(notebooks_dir / src / "README.md"),
        )
        for pid, src in PIPELINES.items()
    }


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


def _adapt_readme(pipeline_src: Path, pipeline_id: str) -> str:
    """Base the new notebooks/README.md on the pipeline's own README.md:
    drop the stale parent-counting explanation, note the new sibling-MERci
    path, and describe the exported folder layout."""
    readme_path = pipeline_src / "README.md"
    base = readme_path.read_text() if readme_path.exists() else f"# {pipeline_id}\n"

    for pat in _STALE_LEVELS_RES:
        base, n = pat.subn(_MERCI_DIR_NOTE, base)
        if n:
            break

    layout_note = (
        "\n## Folder structure\n\n"
        f"This `notebooks/` folder was generated for the `{pipeline_id}` "
        "pipeline by `MERci/notebooks/setup/00_select_pipeline.ipynb`.\n\n"
        "```\n"
        "notebooks/\n"
        "  before_imaging/   this pipeline's pre-experiment notebooks, run in order\n"
        "  during_imaging/   live QC notebooks, run during acquisition\n"
        "  after_imaging/    online-analysis notebooks, run during/after acquisition\n"
        "```\n\n"
        "The `MERci/` clone this was exported from is untouched; re-run "
        "`00_select_pipeline.ipynb` there to regenerate this folder.\n"
    )
    return base.rstrip() + "\n" + layout_note


# ── Export ────────────────────────────────────────────────────────────────────

def _copy_notebooks(src_dir: Path, dst_dir: Path) -> None:
    for nb_path in sorted(src_dir.glob("*.ipynb")):
        notebook = json.loads(nb_path.read_text())
        if not _rewrite_merci_dir_line(notebook):
            raise ValueError(f"No MERCI_DIR line found in {nb_path}")
        (dst_dir / nb_path.name).write_text(json.dumps(notebook, indent=1))


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
    sibling-MERci layout, and writing notebooks/README.md (adapted from the
    pipeline's own README.md). Returns the new notebooks/ directory.

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

    pipeline_src = notebooks_dir / PIPELINES[pipeline_id]
    _copy_notebooks(pipeline_src, out_before)
    _copy_notebooks(notebooks_dir / "during_imaging", out_during)
    _copy_notebooks(notebooks_dir / "after_imaging", out_after)

    (out_dir / "README.md").write_text(_adapt_readme(pipeline_src, pipeline_id))

    return out_dir
