# MERci/acquisition/pipeline_config.py
"""
Per-pipeline configuration: microscope/imaging-recipe/fluidics/analysis
defaults for one before_imaging pipeline (tumor/epi, lineage_tracing/lineage,
etc.), loaded from ``data/pipelines/<pipeline_id>/pipeline.yaml``.

Values here are pipeline-level -- the same for every experiment run through
that pipeline. Per-experiment values (tissue-segmentation thresholds,
positions boundaries, `sequential_genes.csv` contents, the actual gene
targets, etc.) are NOT here; those stay hand-edited in each notebook's own
"Experiment parameters" cell, per notebook run.

Per-channel laser power is NOT part of a pipeline's own YAML: it's locked to
the microscope choice (hardware/alignment property, not a pipeline dial),
so it's looked up from ``data/configs/power/power_by_microscope.yaml``
instead -- see ``load_pipeline_config``.

See ``prompt_history/2026_08_28_1655_list_pipeline_yaml_variables.md``,
``prompt_history/2026_08_28_1712_pipeline_yaml_implementation_plan.md`` and
the entry revising this schema (round_bit_color/analysis/metadata moved
under a `merlin` section, power moved out entirely) for how this was
derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


@dataclass
class ImagingRoundRecipe:
    """One HAL/shutter z-stack recipe (notebook 01's "cells" or "bits" cell)."""
    z_min:          float
    z_max:          float
    z_step:         float
    bead_seq:       List[float]
    color_seq:      List[float]
    end_seq:        List[float]
    scan_mode:      str   = "interleaved"
    z_return_mode:  str   = "progressive"
    return_step:    float = 5


@dataclass
class MerlinConfig:
    """Everything MERlin-specific for one pipeline (notebooks 05/07, merlin
    backend only). ``round_bit_color`` is NOT here -- it's backend-agnostic
    (round_info.csv needs it regardless of analysis backend), so it lives on
    ``PipelineConfig`` directly -- see that class's own docstring."""
    project:                str
    lib_name:                str
    n_optimize_iterations:    int
    tasks:                    Dict[str, bool]      # atom name -> enabled, full menu
    overrides:                Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Which non-barcode companion codebook this pipeline's sequential bits
    # come from -- "sequential" (default, e.g. an epi pipeline's RNA panel)
    # or "immuno" (e.g. a disk pipeline's antibody panel) -- see
    # merlin_config.resolve_sequential_codebook_filename's own docstring.
    sequential_kind:          str = "sequential"

    @property
    def enabled_tasks(self) -> List[str]:
        """Enabled atom names, in the order they're listed in pipeline.yaml."""
        return [name for name, enabled in self.tasks.items() if enabled]


@dataclass
class FishtankTarget:
    """One decode-strategy row for a fishtank pipeline's notebook 05
    (`decoding_strategy_*.csv`). ``reference_file``/``whitelist`` are bare
    filenames under ``{FISHTANK_CLUSTER_DIR}/reference/`` (a per-experiment
    cluster path, not known at pipeline.yaml-authoring time) -- notebook 05
    joins them at run time. ``{lib_version}`` in either filename is already
    substituted with ``FishtankConfig.lineage_lib_version`` by
    ``load_pipeline_config``."""
    name:            str
    method:          str
    reference_file:  str = ""
    whitelist:       str = ""


@dataclass
class FishtankConfig:
    """Everything fishtank-specific for one pipeline (notebooks 05/07,
    fishtank backend only)."""
    lineage_lib_version:  str
    color_usage_colors:    List[str]
    targets:                List[FishtankTarget] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """All pipeline-level values for one before_imaging pipeline.

    ``round_bit_color`` is required for every pipeline regardless of
    ``analysis_backend`` -- notebook 03 (round_info.csv) needs it either way.
    Exactly one of ``merlin``/``fishtank`` is populated, matching
    ``analysis_backend`` -- notebooks 05/07 are the only ones that read
    either.
    """
    id:                str
    label:             str
    analysis_backend:  str   # "merlin" | "fishtank"
    microscope:        str
    objective:         str

    # ── Imaging (notebook 01) ────────────────────────────────────────────
    file_type:        str
    power:            Dict[int, float]   # locked to `microscope`, not pipeline.yaml -- see load_pipeline_config
    power_default:    float
    exposure_time:    float
    cells_round:      ImagingRoundRecipe
    bits_round:       ImagingRoundRecipe
    focus_test_z:     float
    transit_n_blank:  int

    # ── Round/bit/color (notebook 03) -- backend-agnostic ────────────────
    round_bit_color:  List[tuple]   # [(round, bit, color), ...]

    # ── Fluidics (notebook 04) ───────────────────────────────────────────
    use_adaptors:           bool
    include_final_cleave:   bool
    first_hyb_no_cleave:    bool

    # ── Backend-specific (notebooks 05/07) -- exactly one populated ──────
    merlin:    Optional[MerlinConfig]   = None
    fishtank:  Optional[FishtankConfig] = None

    source_path: Path = field(repr=False, default=None)


def _load_round_bit_color(yaml_path: Path, data_dir: Path, csv_relpath: str) -> List[tuple]:
    # Resolved relative to data_dir for the in-repo layout (e.g.
    # data/configs/round_bit_color_map/<id>.csv), but tried relative to
    # yaml_path's own directory first -- pipeline_export.py rewrites an
    # exported pipeline.yaml's round_bit_color_csv to a bare filename
    # pointing at its own local flat copy (round_bit_color.csv next to that
    # copy), which only resolves against the yaml's own directory, not
    # data_dir (still the shared MERci clone's data/ for the export case).
    csv_path = yaml_path.parent / csv_relpath
    if not csv_path.exists():
        csv_path = data_dir / csv_relpath
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- round_bit_color_csv in {yaml_path.name} "
            f"points at a file that doesn't exist."
        )
    df = pd.read_csv(csv_path)
    return [
        (int(r), int(b), int(c))
        for r, b, c in df[["round", "bit", "color"]].itertuples(index=False, name=None)
    ]


def _load_power(data_dir: Path, microscope: str) -> tuple[Dict[int, float], float]:
    """Per-channel laser power for `microscope`, from
    data_dir/configs/power/power_by_microscope.yaml."""
    power_path = data_dir / "configs" / "power" / "power_by_microscope.yaml"
    if not power_path.exists():
        raise FileNotFoundError(f"{power_path} not found.")
    table = yaml.safe_load(power_path.read_text())
    if microscope not in table:
        raise KeyError(f"No power defaults for microscope {microscope!r} in {power_path}.")
    entry = table[microscope]
    return {int(k): float(v) for k, v in entry["power"].items()}, float(entry["power_default"])


def load_pipeline_config(yaml_path: Path, data_dir: Path = None) -> PipelineConfig:
    """
    Load one pipeline's ``pipeline.yaml`` (+ its round_bit_color CSV,
    resolved relative to ``data_dir`` -- lives under
    ``data_dir/configs/round_bit_color_map/``, like the power table) into a
    ``PipelineConfig``.

    `data_dir` locates the shared per-microscope power table
    (``data_dir/configs/power/power_by_microscope.yaml``) -- defaults to
    ``yaml_path.parents[2]`` (``MERCI_DIR/data``, when `yaml_path` is
    ``MERCI_DIR/data/pipelines/<id>/pipeline.yaml``, its usual in-repo
    location). Pass it explicitly when `yaml_path` is a copy that no longer
    sits at that fixed depth under ``data/`` -- e.g. a pipeline.yaml exported
    into ``SAMPLE_DIR/notebooks/`` by pipeline_export.py, which still needs
    the *shared* (not per-experiment) power table from the original
    ``MERci/data/``.

    Raises ``KeyError``/``FileNotFoundError`` loudly on anything missing --
    no silent defaults for pipeline-defining values.
    """
    yaml_path = Path(yaml_path)
    if data_dir is None:
        data_dir = yaml_path.parents[2]   # .../data/pipelines/<id>/pipeline.yaml -> .../data
    raw = yaml.safe_load(yaml_path.read_text())

    imaging          = raw["imaging"]
    fluidics         = raw["fluidics"]
    dataorganization = raw["dataorganization"]
    backend          = raw["analysis_backend"]

    def _recipe(d: dict) -> ImagingRoundRecipe:
        return ImagingRoundRecipe(**d)

    power, power_default = _load_power(data_dir, raw["microscope"])
    round_bit_color = _load_round_bit_color(yaml_path, data_dir, dataorganization["round_bit_color_csv"])

    merlin_config = None
    if "merlin" in raw:
        merlin = raw["merlin"]
        merlin_config = MerlinConfig(
            project                = merlin["project"],
            lib_name                = merlin["lib_name"],
            n_optimize_iterations     = merlin["analysis"]["n_optimize_iterations"],
            tasks                      = dict(merlin["analysis"]["tasks"]),
            overrides                   = dict(merlin["analysis"].get("overrides", {})),
            sequential_kind             = merlin.get("sequential_kind", "sequential"),
        )

    fishtank_config = None
    if "fishtank" in raw:
        fishtank = raw["fishtank"]
        lib_version = fishtank["lineage_lib_version"]
        fishtank_config = FishtankConfig(
            lineage_lib_version  = lib_version,
            color_usage_colors    = list(fishtank["color_usage_colors"]),
            targets                 = [
                FishtankTarget(
                    name           = t["name"],
                    method          = t["method"],
                    reference_file   = t.get("reference_file", "").format(lib_version=lib_version),
                    whitelist         = t.get("whitelist", "").format(lib_version=lib_version),
                )
                for t in fishtank.get("targets", [])
            ],
        )

    if backend == "merlin" and merlin_config is None:
        raise KeyError(f"{yaml_path}: analysis_backend='merlin' but no 'merlin' section.")
    if backend == "fishtank" and fishtank_config is None:
        raise KeyError(f"{yaml_path}: analysis_backend='fishtank' but no 'fishtank' section.")

    return PipelineConfig(
        id                = raw["id"],
        label             = raw["label"],
        analysis_backend  = backend,
        microscope        = raw["microscope"],
        objective         = raw["objective"],

        file_type        = imaging["file_type"],
        power            = power,
        power_default    = power_default,
        exposure_time    = imaging["exposure_time"],
        cells_round      = _recipe(imaging["cells_round"]),
        bits_round       = _recipe(imaging["bits_round"]),
        focus_test_z     = imaging["focus_test_z"],
        transit_n_blank  = imaging["transit_n_blank"],

        round_bit_color  = round_bit_color,

        use_adaptors           = fluidics["use_adaptors"],
        include_final_cleave   = fluidics["include_final_cleave"],
        first_hyb_no_cleave    = fluidics["first_hyb_no_cleave"],

        merlin    = merlin_config,
        fishtank   = fishtank_config,

        source_path = yaml_path,
    )
