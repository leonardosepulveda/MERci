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
from typing import Any, Dict, List

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
    """Everything MERlin-specific for one pipeline (notebooks 03/05/06/07)."""
    project:                str
    lib_name:                str
    round_bit_color:          List[tuple]         # [(round, bit, color), ...]
    n_optimize_iterations:    int
    tasks:                    Dict[str, bool]      # atom name -> enabled, full menu
    overrides:                Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def enabled_tasks(self) -> List[str]:
        """Enabled atom names, in the order they're listed in pipeline.yaml."""
        return [name for name, enabled in self.tasks.items() if enabled]


@dataclass
class PipelineConfig:
    """All pipeline-level values for one before_imaging pipeline."""
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

    # ── Fluidics (notebook 04) ───────────────────────────────────────────
    use_adaptors:           bool
    include_final_cleave:   bool
    first_hyb_no_cleave:    bool

    # ── MERlin-specific (notebooks 03/05/06/07) ──────────────────────────
    merlin: MerlinConfig

    source_path: Path = field(repr=False, default=None)


def _load_round_bit_color(yaml_path: Path, csv_relpath: str) -> List[tuple]:
    csv_path = yaml_path.parent / csv_relpath
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


def _load_power(yaml_path: Path, microscope: str) -> tuple[Dict[int, float], float]:
    """Per-channel laser power for `microscope`, from data/configs/power/
    power_by_microscope.yaml (MERCI_DIR/data/, resolved relative to
    yaml_path == MERCI_DIR/data/pipelines/<id>/pipeline.yaml)."""
    data_dir = yaml_path.parents[2]   # .../data/pipelines/<id>/pipeline.yaml -> .../data
    power_path = data_dir / "configs" / "power" / "power_by_microscope.yaml"
    if not power_path.exists():
        raise FileNotFoundError(f"{power_path} not found.")
    table = yaml.safe_load(power_path.read_text())
    if microscope not in table:
        raise KeyError(f"No power defaults for microscope {microscope!r} in {power_path}.")
    entry = table[microscope]
    return {int(k): float(v) for k, v in entry["power"].items()}, float(entry["power_default"])


def load_pipeline_config(yaml_path: Path) -> PipelineConfig:
    """
    Load one pipeline's ``pipeline.yaml`` (+ its round_bit_color CSV and the
    shared per-microscope power table, both resolved relative to
    ``MERCI_DIR/data/``) into a ``PipelineConfig``.

    Raises ``KeyError``/``FileNotFoundError`` loudly on anything missing --
    no silent defaults for pipeline-defining values.
    """
    yaml_path = Path(yaml_path)
    raw = yaml.safe_load(yaml_path.read_text())

    imaging  = raw["imaging"]
    fluidics = raw["fluidics"]
    merlin   = raw["merlin"]

    def _recipe(d: dict) -> ImagingRoundRecipe:
        return ImagingRoundRecipe(**d)

    power, power_default = _load_power(yaml_path, raw["microscope"])

    merlin_config = MerlinConfig(
        project                = merlin["project"],
        lib_name                = merlin["lib_name"],
        round_bit_color          = _load_round_bit_color(yaml_path, merlin["dataorganization"]["round_bit_color_csv"]),
        n_optimize_iterations     = merlin["analysis"]["n_optimize_iterations"],
        tasks                      = dict(merlin["analysis"]["tasks"]),
        overrides                   = dict(merlin["analysis"].get("overrides", {})),
    )

    return PipelineConfig(
        id                = raw["id"],
        label             = raw["label"],
        analysis_backend  = raw["analysis_backend"],
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

        use_adaptors           = fluidics["use_adaptors"],
        include_final_cleave   = fluidics["include_final_cleave"],
        first_hyb_no_cleave    = fluidics["first_hyb_no_cleave"],

        merlin = merlin_config,

        source_path = yaml_path,
    )
