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

See ``prompt_history/2026_08_28_1655_list_pipeline_yaml_variables.md`` and
``prompt_history/2026_08_28_1712_pipeline_yaml_implementation_plan.md`` for
how this variable list was derived and why some things (round_bit_color,
the decoding-strategy table) are pipeline-level while others
(sequential_genes.csv, GRID_METHOD) deliberately are not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
class PipelineConfig:
    """All pipeline-level values for one before_imaging pipeline."""
    id:                str
    label:             str
    analysis_backend:  str   # "merlin" | "fishtank"
    microscope:        str
    objective:         str

    # ── Imaging (notebook 01) ────────────────────────────────────────────
    file_type:        str
    exposure_time:    float
    power:            Dict[int, float]
    power_default:    float
    cells_round:      ImagingRoundRecipe
    bits_round:       ImagingRoundRecipe
    focus_test_z:     float
    transit_n_blank:  int

    # ── Fluidics (notebook 04) ───────────────────────────────────────────
    use_adaptors:           bool
    include_final_cleave:   bool
    first_hyb_no_cleave:    bool

    # ── Round/bit/color recipe (notebook 03) ─────────────────────────────
    round_bit_color:  List[tuple]   # [(round, bit, color), ...]

    # ── Analysis (notebook 07) ───────────────────────────────────────────
    include_segmentation:      bool
    sum_signal_channel_names:  List[str]

    # ── Metadata (notebook 06) ───────────────────────────────────────────
    project:   str
    lib_name:  str
    n_opt:     int

    source_path: Path = field(repr=False, default=None)


def _load_round_bit_color(yaml_path: Path, csv_name: str) -> List[tuple]:
    csv_path = yaml_path.parent / csv_name
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


def load_pipeline_config(yaml_path: Path) -> PipelineConfig:
    """
    Load one pipeline's ``pipeline.yaml`` (+ its round_bit_color CSV,
    resolved relative to the YAML's own directory) into a ``PipelineConfig``.

    Raises ``KeyError``/``FileNotFoundError`` loudly on anything missing --
    no silent defaults for pipeline-defining values.
    """
    yaml_path = Path(yaml_path)
    raw = yaml.safe_load(yaml_path.read_text())

    imaging  = raw["imaging"]
    fluidics = raw["fluidics"]
    analysis = raw["analysis"]
    metadata = raw["metadata"]

    def _recipe(d: dict) -> ImagingRoundRecipe:
        return ImagingRoundRecipe(**d)

    return PipelineConfig(
        id                = raw["id"],
        label             = raw["label"],
        analysis_backend  = raw["analysis_backend"],
        microscope        = raw["microscope"],
        objective         = raw["objective"],

        file_type        = imaging["file_type"],
        exposure_time    = imaging["exposure_time"],
        power            = {int(k): float(v) for k, v in imaging["power"].items()},
        power_default    = imaging["power_default"],
        cells_round      = _recipe(imaging["cells_round"]),
        bits_round       = _recipe(imaging["bits_round"]),
        focus_test_z     = imaging["focus_test_z"],
        transit_n_blank  = imaging["transit_n_blank"],

        use_adaptors           = fluidics["use_adaptors"],
        include_final_cleave   = fluidics["include_final_cleave"],
        first_hyb_no_cleave    = fluidics["first_hyb_no_cleave"],

        round_bit_color  = _load_round_bit_color(yaml_path, raw["round_bit_color_csv"]),

        include_segmentation      = analysis["include_segmentation"],
        sum_signal_channel_names  = analysis["sum_signal_channel_names"],

        project   = metadata["project"],
        lib_name  = metadata["lib_name"],
        n_opt     = metadata["n_opt"],

        source_path = yaml_path,
    )
