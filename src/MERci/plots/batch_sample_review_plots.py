# MERci/plots/batch_sample_review_plots.py
"""Plotting for ``notebooks/after_imaging/05_batch_sample_review.ipynb`` (see
:mod:`MERci.analysis.batch_sample_review` for the backfill/loading side)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_round_comparison(all_stats: pd.DataFrame, round_id: int, metric: str = "median") -> None:
    """
    One violin per sample, comparing *metric* (a column of ``all_stats``,
    e.g. ``"median"`` or ``"max"`` as a saturation proxy) for one round
    across every sample in the batch.
    """
    data = all_stats[all_stats["round_id"] == round_id]
    if data.empty:
        print(f"No data for round {round_id}.")
        return

    sample_names = sorted(data["sample_name"].unique())
    fig, ax = plt.subplots(figsize=(1.4 * len(sample_names) + 2, 4))
    values = [data.loc[data["sample_name"] == s, metric].dropna().values for s in sample_names]
    ax.violinplot(values, showmedians=True)
    ax.set_xticks(range(1, len(sample_names) + 1))
    ax.set_xticklabels(sample_names, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Round {round_id} — {metric} per frame, by sample")
    fig.tight_layout()
    plt.show()
