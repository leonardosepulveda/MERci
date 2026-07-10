#!/usr/bin/env python
"""
plot_drift.py -- visualize detect-spots drift across all rounds and FOVs.

Reads the per-FOV drift records written by detect-spots and makes one 1x3 figure:
  (1) scatter of x_drift vs y_drift (one point per fov/round, colored by round),
      on square symmetric axes (+/- scatter_max) so outliers don't set the scale.
  (2) histogram of drift distance over [0, dmax] with fixed bin width, median marked.
  (3) heatmap of (drift distance - median) with x = fov, y = round, diverging cmap.

scatter_max and dmax each accept a number or "auto" (a robust, outlier-resistant
limit = the auto_pct-th percentile of |values|, rounded up to a clean number).

Source preference:
  * drift_*.csv  (patched detect-spots: sub-pixel x_drift/y_drift)
  * else channels_*.csv (always written: integer x_drift/y_drift), deduped per (fov, series).

Run on the cluster in fishtank_env, e.g.:
    python plot_drift.py --input ../output/spots
    python plot_drift.py --input ../output/spots --scatter_max auto --dmax auto --bin_size 5
"""
import argparse
import math
import pathlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _robust_max(vals, pct=99, floor=10.0):
    """Outlier-resistant upper bound: the pct-th percentile of |vals|, rounded up to a
    clean 1/2/5 x 10^k number. Used for 'auto' axis limits."""
    v = np.asarray(vals, dtype=float)
    v = np.abs(v[np.isfinite(v)])
    if v.size == 0:
        return floor
    hi = float(np.percentile(v, pct))
    if hi <= 0:
        return floor
    mag = 10 ** math.floor(math.log10(hi))
    for m in (1, 2, 5, 10):
        if hi <= m * mag:
            return float(m * mag)
    return float(10 * mag)


def make_drift_figure(drift: pd.DataFrame, out=None, title=None, dpi=150,
                      scatter_max=100, dmax=100, bin_size=5, auto_pct=99,
                      cmap="coolwarm", fontsize=12):
    """drift: DataFrame with columns fov, series, x_drift, y_drift [, distance].

    scatter_max / dmax: a number, or "auto" for a robust outlier-resistant limit."""
    df = drift.copy()
    if "distance" not in df.columns:
        df["distance"] = np.hypot(df["x_drift"], df["y_drift"])
    rounds = list(dict.fromkeys(df["series"]))
    df["round"] = df["series"].map({s: i for i, s in enumerate(rounds)})
    n_round = len(rounds)
    median = float(df["distance"].median())

    # Resolve "auto" limits robustly
    smax = (_robust_max(np.concatenate([df["x_drift"].values, df["y_drift"].values]), auto_pct)
            if str(scatter_max).lower() == "auto" else float(scatter_max))
    dlim = (_robust_max(df["distance"].values, auto_pct)
            if str(dmax).lower() == "auto" else float(dmax))

    # Bump default font sizes (matplotlib base is 10; default here is 12 = +2).
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "figure.titlesize": fontsize + 3,
    })
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    if title:
        fig.suptitle(title)

    # (1) x vs y drift scatter, square symmetric axes +/- smax
    n_off = int(((df["x_drift"].abs() > smax) | (df["y_drift"].abs() > smax)).sum())
    sc = ax[0].scatter(df["x_drift"], df["y_drift"], c=df["round"], cmap="viridis",
                       s=14, alpha=0.7, edgecolors="none")
    ax[0].axhline(0, color="k", lw=0.6, ls=":")
    ax[0].axvline(0, color="k", lw=0.6, ls=":")
    ax[0].set_xlim(-smax, smax)
    ax[0].set_ylim(-smax, smax)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x drift (px)")
    ax[0].set_ylabel("y drift (px)")
    off = f"  ({n_off} outliers)" if n_off else ""
    ax[0].set_title(f"(1) drift vectors{off}")
    fig.colorbar(sc, ax=ax[0], label="round", fraction=0.046, pad=0.04)

    # (2) histogram of drift distance over [0, dlim], fixed bin width, median marked
    bins = np.arange(0, dlim + bin_size, bin_size)
    in_range = df["distance"][(df["distance"] >= 0) & (df["distance"] <= dlim)]
    n_over = int((df["distance"] > dlim).sum())
    ax[1].hist(in_range, bins=bins, color="slategray", edgecolor="white", lw=0.4)
    ax[1].axvline(median, color="crimson", lw=1.6, ls="--", label=f"median = {median:.2f}")
    ax[1].set_xlim(0, dlim)
    ax[1].set_xlabel("drift distance (px)")
    ax[1].set_ylabel("count (fov x round)")
    over = f"  ({n_over} outliers)" if n_over else ""
    ax[1].set_title(f"(2) drift distance{over}")
    ax[1].legend(loc="best")

    # (3) heatmap of (distance - median): x = fov, y = round, diverging cmap
    piv = df.pivot_table(index="round", columns="fov", values="distance", aggfunc="mean")
    piv = piv.reindex(sorted(piv.columns), axis=1).sort_index()
    dev = piv.values - median
    vlim = _robust_max(dev, max(auto_pct, 98), floor=1.0)  # symmetric, robust to outliers
    im = ax[2].imshow(dev, aspect="auto", origin="lower", cmap=cmap,
                      vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax[2].set_xlabel("fov")
    ax[2].set_ylabel("round")
    ax[2].set_title(f"(3) drift distance − median ({median:.1f} px)")
    fovvals = piv.columns.values
    xidx = np.linspace(0, len(fovvals) - 1, min(len(fovvals), 12)).astype(int)
    ax[2].set_xticks(xidx)
    ax[2].set_xticklabels(fovvals[xidx])
    ax[2].set_yticks(np.linspace(0, n_round - 1, min(n_round, 14)).astype(int))
    fig.colorbar(im, ax=ax[2], label="distance − median (px)", fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    if out is not None:
        fig.savefig(out, dpi=dpi)
    return fig


def _read_concat(files, required):
    """Concatenate CSVs, skipping empty / unparseable / missing-column files.

    Tolerates partial or failed runs: only the FOVs that actually produced a valid
    file are loaded. Returns (DataFrame|None, n_loaded, n_skipped)."""
    frames, skipped = [], []
    for f in files:
        try:
            t = pd.read_csv(f)
        except Exception:  # noqa: BLE001  (empty / truncated / corrupt file)
            skipped.append(f.name); continue
        if t.empty or not required.issubset(t.columns):
            skipped.append(f.name); continue
        frames.append(t)
    if skipped:
        head = ", ".join(sorted(skipped)[:5]) + (" ..." if len(skipped) > 5 else "")
        print(f"  WARNING: skipped {len(skipped)} empty/invalid file(s): {head}")
    df = pd.concat(frames, ignore_index=True) if frames else None
    return df, len(frames), len(skipped)


def load_drift(input_dir):
    required = {"fov", "series", "x_drift", "y_drift"}
    d = sorted(pathlib.Path(input_dir).glob("drift_*.csv"))
    if d:
        df, n, _ = _read_concat(d, required)
        if df is not None:
            return df, f"{n}/{len(d)} drift_*.csv (sub-pixel)"
    c = sorted(pathlib.Path(input_dir).glob("channels_*.csv"))
    if c:
        df, n, _ = _read_concat(c, required)
        if df is not None:
            df = (df.dropna(subset=["x_drift", "y_drift"])
                    .drop_duplicates(["fov", "series"])[["fov", "series", "x_drift", "y_drift"]])
            return df, f"{n}/{len(c)} channels_*.csv (integer drift)"
    raise FileNotFoundError(f"No valid drift_*.csv or channels_*.csv found in {input_dir}")


def _limit(v):
    """argparse type: accept 'auto' or a float."""
    return "auto" if str(v).lower() == "auto" else float(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="detect-spots output dir (has drift_*.csv / channels_*.csv)")
    p.add_argument("-o", "--out", default=None,
                   help="Output figure path (relative or absolute). Default: <input>/drift_qc.png")
    p.add_argument("--title", default="detect-spots drift QC")
    p.add_argument("--scatter_max", type=_limit, default=100.0,
                   help="Panel 1 axis half-range (px), symmetric. Number or 'auto'. Default 100.")
    p.add_argument("--dmax", type=_limit, default=100.0,
                   help="Panel 2 histogram upper bound (px). Number or 'auto'. Default 100.")
    p.add_argument("--bin_size", type=float, default=5.0, help="Panel 2 histogram bin width (px). Default 5.")
    p.add_argument("--auto_pct", type=float, default=99.0,
                   help="Percentile used for 'auto' limits (outlier-resistant). Default 99.")
    p.add_argument("--cmap", default="coolwarm", help="Diverging colormap for panel 3 (blue-white-red). Default coolwarm.")
    p.add_argument("--fontsize", type=float, default=12.0,
                   help="Base font size (matplotlib default is 10; 12 = +2). Default 12.")
    args = p.parse_args()

    out = args.out if args.out is not None else str(pathlib.Path(args.input) / "drift_qc.png")
    df, src = load_drift(args.input)
    n_fov = df["fov"].nunique()
    n_round = df["series"].nunique()
    print(f"loaded {len(df)} drift records from {src}: {n_fov} fovs x {n_round} rounds")
    dist = np.hypot(df["x_drift"], df["y_drift"])
    print(f"drift distance: median={dist.median():.2f} px, max={dist.max():.2f} px, "
          f"zero-drift records={(dist == 0).sum()}")
    make_drift_figure(df, out=out, title=f"{args.title}  ({src})",
                      scatter_max=args.scatter_max, dmax=args.dmax, bin_size=args.bin_size,
                      auto_pct=args.auto_pct, cmap=args.cmap, fontsize=args.fontsize)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
