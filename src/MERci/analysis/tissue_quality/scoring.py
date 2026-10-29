"""Score one FOV's DAPI with the BC561 tissue-quality models.

Two models, both ``HistGradientBoostingClassifier`` fit on BC561's labelled
BC553disk windows (``data/tissue_quality/training_windows.parquet``): the
image-only v2 model (29 window features) and v2 + nuclei (plus 6 nucleus
features), used when a nucleus table is given. They are fit at first use
and cached in ``data/tissue_quality/cache/`` (gitignored), one file per
scikit-learn version, instead of shipping pickles that may not load under
another version. The refit reproduces BC561's saved models exactly.

Geometry (binned px of 0.357 µm, the training scale): 288 px windows in
144 px steps, scored when at least 20% tissue; 144 px (51 µm) squares,
each scored by the mean P(damaged) of every (window, plane) covering it.
Images must be in MERlin's orientation of the camera
(``configs.apply_microscope_orientation``), the training orientation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import shapely
import sklearn
from shapely.geometry import box
from sklearn.ensemble import HistGradientBoostingClassifier

from .features_v2 import fov_features
from .norm_features import UM_PX, WIN, STEP
from .nucleus_features import NUC_FEATURES, window_nucleus_features

DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "tissue_quality"
V2_FEATURES = [
    "fg_frac", "log_p10", "log_p50", "log_p90", "log_std",
    "grad_0.5um", "log_0.5um", "grad_1um", "log_1um", "grad_2um", "log_2um",
    "grad_4um", "log_4um", "grad_8um", "log_8um",
    "spec_1_2um", "spec_2_4um", "spec_4_8um", "spec_8_16um", "spec_16_32um",
    "coh_2um", "align_2um", "coh_5um", "align_5um", "coh_10um", "align_10um",
    "gap_ratio", "gap_frac", "gap_p10_p90",
]
MODEL_FEATURES = {"image": V2_FEATURES, "nuclei": V2_FEATURES + NUC_FEATURES}
SQUARE_UM = STEP * UM_PX
WINDOW_UM = WIN * UM_PX
DAMAGED_P = 0.5            # a window counts as damaged in the FOV damaged %


def load_model(kind: str = "image") -> dict:
    """``{'model', 'features', 'fill'}`` for *kind* ``"image"`` or
    ``"nuclei"``: fit from the training table on first use, then cached."""
    features = MODEL_FEATURES[kind]
    path = DATA_DIR / "cache" / f"model_{kind}_sklearn{sklearn.__version__}.joblib"
    if path.exists():
        return joblib.load(path)
    df = pd.read_parquet(DATA_DIR / "training_windows.parquet", columns=features + ["label"])
    fill = df[features].median()
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0,
    ).fit(df[features].fillna(fill).values, (df["label"] == "damaged").values)
    fitted = {"model": model, "features": features, "fill": fill.to_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{np.random.randint(1 << 30)}")
    joblib.dump(fitted, tmp)
    tmp.replace(path)
    return fitted


@dataclass
class FovScore:
    """Result of :func:`score_fov`.

    windows     : one row per (plane, window): ``plane, wy, wx, fg_frac,
                  p_damaged`` (wy, wx = window corner, binned px)
    squares     : scored 51 µm squares: ``i, j`` (row, column), ``score``,
                  ``n_windows`` (covering (window, plane) pairs), ``x0, y0``
                  (µm, frame of *fov_origin_um*), ``removed`` (score > line)
    damaged_pct : 100 * tissue-weighted share of windows with P > 0.5
                  (P and tissue fraction averaged over planes); NaN = no tissue
    n_tissue_windows : windows with tissue in at least one plane
    removed     : union of the removed squares (shapely, µm), or None

    Per-nucleus flags come from :func:`flag_nuclei`, over every FOV's squares.
    """
    windows: pd.DataFrame
    squares: pd.DataFrame
    damaged_pct: float
    n_tissue_windows: int
    removed: Optional[shapely.Geometry]


def score_fov(
    images: np.ndarray,
    dark,
    haze,
    fov_origin_um: Tuple[float, float] = (0.0, 0.0),
    nuclei: Optional[pd.DataFrame] = None,
    line: float = 0.1,
    binning: int = 4,
    model: Optional[dict] = None,
) -> FovScore:
    """
    Damage scores of one FOV's DAPI.

    Parameters
    ----------
    images        : one plane (y, x) or several (plane, y, x), in MERlin's
                    orientation; several planes are averaged
    dark          : camera dark level (counts), or a blank frame (its median)
    haze          : the acquisition's haze (:func:`haze_level`), or one per plane
    fov_origin_um : (x, y) of the FOV's pixel (0, 0) in µm, the frame of
                    *nuclei* and of the returned squares
    nuclei        : optional nucleus table: ``center_x, center_y`` (µm) and
                    ``volume`` (µm³, trained on 3D CellPose); selects the
                    v2 + nuclei model. A window counts every nucleus whose
                    centre is inside it, whichever FOV segmented it (as in
                    training), so pass the whole table or a crop with margin,
                    not only this FOV's own nuclei
    line          : squares scoring above it are removed
    binning       : raw px per binned px, so that binned px ~ 0.357 µm
                    (4 at ST2 60x)
    model         : a :func:`load_model` result, to skip reloading it per FOV
    """
    images = np.asarray(images)
    planes = images[None] if images.ndim == 2 else images
    dark = float(np.median(dark)) if np.ndim(dark) else float(dark)
    if model is None:
        model = load_model("image" if nuclei is None else "nuclei")
    x_origin, y_origin = fov_origin_um

    hazes = np.broadcast_to(np.asarray(haze, float), len(planes))
    w = pd.DataFrame([{"plane": k, **r} for k, (im, h) in enumerate(zip(planes, hazes))
                      for r in fov_features(im, dark, h, binning)])
    if len(w):
        if nuclei is not None:
            side = max(planes.shape[1:]) // binning * UM_PX
            near = nuclei[nuclei["center_x"].between(x_origin, x_origin + side)
                          & nuclei["center_y"].between(y_origin, y_origin + side)]
            nf = window_nucleus_features(near["center_x"], near["center_y"], near["volume"],
                                         x_origin + w["wx"].values * UM_PX,
                                         y_origin + w["wy"].values * UM_PX, WINDOW_UM)
            w = pd.concat([w, nf.set_index(w.index)], axis=1)
        X = w[model["features"]].fillna(model["fill"]).values
        w["p_damaged"] = model["model"].predict_proba(X)[:, 1]
    else:
        w = pd.DataFrame(columns=["plane", "wy", "wx", "fg_frac", "p_damaged"])

    cover = pd.concat([w.assign(i=w["wy"] // STEP + di, j=w["wx"] // STEP + dj)
                       for di in (0, 1) for dj in (0, 1)])
    sq = (cover.groupby(["i", "j"])
          .agg(score=("p_damaged", "mean"), n_windows=("p_damaged", "size")).reset_index())
    sq[["i", "j"]] = sq[["i", "j"]].astype(int)
    sq["x0"] = x_origin + sq["j"] * SQUARE_UM
    sq["y0"] = y_origin + sq["i"] * SQUARE_UM
    sq["removed"] = sq["score"] > line

    per_window = w.groupby(["wy", "wx"])[["fg_frac", "p_damaged"]].mean()
    fg = per_window["fg_frac"].astype(float)
    damaged_pct = (100 * float((fg * (per_window["p_damaged"] > DAMAGED_P)).sum() / fg.sum())
                   if len(per_window) else np.nan)

    rem = sq[sq["removed"]]
    removed = (shapely.union_all([box(r.x0, r.y0, r.x0 + SQUARE_UM, r.y0 + SQUARE_UM)
                                  for r in rem.itertuples()]) if len(rem) else None)
    return FovScore(w, sq, damaged_pct, len(per_window), removed)


def flag_nuclei(squares: pd.DataFrame, nuclei: pd.DataFrame, n_squares: int = 4) -> pd.DataFrame:
    """
    *nuclei* plus ``i, j, score, quality`` (good / damaged / no score): each
    nucleus takes the square of its own FOV containing its centre (clipped
    to the FOV's squares).

    squares   : :attr:`FovScore.squares` of several FOVs, with a ``fov`` column
    nuclei    : ``fov`` (the FOV that segmented it), ``center_x, center_y`` (µm)
    n_squares : squares per FOV side (4 at ST2 60x: 2304 px / 4 / 144)
    """
    origin = squares.assign(x=squares["x0"] - squares["j"] * SQUARE_UM,
                            y=squares["y0"] - squares["i"] * SQUARE_UM).groupby("fov")[["x", "y"]].first()
    nuc = nuclei.copy()
    o = origin.reindex(nuc["fov"])
    nuc["i"] = np.clip(np.floor((nuc["center_y"].values - o["y"].values) / SQUARE_UM), 0, n_squares - 1)
    nuc["j"] = np.clip(np.floor((nuc["center_x"].values - o["x"].values) / SQUARE_UM), 0, n_squares - 1)
    nuc[["i", "j"]] = nuc[["i", "j"]].fillna(-1).astype(int)    # FOV without scored squares
    nuc = nuc.join(squares.set_index(["fov", "i", "j"])[["score", "removed"]], on=["fov", "i", "j"])
    nuc["quality"] = np.select([nuc["score"].isna(), nuc["removed"].eq(True)],
                               ["no score", "damaged"], "good")
    return nuc.drop(columns="removed")
