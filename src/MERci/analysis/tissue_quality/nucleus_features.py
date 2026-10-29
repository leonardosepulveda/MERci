"""Nucleus features per window, for the v2 + nuclei model (BC561 G3 03).

Same rule as BC561 G3 03's notebook: nuclei whose centre is inside the window
(x0 <= x < x0 + size, same for y), from any segmentation that gives the
centre in µm and the volume in µm³ (trained on 3D CellPose volumes).
Needs numpy and pandas only.
"""
import numpy as np
import pandas as pd

NUC_FEATURES = ['nuc_n', 'nuc_vol_med', 'nuc_vol_q10', 'nuc_vol_q90', 'nuc_frac_small', 'nuc_frac_large']


def window_nucleus_features(x, y, volume, x0, y0, size):
    """One row per window (x0[k], y0[k], side `size`, all µm); NaN volume
    features where a window has no nucleus (the model fills them)."""
    order = np.argsort(x)
    x, y, volume = np.asarray(x)[order], np.asarray(y)[order], np.asarray(volume)[order]
    rows = []
    for a, b in zip(np.atleast_1d(x0), np.atleast_1d(y0)):
        lo, hi = np.searchsorted(x, [a, a + size])
        v = volume[lo:hi][(y[lo:hi] >= b) & (y[lo:hi] < b + size)
                          & (x[lo:hi] >= a) & (x[lo:hi] < a + size)]
        r = {'nuc_n': len(v)}
        if len(v):
            r.update(nuc_vol_med=np.median(v), nuc_vol_q10=np.quantile(v, 0.1), nuc_vol_q90=np.quantile(v, 0.9),
                     nuc_frac_small=(v < 20).mean(), nuc_frac_large=(v > 1000).mean())
        rows.append(r)
    return pd.DataFrame(rows, columns=NUC_FEATURES)
