"""Window features v2: BC561 G4 01's normalized features (norm_features.py), plus

- **haze subtraction**: the optical haze of the acquisition (measured off
  tissue, `haze_level`) is subtracted together with the dark level before
  the features (the tissue mask stays that of the dark-subtracted image:
  its floor relative to the background needs the haze). Real
  DAPI between nuclei, the user's sign of damage, is kept: it is only on
  tissue, while haze is also off tissue.
- **gap features**: tissue area = the tissue (nucleus) mask closed with a
  GAP_CLOSE_UM disk, so the cytoplasm between nearby nuclei is inside it;
  gap = area minus nuclei. `gap_ratio` = median J in the gap / median J in
  the nuclei (dark cytoplasm in ok tissue, DAPI everywhere in damaged
  tissue), `gap_frac` = gap share of the area, `gap_p10_p90` = p10 / p90 of
  J in the area.
"""
import numpy as np
from scipy import ndimage as ndi

from .norm_features import UM_PX, WIN, STEP, MIN_FG, normalize, window_features

GAP_CLOSE_UM = 3.0
EMPTY_FG = 0.01            # a FOV with less tissue than this is empty (for the haze)
_r = int(round(GAP_CLOSE_UM / UM_PX))
_DISK = np.hypot(*np.mgrid[-_r:_r + 1, -_r:_r + 1]) <= _r


def empty_fov_level(img, dark, binning=4):
    """Median dark-subtracted binned signal of a FOV if it is empty, else None."""
    J, _, fg = normalize(img, dark, binning)
    return float(np.median(J)) if fg.mean() < EMPTY_FG else None


def haze_level(levels):
    """Haze of one acquisition: median of its empty FOVs' levels (0 if none)."""
    v = [x for x in levels if x is not None]
    return float(np.median(v)) if v else 0.0


def tissue_area(fg):
    return ndi.binary_closing(np.pad(fg, _r), _DISK)[_r:-_r, _r:-_r] | fg


def gap_features(J, fg, area):
    gap = area & ~fg
    nuc = np.median(J[fg])
    return {'gap_ratio': np.median(J[gap]) / nuc if gap.any() and nuc > 0 else np.nan,
            'gap_frac': gap.sum() / area.sum(),
            'gap_p10_p90': np.percentile(J[area], 10) / max(np.percentile(J[area], 90), 1e-6)}


def fov_features(img, dark, haze, binning=4):
    """Rows {'wy', 'wx', features...} of the windows of one FOV with enough tissue."""
    fg = normalize(img, dark, binning)[2]
    J, L, _ = normalize(img, dark + haze, binning)
    area = tissue_area(fg)
    rows = []
    for wy in range(0, L.shape[0] - WIN + 1, STEP):
        for wx in range(0, L.shape[1] - WIN + 1, STEP):
            s = np.s_[wy:wy + WIN, wx:wx + WIN]
            if fg[s].mean() >= MIN_FG:
                rows.append({'wy': wy, 'wx': wx, **window_features(L[s], fg[s]), **gap_features(J[s], fg[s], area[s])})
    return rows
