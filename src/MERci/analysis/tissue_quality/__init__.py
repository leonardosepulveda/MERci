"""Tissue-quality model from the BC561 analysis: P(damaged) of 103 µm DAPI
windows, the damaged % of a FOV, and a 51 µm map of damaged tissue.

Trained on ST2 60x DAPI only (BC553disk, 5 planes z 3-23 µm). Another
microscope needs a ``binning`` giving ~0.357 µm/px; results there are
unvalidated. The haze of an acquisition (median signal of its empty FOVs)
must be measured: DAPI between nuclei is the main sign of damage, so an
underestimated haze reads as damage.

Typical use::

    levels = [empty_fov_level(img, dark) for img, dark in all_fovs]
    haze = haze_level(levels)
    result = score_fov(img, dark, haze)          # result.damaged_pct, .squares, ...
    # nucleus flags: flag_nuclei(<every FOV's squares, with a fov column>, nuclei)
"""
from .features_v2 import empty_fov_level, haze_level
from .norm_features import UM_PX
from .scoring import FovScore, flag_nuclei, load_model, score_fov, SQUARE_UM, WINDOW_UM

__all__ = ["empty_fov_level", "haze_level", "load_model", "score_fov", "flag_nuclei", "FovScore",
           "UM_PX", "SQUARE_UM", "WINDOW_UM"]
