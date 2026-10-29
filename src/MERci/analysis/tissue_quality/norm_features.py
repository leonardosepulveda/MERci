"""Intensity-normalized image features of one single-plane 60x DAPI FOV image.

BC561 G2 01 features measured log-DAPI in BC553's own units (fixed tissue
threshold, absolute log percentiles). Here everything is relative to the FOV
itself, so the features do not change when the image is scaled (other gain,
exposure, laser power, microscope) or shifted by a camera offset:

- J = image - dark level (median of a dark frame of the same camera).
- The image is binned to 0.357 µm/px (4x4 at the ST2 60x 0.0893 µm/px).
- L = log1p(J / (C_REL * p95(J))): log brightness relative to the FOV's
  bright tissue level.
- Tissue pixels: L above the FOV's Otsu threshold, and J at least R_BG
  times the median J of the pixels below it. An empty FOV has only a smooth
  haze (BC553disk: Otsu classes differ by 1.3x, against 4-18x with nuclei),
  so it gets no tissue.

Features are computed on 288 px (103 µm) windows in 144 px steps, the BC561 G2/G3
window, with the same names as G2 01's image features.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import threshold_otsu

UM_PX = 0.35720            # binned pixel, µm
WIN, STEP = 288, 144       # window and step, binned px
C_REL = 0.02
R_BG = 2.0
MIN_FG = 0.2               # minimum tissue fraction of a window
SIG_UM = [0.5, 1, 2, 4, 8]
RHO_UM = [2, 5, 10]
BANDS_UM = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]
_fr = np.fft.fftfreq(WIN, d=UM_PX)
_FR = np.hypot(*np.meshgrid(_fr, _fr))
_HANN = np.outer(np.hanning(WIN), np.hanning(WIN))


def dark_level(dark):
    """Camera offset: median of a dark frame."""
    return float(np.median(dark))


def normalize(img, dark, binning):
    """Binned, offset-subtracted image J, relative log image L and tissue mask."""
    a = np.asarray(img, np.float32) - dark
    if binning > 1:
        h, w = (s // binning * binning for s in a.shape)
        a = a[:h, :w].reshape(h // binning, binning, w // binning, binning).mean((1, 3))
    J = np.clip(a, 0, None)
    L = np.log1p(J / (C_REL * max(np.percentile(J, 95), 1e-6)))
    try:
        thr = threshold_otsu(L)
    except ValueError:             # flat image
        return J, L, np.zeros(L.shape, bool)
    return J, L, (L > thr) & (J >= R_BG * np.median(J[L <= thr]))


def window_features(L, fg):
    """Features of one WIN x WIN window (L and its tissue mask)."""
    f = {'fg_frac': fg.mean()}
    v = L[fg]
    f.update({f'log_p{q}': np.percentile(v, q) for q in (10, 50, 90)})
    f['log_std'] = v.std()
    n = L - L.mean()
    for s in SIG_UM:
        f[f'grad_{s}um'] = ndi.gaussian_gradient_magnitude(n, s / UM_PX)[fg].mean()
        f[f'log_{s}um'] = np.abs(ndi.gaussian_laplace(n, s / UM_PX))[fg].mean() * (s / UM_PX) ** 2
    P = np.abs(np.fft.fft2(n * _HANN)) ** 2
    tot = P[_FR > 0].sum()
    for a, b in BANDS_UM:
        f[f'spec_{a}_{b}um'] = P[(_FR > 1 / b) & (_FR <= 1 / a)].sum() / tot
    for r in RHO_UM:
        A = structure_tensor(n, sigma=r / UM_PX, order='rc')
        l1, l2 = structure_tensor_eigenvalues(A)
        coh = ((l1 - l2) / (l1 + l2 + 1e-12)) ** 2
        f[f'coh_{r}um'] = coh[fg].mean()
        a, b2, c = (x[fg].mean() for x in A)
        E = np.linalg.eigvalsh([[a, b2], [b2, c]])
        f[f'align_{r}um'] = ((E[1] - E[0]) / (E[1] + E[0] + 1e-12)) ** 2
    return f


def fov_features(img, dark, binning=4):
    """Rows {'wy', 'wx', 'fg_frac', ...} of the windows of one FOV image with
    enough tissue; (wy, wx) is the window's top-left corner in binned px."""
    _, L, fg = normalize(img, dark, binning)
    rows = []
    for wy in range(0, L.shape[0] - WIN + 1, STEP):
        for wx in range(0, L.shape[1] - WIN + 1, STEP):
            m = fg[wy:wy + WIN, wx:wx + WIN]
            if m.mean() >= MIN_FG:
                rows.append({'wy': wy, 'wx': wx, **window_features(L[wy:wy + WIN, wx:wx + WIN], m)})
    return rows
