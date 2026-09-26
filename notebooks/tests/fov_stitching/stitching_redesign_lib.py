"""Shared code for the stitching_redesign_step_NN notebooks.

The redesigned stitching method, in four parts:
  1. per-edge overlap registration (reused from MERci.acquisition.camera_rotation),
  2. per-direction outlier rejection,
  3. one affine A fitted on edge DISPLACEMENTS (m_ij = A d_ij),
  4. per-FOV least squares on all kept edges, with a weak A d_ij prior on
     every grid edge.
Plus the prior method (absolute-coordinate affine) for comparison, and the
3x3 / whole-grid plotting helpers the notebooks share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------- measure
def crop_pair(A: np.ndarray, B: np.ndarray, direction: str, n: int):
    """The n-px-wide bands of anchor A and neighbour B that face each other.
    Image convention: column = +x, row = +y (frames already oriented)."""
    if direction == "right":
        return A[:, -n:], B[:, :n]
    if direction == "left":
        return A[:, :n], B[:, -n:]
    if direction == "up":
        return A[-n:, :], B[:n, :]
    if direction == "down":
        return A[:n, :], B[-n:, :]
    raise ValueError(direction)


def measure_edge(A: np.ndarray, B: np.ndarray, direction: str, n: int,
                 pixel_um: float, upsample: int = 10):
    """Displacement (um) of neighbour B's origin from anchor A's origin,
    from phase correlation of their facing bands.

    The displacement is computed from the crop geometry itself
    (frame size - n + shift), NOT as nominal + shift. The prior code
    (camera_rotation.register_neighbor_pair) used nominal + shift, which is
    only right when n equals the nominal overlap exactly; any mismatch in n
    became a constant along-axis bias of (nominal overlap - n) px.

    Returns (dx_um, dy_um, pearson) -- pearson is the band correlation at
    the found integer shift, a confidence score (skimage's own `error`
    saturates at 1.0 on this data)."""
    from skimage.registration import phase_cross_correlation
    from MERci.acquisition.alignment import remove_hot_pixels
    a, b = crop_pair(A, B, direction, n)
    a = remove_hot_pixels(a).astype(float); b = remove_hot_pixels(b).astype(float)
    (sy, sx), *_ = phase_cross_correlation(a, b, upsample_factor=upsample)
    H, W = A.shape
    if direction == "right":
        ox, oy = W - n + sx, sy
    elif direction == "left":
        ox, oy = sx - (W - n), sy
    elif direction == "up":
        ox, oy = sx, H - n + sy
    else:
        ox, oy = sx, sy - (H - n)
    return ox * pixel_um, oy * pixel_um, band_correlation(A, B, int(round(ox)), int(round(oy)))


def band_correlation(A: np.ndarray, B: np.ndarray, ox: int, oy: int) -> float:
    """Pearson correlation of A and B over their overlap when B's pixel
    (r, c) sits on A's pixel (r + oy, c + ox)."""
    H, W = A.shape
    r0, r1 = max(0, oy), min(H, H + oy); c0, c1 = max(0, ox), min(W, W + ox)
    if r1 - r0 < 10 or c1 - c0 < 10:
        return 0.0
    a = A[r0:r1, c0:c1].astype(float).ravel(); b = B[r0 - oy:r1 - oy, c0 - ox:c1 - ox].astype(float).ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- edges
class Edges:
    """Arrays for every measured edge i -> j.

    d : nominal displacement  nominal[j] - nominal[i]  (um)
    m : measured displacement measured_xy - nominal[i] (um)
    key : physical-edge id (min(i, j), max(i, j)) -- an edge measured from
          both ends appears twice with the same key.
    """

    def __init__(self, corr: pd.DataFrame, nominal: np.ndarray):
        self.df = corr.reset_index(drop=True)
        self.i = self.df.anchor_fov.to_numpy()
        self.j = self.df.neighbor_fov.to_numpy()
        self.direction = self.df.direction.to_numpy()
        self.d = nominal[self.j] - nominal[self.i]
        self.m = self.df[["measured_x", "measured_y"]].to_numpy() - nominal[self.i]
        self.r = self.m - self.d
        self.key = np.array([f"{min(a, b)}-{max(a, b)}" for a, b in zip(self.i, self.j)])


def reject_outliers(edges: Edges, n_mad: float = 5.0) -> np.ndarray:
    """Per direction: drop an edge whose 2-D residual r = m - d lies more
    than median + n_mad * 1.4826 * MAD from that direction's median residual.
    Returns a boolean keep mask."""
    keep = np.ones(len(edges.i), bool)
    for dname in np.unique(edges.direction):
        idx = np.where(edges.direction == dname)[0]
        med = np.median(edges.r[idx], axis=0)
        dev = np.hypot(*(edges.r[idx] - med).T)
        thr = np.median(dev) + n_mad * 1.4826 * np.median(np.abs(dev - np.median(dev)))
        keep[idx[dev > thr]] = False
    return keep


# ---------------------------------------------------------------- models
def fit_displacement_affine(edges: Edges, idx) -> np.ndarray:
    """Redesign: least squares m = A d over edges idx. Returns 2x2 A.
    No translation term: a displacement has none."""
    P, *_ = np.linalg.lstsq(edges.d[idx], edges.m[idx], rcond=None)
    return P.T


def fit_old_affine(edges: Edges, idx, nominal: np.ndarray) -> np.ndarray:
    """Prior method (camera_rotation.fit_camera_rotation): least squares
    measured_xy = M nominal[j] + t on ABSOLUTE coordinates, t then dropped.
    Returns the 2x2 linear part M."""
    X = np.hstack([nominal[edges.j[idx]], np.ones((len(idx), 1))])
    Y = edges.m[idx] + nominal[edges.i[idx]]
    P, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return P[:2].T


def rotation_and_scale(A: np.ndarray):
    """Polar split A = R S: rotation angle of R (deg), singular values of A."""
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), S


def grid_edges(nominal: np.ndarray, step_um: float, tol: float = 0.25):
    """Every 4-connected grid edge (a < b) of the nominal grid."""
    from scipy.spatial import cKDTree
    pairs = cKDTree(nominal).query_pairs(step_um * (1 + tol), output_type="ndarray")
    dist = np.hypot(*(nominal[pairs[:, 1]] - nominal[pairs[:, 0]]).T)
    pairs = pairs[np.abs(dist - step_um) < tol * step_um]
    return pairs[:, 0], pairs[:, 1]


def solve_positions(edges: Edges, idx, nominal: np.ndarray, A_prior: np.ndarray,
                    lam: float, step_um: float, pin_fov: int = 0,
                    pin_weight: float = 1e6) -> np.ndarray:
    """Least-squares FOV positions P (N x 2):

        min  sum_{e in idx} |P_j - P_i - m_e|^2
           + lam * sum_{grid edges} |P_j - P_i - A_prior d_ij|^2
           + pin_weight * |P_pin - nominal_pin|^2

    Solved directly (sparse LU), per axis: the normal matrix is a weighted
    graph Laplacian, so there is no iterative-solver tolerance to tune.
    """
    N = len(nominal)
    I = [edges.i[idx]]; J = [edges.j[idx]]; T = [edges.m[idx]]; W = [np.ones(len(idx))]
    if lam > 0:
        gi, gj = grid_edges(nominal, step_um)
        I.append(gi); J.append(gj)
        T.append((A_prior @ (nominal[gj] - nominal[gi]).T).T)
        W.append(np.full(len(gi), lam))
    I = np.concatenate(I); J = np.concatenate(J); T = np.vstack(T); W = np.concatenate(W)
    k = len(I)
    B = sp.coo_matrix((np.r_[-np.ones(k), np.ones(k)],
                       (np.r_[np.arange(k), np.arange(k)], np.r_[I, J])), shape=(k, N)).tocsr()
    L = (B.T @ sp.diags(W) @ B).tolil()
    L[pin_fov, pin_fov] += pin_weight
    L = L.tocsc()
    P = np.zeros((N, 2))
    for ax in range(2):
        rhs = B.T @ (W * T[:, ax])
        rhs[pin_fov] += pin_weight * nominal[pin_fov, ax]
        P[:, ax] = spla.spsolve(L, rhs)
    return P


def apply_affine(A: np.ndarray, nominal: np.ndarray, pin_fov: int = 0) -> np.ndarray:
    """Positions from one affine, fixed at pin_fov: P = p0 + A (nominal - p0)."""
    p0 = nominal[pin_fov]
    return p0 + (A @ (nominal - p0).T).T


def edge_error(P: np.ndarray, edges: Edges, idx) -> np.ndarray:
    """|P_j - P_i - m_e| (um) for edges idx: how far positions P leave each
    measured overlap from lining up."""
    return np.hypot(*((P[edges.j[idx]] - P[edges.i[idx]]) - edges.m[idx]).T)


def edge_folds(edges: Edges, keep: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per edge, grouped by physical edge (-1 for rejected edges)."""
    ukeys = np.unique(edges.key[keep])
    f = np.random.default_rng(seed).integers(0, n_folds, len(ukeys))
    lookup = dict(zip(ukeys, f))
    return np.array([lookup[k] if kp else -1 for k, kp in zip(edges.key, keep)])


# ---------------------------------------------------------------- 3x3 rendering
def neighbourhood_ids(center: int, nominal: np.ndarray, step_um: float):
    """3x3 FOV ids around center as a (3, 3) array, row 0 = +y (top)."""
    out = np.full((3, 3), -1)
    for r, dy in enumerate((1, 0, -1)):
        for c, dx in enumerate((-1, 0, 1)):
            target = nominal[center] + np.array([dx, dy]) * step_um
            dist = np.hypot(*(nominal - target).T)
            if dist.min() < 0.25 * step_um:
                out[r, c] = int(np.argmin(dist))
    return out


def render_tiles(frames: dict, P: dict, pixel_um: float, bounds=None, clip=(1, 99.8)):
    """Place each oriented frame at its position (lower-left corner, um;
    column = +x, row = +y) on one canvas, coloured magenta or green in a
    checkerboard by grid parity, so aligned overlaps turn white and
    misaligned ones show as separate magenta and green copies.

    frames : {fov: 2-D array}; P : {fov: (x, y, parity)}.
    bounds : (x0, y0, x1, y1) um region to render (default: all tiles).
    Returns (rgb image with row 0 = lowest y, extent for imshow origin='lower').
    """
    h, w = next(iter(frames.values())).shape
    if bounds is None:
        xs = [P[f][0] for f in frames]; ys = [P[f][1] for f in frames]
        bounds = (min(xs), min(ys), max(xs) + w * pixel_um, max(ys) + h * pixel_um)
    x0, y0, x1, y1 = bounds
    W = int(round((x1 - x0) / pixel_um)); H = int(round((y1 - y0) / pixel_um))
    acc = np.zeros((H, W, 2))
    for f, img in frames.items():
        lo, hi = np.percentile(img, clip)
        norm = np.clip((img.astype(float) - lo) / max(hi - lo, 1e-9), 0, 1)
        c0 = int(round((P[f][0] - x0) / pixel_um)); r0 = int(round((P[f][1] - y0) / pixel_um))
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + h, H), min(c0 + w, W)
        if re <= rs or ce <= cs:
            continue
        ch = int(P[f][2])
        acc[rs:re, cs:ce, ch] = np.maximum(acc[rs:re, cs:ce, ch], norm[rs - r0:re - r0, cs - c0:ce - c0])
    rgb = np.stack([acc[..., 0], acc[..., 1], acc[..., 0]], axis=-1)   # magenta / green
    return rgb, (x0, x1, y0, y1)
