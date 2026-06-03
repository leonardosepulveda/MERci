"""
3D spot/bead localisation and PSF image simulation.

Public API
----------
**DAX I/O helpers**
    read_dax_crop            – row-efficient crop reader for raw DAX files
    compute_max_projection   – streaming z-max projection from a DAX file

**3D Gaussian fitting**
    fit_bead_3d              – curve_fit a 3D Gaussian to a cropped volume

**Detection and localisation**
    detect_beads_2d          – find bead centres in a z-max projection
    localize_beads_in_volume – localise in in-memory colour stacks (core)
    localize_beads_in_file   – localise from a DAX file (wraps I/O helpers)

**Cross-colour matching**
    match_beads_across_colors – KDTree match across colour channels

**Simulation**
    generate_emitter_positions – uniform random emitters in a 3-D volume
    simulate_psf_image         – synthesise one (n_z, H, W) image stack
    add_chromatic_aberration   – shift emitter positions per colour channel
    simulate_multicolor_stack  – multi-colour convenience wrapper

**Visualisation**
    plot_max_projections – XY / ZX / ZY max-projection figure
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage   import gaussian_filter, maximum_filter
from scipy.optimize  import curve_fit
from scipy.spatial   import KDTree

log = logging.getLogger(__name__)


# ── A. DAX I/O helpers ────────────────────────────────────────────────────────

def read_dax_crop(
    dax_path: Union[str, Path],
    frame_indices: List[int],
    H: int,
    W: int,
    r1: int, r2: int,
    c1: int, c2: int,
) -> np.ndarray:
    """Read a spatial crop [r1:r2, c1:c2] from selected frames of a DAX file.

    Only the required rows are loaded from disk — much more memory-efficient
    than reading full frames.

    Returns
    -------
    (n_z, crop_h, crop_w) uint16 array
    """
    n_z    = len(frame_indices)
    crop_h = r2 - r1
    crop_w = c2 - c1
    out    = np.zeros((n_z, crop_h, crop_w), dtype=np.uint16)
    row_bytes = W * 2
    frm_bytes = H * W * 2
    with open(dax_path, "rb") as fh:
        for zi, idx in enumerate(frame_indices):
            frm_off = int(idx) * frm_bytes
            for ri, row in enumerate(range(r1, r2)):
                fh.seek(frm_off + row * row_bytes + c1 * 2)
                out[zi, ri] = np.frombuffer(fh.read(crop_w * 2), dtype=np.uint16)
    return out


def compute_max_projection(
    dax_path: Union[str, Path],
    frame_indices: List[int],
    H: int,
    W: int,
) -> np.ndarray:
    """Stream frames from a DAX file and return the z-max projection (H, W) float32."""
    proj      = np.zeros((H, W), dtype=np.float32)
    frm_bytes = H * W * 2
    with open(dax_path, "rb") as fh:
        for idx in frame_indices:
            fh.seek(int(idx) * frm_bytes)
            frame = np.frombuffer(fh.read(frm_bytes), dtype=np.uint16
                                  ).reshape(H, W).astype(np.float32)
            np.maximum(proj, frame, out=proj)
    return proj


# ── B. 3D Gaussian core ───────────────────────────────────────────────────────

def _gauss3d(
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    amp: float,
    x0: float, y0: float, z0: float,
    sig_xy: float, sig_z: float,
    offset: float,
) -> np.ndarray:
    """Evaluate a symmetric 3D Gaussian on ravelled coordinate arrays."""
    x, y, z = coords
    return (
        amp * np.exp(
            -((x - x0)**2 + (y - y0)**2) / (2 * sig_xy**2)
            - (z - z0)**2               / (2 * sig_z **2)
        ) + offset
    ).ravel()


def fit_bead_3d(
    sub_vol: np.ndarray,
    z_um_vals: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Fit a 3D Gaussian to *sub_vol* and return (x_px, y_px, z_um) or None.

    Parameters
    ----------
    sub_vol   : (n_z, n_y, n_x) float32 — intensity crop
    z_um_vals : (n_z,) float — µm position of each z-plane in the crop

    Returns
    -------
    (x_px, y_px, z_um) relative to the crop origin, or None if fitting fails.
    x, y are in pixels; z is in µm.
    """
    n_z, n_y, n_x = sub_vol.shape
    if n_z < 3 or n_y < 3 or n_x < 3:
        return None

    x_arr = np.arange(n_x, dtype=float)
    y_arr = np.arange(n_y, dtype=float)
    z_arr = z_um_vals.astype(float)

    # meshgrid layout matching (z, y, x) array indexing
    zg, yg, xg = np.meshgrid(z_arr, y_arr, x_arr, indexing="ij")

    z_peak = int(np.unravel_index(sub_vol.argmax(), sub_vol.shape)[0])
    amp    = float(sub_vol.max() - sub_vol.min())
    p0     = [amp, n_x / 2.0, n_y / 2.0, float(z_arr[z_peak]), 1.5, 0.6, float(sub_vol.min())]
    lo     = [0,   0,         0,         z_arr[0],  0.1, 0.1, -np.inf]
    hi     = [np.inf, n_x,    n_y,       z_arr[-1], 8.0, 5.0,  np.inf]

    try:
        popt, _ = curve_fit(
            _gauss3d,
            (xg.ravel(), yg.ravel(), zg.ravel()),
            sub_vol.ravel().astype(float),
            p0=p0, bounds=(lo, hi), maxfev=20_000,
        )
        return float(popt[1]), float(popt[2]), float(popt[3])
    except Exception:
        return None


# ── C. Detection and localisation ─────────────────────────────────────────────

def detect_beads_2d(
    max_proj: np.ndarray,
    min_dist_px: float,
    thresh_sigma: float,
) -> np.ndarray:
    """Find bead centres in a z-max projection.

    Returns (N, 2) integer array of (row, col) positions.
    """
    blurred = gaussian_filter(max_proj.astype(float), sigma=1.5)
    bg_mask = blurred < np.percentile(blurred, 80)
    bg_med  = np.median(blurred[bg_mask])
    bg_std  = blurred[bg_mask].std() if bg_mask.any() else 1.0
    thresh  = bg_med + thresh_sigma * bg_std
    local_mx = maximum_filter(blurred, size=int(min_dist_px)) == blurred
    return np.argwhere(local_mx & (blurred > thresh))


def localize_beads_in_volume(
    color_stacks: Dict[int, np.ndarray],
    z_um_vals: np.ndarray,
    crop_xy: int,
    crop_z: int,
    min_dist_px: float,
    thresh_sigma: float,
) -> Dict[int, pd.DataFrame]:
    """Localise beads in in-memory colour stacks via 3D Gaussian fitting.

    Parameters
    ----------
    color_stacks : {color_nm: (n_z, H, W) array} — one stack per colour channel
    z_um_vals    : (n_z,) µm — z positions corresponding to the stack planes
    crop_xy      : half-width of XY crop around each candidate (pixels)
    crop_z       : half-width of Z crop around the rough z-peak (planes)
    min_dist_px  : minimum centre-to-centre separation for detection (pixels)
    thresh_sigma : detection threshold above background (sigma units)

    Returns
    -------
    {color_nm: DataFrame(x_px, y_px, z_um)} — local pixel coords + µm z
    """
    results: Dict[int, pd.DataFrame] = {}
    z_um_vals = np.asarray(z_um_vals, dtype=float)
    n_z_total = len(z_um_vals)

    for color, stack in color_stacks.items():
        stack_f = stack.astype(np.float32)
        _, H, W = stack_f.shape

        max_proj   = stack_f.max(axis=0)
        candidates = detect_beads_2d(max_proj, min_dist_px, thresh_sigma)
        log.debug("  %d nm : %d candidates", color, len(candidates))

        rows = []
        for (r0, c0) in candidates:
            r1 = max(0, r0 - crop_xy);  r2 = min(H, r0 + crop_xy)
            c1 = max(0, c0 - crop_xy);  c2 = min(W, c0 + crop_xy)

            vol       = stack_f[:, r1:r2, c1:c2]
            z_profile = vol.mean(axis=(1, 2))
            z_peak    = int(np.argmax(z_profile))
            zi1 = max(0, z_peak - crop_z);  zi2 = min(n_z_total, z_peak + crop_z)

            fit = fit_bead_3d(vol[zi1:zi2], z_um_vals[zi1:zi2])
            if fit is None:
                continue
            lx, ly, z_um = fit
            rows.append({"x_px": c1 + lx, "y_px": r1 + ly, "z_um": z_um})

        results[color] = pd.DataFrame(rows)
        log.debug("         %d beads fitted", len(rows))

    return results


def localize_beads_in_file(
    dax_path: Union[str, Path],
    frame_table: pd.DataFrame,
    crop_xy: int,
    crop_z: int,
    min_dist_px: float,
    thresh_sigma: float,
) -> Dict[int, pd.DataFrame]:
    """Localise beads in every colour channel of one DAX file.

    Memory-efficient: uses ``compute_max_projection`` (streaming) for bead
    detection and ``read_dax_crop`` (row-by-row) for the 3-D Gaussian fit —
    never loads the full volume.  Uses ``parse_inf`` to get H and W.

    Returns
    -------
    {color_nm: DataFrame(x_px, y_px, z_um)}
    """
    from MERci.common.io import parse_inf

    inf = parse_inf(Path(dax_path).with_suffix(".inf"))
    H, W = int(inf["frame_height"]), int(inf["frame_width"])

    colors = sorted(
        int(c) for c in frame_table["color"].dropna().unique()
        if not np.isnan(float(c))
    )
    results: Dict[int, pd.DataFrame] = {}

    for color in colors:
        mask      = frame_table["color"] == color
        f_indices = frame_table.index[mask].tolist()
        z_um_vals = frame_table.loc[mask, "z"].values.astype(float)
        n_z       = len(f_indices)

        # ── Step 1: streaming max-projection for 2-D detection ────────
        max_proj   = compute_max_projection(dax_path, f_indices, H, W)
        candidates = detect_beads_2d(max_proj, min_dist_px, thresh_sigma)
        log.debug("  %d nm : %d candidates", color, len(candidates))
        print(f"  {color} nm : {len(candidates)} candidates")

        # ── Step 2: crop-and-fit each candidate ───────────────────────
        rows = []
        for (r0, c0) in candidates:
            r1 = max(0, r0 - crop_xy);  r2 = min(H, r0 + crop_xy)
            c1 = max(0, c0 - crop_xy);  c2 = min(W, c0 + crop_xy)

            # Load full Z range for the XY crop to find the rough z-peak
            vol = read_dax_crop(
                dax_path, f_indices, H, W, r1, r2, c1, c2
            ).astype(np.float32)

            z_profile  = vol.mean(axis=(1, 2))
            z_peak_idx = int(np.argmax(z_profile))
            zi1 = max(0, z_peak_idx - crop_z)
            zi2 = min(n_z, z_peak_idx + crop_z)

            fit = fit_bead_3d(vol[zi1:zi2], z_um_vals[zi1:zi2])
            if fit is None:
                continue
            lx, ly, z_um = fit
            rows.append({"x_px": c1 + lx, "y_px": r1 + ly, "z_um": z_um})

        results[color] = pd.DataFrame(rows)
        print(f"         {len(rows)} beads fitted")

    return results


# ── D. Cross-colour matching ──────────────────────────────────────────────────

def match_beads_across_colors(
    color_dfs: Dict[int, pd.DataFrame],
    ref_color: int,
    match_tol_px: float,
    pixel_size_um: float,
) -> pd.DataFrame:
    """Match beads across colour channels by XY proximity.

    Uses *ref_color* as the reference spatial grid; keeps only beads found
    in every channel.  Output x, y, z coordinates are in µm.

    Returns
    -------
    DataFrame with columns:
        bead_id, {color}_x, {color}_y, {color}_z  for every colour in color_dfs
    """
    if ref_color not in color_dfs or color_dfs[ref_color].empty:
        return pd.DataFrame()

    ref_df     = color_dfs[ref_color].reset_index(drop=True)
    ref_tree   = KDTree(ref_df[["x_px", "y_px"]].values)
    all_colors = sorted(color_dfs.keys())

    matched: Dict[int, Dict[int, pd.Series]] = {
        ref_color: {i: ref_df.loc[i] for i in ref_df.index}
    }
    for color in all_colors:
        if color == ref_color:
            continue
        q_df = color_dfs[color]
        if q_df.empty:
            matched[color] = {}
            continue
        dists, idxs = ref_tree.query(q_df[["x_px", "y_px"]].values, k=1)
        matched[color] = {}
        for qi, (dist, ref_i) in enumerate(zip(dists, idxs)):
            if dist < match_tol_px:
                # keep closest match; overwrite if a nearer one appears
                if ref_i not in matched[color] or dist < matched[color][ref_i]["_dist"]:
                    row = q_df.iloc[qi].copy()
                    row["_dist"] = dist
                    matched[color][int(ref_i)] = row

    valid_idx = [
        i for i in ref_df.index
        if all(i in matched[c] for c in all_colors)
    ]

    rows = []
    for bead_id, ref_i in enumerate(valid_idx):
        row: Dict = {"bead_id": bead_id}
        for color in all_colors:
            r = matched[color][ref_i]
            row[f"{color}_x"] = float(r["x_px"]) * pixel_size_um
            row[f"{color}_y"] = float(r["y_px"]) * pixel_size_um
            row[f"{color}_z"] = float(r["z_um"])
        rows.append(row)

    return pd.DataFrame(rows)


# ── E. Simulation ─────────────────────────────────────────────────────────────

def generate_emitter_positions(
    volume_um: Tuple[float, float, float],
    voxel_size_um: Tuple[float, float, float],
    *,
    n_emitters: Optional[int] = None,
    density_per_um3: Optional[float] = None,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> pd.DataFrame:
    """Generate uniformly random emitter positions inside a 3-D volume.

    Parameters
    ----------
    volume_um     : (x_um, y_um, z_um) — total volume dimensions in µm
    voxel_size_um : (vx, vy, vz) — voxel sizes in µm (used to infer volume
                    from a frame table when combining with other helpers)
    n_emitters    : exact number of emitters (mutually exclusive with density)
    density_per_um3 : emitters per µm³ (mutually exclusive with n_emitters)
    rng           : numpy Generator or integer seed (for reproducibility)

    Returns
    -------
    DataFrame(x_um, y_um, z_um) with *n_emitters* rows.
    """
    if (n_emitters is None) == (density_per_um3 is None):
        raise ValueError("Specify exactly one of n_emitters or density_per_um3.")

    x_um, y_um, z_um = volume_um
    if density_per_um3 is not None:
        n_emitters = max(1, int(round(density_per_um3 * x_um * y_um * z_um)))

    gen = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    return pd.DataFrame({
        "x_um": gen.uniform(0.0, x_um, n_emitters),
        "y_um": gen.uniform(0.0, y_um, n_emitters),
        "z_um": gen.uniform(0.0, z_um, n_emitters),
    })


def simulate_psf_image(
    positions_um: pd.DataFrame,
    volume_shape_px: Tuple[int, int, int],
    voxel_size_um: Tuple[float, float, float],
    psf_sigma_xy_um: float,
    psf_sigma_z_um: float,
    *,
    photon_budget: float = 1000.0,
    readout_noise_std: float = 100.0,
    bg_photons: float = 50.0,
    bit_depth: int = 16,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> np.ndarray:
    """Simulate a single (n_z, H, W) PSF image stack.

    Algorithm
    ---------
    1. Place each emitter at its nearest voxel with amplitude = photon_budget.
    2. Convolve with an anisotropic 3-D Gaussian PSF via gaussian_filter.
    3. Add uniform background (bg_photons).
    4. Apply Poisson shot noise.
    5. Add Gaussian readout noise.
    6. Clip to [0, 2**bit_depth − 1] and cast to uint16.

    Parameters
    ----------
    positions_um    : DataFrame with columns x_um, y_um, z_um
    volume_shape_px : (n_z, n_y, n_x) — output array shape in pixels/planes
    voxel_size_um   : (vx, vy, vz) — µm per pixel/plane; vx=vy=lateral, vz=axial
    psf_sigma_xy_um : lateral PSF sigma in µm
    psf_sigma_z_um  : axial PSF sigma in µm
    photon_budget   : mean photons per emitter peak
    readout_noise_std : std of Gaussian readout noise (in ADU / photon-equivalent)
    bg_photons      : uniform background level
    bit_depth       : dynamic range of output (default 16 → uint16)
    rng             : numpy Generator or integer seed

    Returns
    -------
    (n_z, n_y, n_x) uint16 array
    """
    gen = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    n_z, n_y, n_x = volume_shape_px
    vx, vy, vz    = voxel_size_um

    # Convert PSF sigmas from µm to pixels/planes
    sig_x_px = psf_sigma_xy_um / vx
    sig_y_px = psf_sigma_xy_um / vy
    sig_z_pl = psf_sigma_z_um  / vz

    # Place emitters on the voxel grid
    volume = np.zeros((n_z, n_y, n_x), dtype=np.float64)
    for _, em in positions_um.iterrows():
        iz = int(round(em["z_um"] / vz))
        iy = int(round(em["y_um"] / vy))
        ix = int(round(em["x_um"] / vx))
        if 0 <= iz < n_z and 0 <= iy < n_y and 0 <= ix < n_x:
            volume[iz, iy, ix] += photon_budget

    # Convolve with separable anisotropic Gaussian (z, y, x order)
    volume = gaussian_filter(volume, sigma=[sig_z_pl, sig_y_px, sig_x_px])

    # Add background
    volume += bg_photons

    # Poisson shot noise
    volume = gen.poisson(np.clip(volume, 0, None)).astype(np.float64)

    # Gaussian readout noise
    volume += gen.normal(0.0, readout_noise_std, size=volume.shape)

    # Clip and quantise
    max_val = 2 ** bit_depth - 1
    np.clip(volume, 0, max_val, out=volume)
    return volume.astype(np.uint16)


def add_chromatic_aberration(
    positions_df: pd.DataFrame,
    color_shifts_um: Dict[int, Tuple[float, float, float]],
) -> Dict[int, pd.DataFrame]:
    """Shift emitter positions independently for each colour channel.

    Parameters
    ----------
    positions_df    : DataFrame(x_um, y_um, z_um) — ground-truth positions
    color_shifts_um : {color_nm: (dx_um, dy_um, dz_um)} — shift per channel.
                      Include the reference colour with (0, 0, 0) to get it
                      in the returned dict.

    Returns
    -------
    {color_nm: shifted DataFrame(x_um, y_um, z_um)}
    Shifted coordinates are not clamped — they may exceed the original bounding
    box, which is intentional so that callers can handle boundary conditions.
    """
    result: Dict[int, pd.DataFrame] = {}
    for color, (dx, dy, dz) in color_shifts_um.items():
        shifted = positions_df.copy()
        shifted["x_um"] = shifted["x_um"] + dx
        shifted["y_um"] = shifted["y_um"] + dy
        shifted["z_um"] = shifted["z_um"] + dz
        result[color] = shifted
    return result


def simulate_multicolor_stack(
    positions_df: pd.DataFrame,
    colors: List[int],
    z_um_vals: np.ndarray,
    volume_shape_px: Tuple[int, int, int],
    voxel_size_um: Tuple[float, float, float],
    psf_sigma_xy_um: float,
    psf_sigma_z_um: float,
    *,
    color_shifts_um: Optional[Dict[int, Tuple[float, float, float]]] = None,
    photon_budget: float = 1000.0,
    readout_noise_std: float = 100.0,
    bg_photons: float = 50.0,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> Dict[int, np.ndarray]:
    """Simulate a multi-colour image stack, optionally with chromatic aberration.

    Parameters
    ----------
    positions_df    : ground-truth emitter positions (x_um, y_um, z_um)
    colors          : list of colour channel wavelengths in nm
    z_um_vals       : (n_z,) µm — z positions (used only to set n_z via len)
    volume_shape_px : (n_z, n_y, n_x) output shape; must match len(z_um_vals) for n_z
    voxel_size_um   : (vx, vy, vz) µm per pixel/plane
    psf_sigma_xy_um : lateral PSF sigma in µm (same for all colours)
    psf_sigma_z_um  : axial PSF sigma in µm (same for all colours)
    color_shifts_um : optional {color_nm: (dx, dy, dz)} chromatic shifts in µm;
                      omit or set to None for no aberration
    photon_budget, readout_noise_std, bg_photons : noise parameters
    rng             : numpy Generator or integer seed

    Returns
    -------
    {color_nm: (n_z, H, W) uint16 array}
    """
    gen = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng

    # Build per-colour position tables
    if color_shifts_um is not None:
        # Only shift colours that appear in color_shifts_um; rest are unshifted
        shifted_positions = add_chromatic_aberration(
            positions_df,
            {c: color_shifts_um.get(c, (0.0, 0.0, 0.0)) for c in colors},
        )
    else:
        shifted_positions = {c: positions_df for c in colors}

    stacks: Dict[int, np.ndarray] = {}
    for color in colors:
        stacks[color] = simulate_psf_image(
            shifted_positions[color],
            volume_shape_px,
            voxel_size_um,
            psf_sigma_xy_um,
            psf_sigma_z_um,
            photon_budget     = photon_budget,
            readout_noise_std = readout_noise_std,
            bg_photons        = bg_photons,
            rng               = gen,
        )
    return stacks


# ── F. Visualisation ──────────────────────────────────────────────────────────

def plot_max_projections(
    volume: np.ndarray,
    voxel_size_um: Tuple[float, float, float],
    *,
    title: str = "",
    percentile_clip: Tuple[float, float] = (1.0, 99.9),
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot XY, ZX, and ZY max-intensity projections of a 3-D volume.

    Parameters
    ----------
    volume        : (n_z, n_y, n_x) array
    voxel_size_um : (vx, vy, vz) µm per pixel/plane — used for axis labels
                    and correct aspect ratios
    title         : figure suptitle
    percentile_clip : (lo, hi) percentile clipping for contrast stretching

    Returns
    -------
    matplotlib Figure — caller may further customise or save it.
    """
    n_z, n_y, n_x = volume.shape
    vx, vy, vz    = voxel_size_um

    vol_f = volume.astype(np.float32)
    lo, hi = np.percentile(vol_f, [percentile_clip[0], percentile_clip[1]])

    def _norm(arr: np.ndarray) -> np.ndarray:
        return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    xy = _norm(vol_f.max(axis=0))               # (n_y, n_x)
    zx = _norm(vol_f.max(axis=1))               # (n_z, n_x)
    zy = _norm(vol_f.max(axis=2))               # (n_z, n_y)

    # Physical extents in µm
    ext_xy = [0, n_x * vx, 0, n_y * vy]        # [xmin, xmax, ymin, ymax]
    ext_zx = [0, n_x * vx, 0, n_z * vz]
    ext_zy = [0, n_y * vy, 0, n_z * vz]

    if figsize is None:
        figsize = (12, 4)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    def _show(ax, img, extent, xlabel, ylabel, panel_title):
        ax.imshow(img, cmap="gray", origin="lower", aspect="auto",
                  extent=extent, interpolation="nearest")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)

    _show(axes[0], xy, ext_xy, "X (µm)", "Y (µm)", "XY (z-max)")
    _show(axes[1], zx, ext_zx, "X (µm)", "Z (µm)", "ZX (y-max)")
    _show(axes[2], zy, ext_zy, "Y (µm)", "Z (µm)", "ZY (x-max)")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
