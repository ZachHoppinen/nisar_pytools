"""Quick test: does subtracting solid-earth-tide phase improve pytools vs JPL?

Pipeline:
    1. Compute differential ENU SET at the AOI for the two acquisition midtimes
       using pysolid (in the isce3 env -- pysolid isn't in nisar_pytools).
    2. Get LOS unit vectors from one GSLC's radarGrid (assume ENU components).
    3. Project differential ENU onto LOS to get differential LOS displacement.
    4. Convert to ifg phase via phi_set = -4 pi / lambda * delta_LOS.
    5. Subtract from pytools unwrapped phase.
    6. Print R², RMSE before and after.
    7. Save scatter plot to figures/isce3_comparisons/pytools_set_corrected_vs_jpl.png.

Run with:
    /Users/zmhoppinen/miniforge3/envs/isce3/bin/python scripts/isce3/test_set_correction.py
"""

import datetime as dt
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pysolid
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

GSLC_REF = Path(
    "local/gslcs/"
    "NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_"
    "20251103T124615_20251103T124650_X05009_N_F_J_001.h5"
)
PYTOOLS = Path("scripts/isce3/output/gslc_path.h5")
JPL = Path(
    "local/gunws/"
    "NISAR_L2_PR_GUNW_004_077_A_024_005_4000_SH_"
    "20251103T124615_20251103T124650_"
    "20251115T124615_20251115T124650_"
    "X05010_N_F_J_001.h5"
)
OUT_FIG = Path("figures/isce3_comparisons/pytools_set_corrected_vs_jpl.png")
JPL_BASE = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"

# Acquisition midtimes from the filenames (start ~ end):
#   ref: 20251103T124615 .. 20251103T124650 -> 12:46:32.5
#   sec: 20251115T124615 .. 20251115T124650 -> 12:46:32.5
DT_REF = dt.datetime(2025, 11, 3, 12, 46, 32, 500000)
DT_SEC = dt.datetime(2025, 11, 15, 12, 46, 32, 500000)


def _crop_to_bbox(arr, x_full, y_full, x_sub, y_sub):
    ix0 = int(np.argmin(np.abs(x_full - x_sub[0])))
    ix1 = int(np.argmin(np.abs(x_full - x_sub[-1])))
    iy0 = int(np.argmin(np.abs(y_full - y_sub[0])))
    iy1 = int(np.argmin(np.abs(y_full - y_sub[-1])))
    ix0, ix1 = sorted((ix0, ix1))
    iy0, iy1 = sorted((iy0, iy1))
    return arr[iy0:iy1 + 1, ix0:ix1 + 1]


def _metrics(ref, ours):
    valid = np.isfinite(ref) & np.isfinite(ours)
    x = ref[valid]
    y = ours[valid]
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    ss_res = float(np.sum((y - x) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    return r2, rmse, pearson


def main():
    # ---- Load pytools output and JPL on the AOI grid ----
    with h5py.File(PYTOOLS, "r") as f:
        x_utm = f["xCoordinates"][...]
        y_utm = f["yCoordinates"][...]
        unw_pytools = f["unwrappedPhase"][...].astype(np.float32)

    with h5py.File(JPL, "r") as f:
        x_jpl = f[f"{JPL_BASE}/xCoordinates"][...]
        y_jpl = f[f"{JPL_BASE}/yCoordinates"][...]
        unw_jpl_full = f[f"{JPL_BASE}/unwrappedPhase"][...].astype(np.float32)
    unw_jpl = _crop_to_bbox(unw_jpl_full, x_jpl, y_jpl, x_utm, y_utm)
    print(f"Output grid: {unw_pytools.shape}  posting={abs(x_utm[1]-x_utm[0]):.1f} m")

    # ---- pysolid lat/lon grid covering the AOI ----
    tf = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)
    lon_min, lat_min = tf.transform(x_utm.min(), y_utm.min())
    lon_max, lat_max = tf.transform(x_utm.max(), y_utm.max())
    pad = 0.05  # padding so interp stays inside
    lon_min -= pad
    lon_max += pad
    lat_min -= pad
    lat_max += pad

    n_lat, n_lon = 60, 60
    atr = {
        "LENGTH": n_lat,
        "WIDTH": n_lon,
        "X_FIRST": lon_min,
        "Y_FIRST": lat_max,
        "X_STEP": (lon_max - lon_min) / (n_lon - 1),
        "Y_STEP": -(lat_max - lat_min) / (n_lat - 1),
    }
    print(f"pysolid lat/lon grid: {n_lat} x {n_lon}  "
          f"lat=[{lat_min:.3f}, {lat_max:.3f}]  lon=[{lon_min:.3f}, {lon_max:.3f}]")

    print(f"Computing SET at ref={DT_REF}...")
    e_ref, n_ref, u_ref = pysolid.calc_solid_earth_tides_grid(DT_REF, atr, verbose=False)
    print(f"Computing SET at sec={DT_SEC}...")
    e_sec, n_sec, u_sec = pysolid.calc_solid_earth_tides_grid(DT_SEC, atr, verbose=False)
    de, dn, du = e_sec - e_ref, n_sec - n_ref, u_sec - u_ref
    print(f"  Differential ENU range (mm):  "
          f"E [{1e3*de.min():+.2f}, {1e3*de.max():+.2f}]  "
          f"N [{1e3*dn.min():+.2f}, {1e3*dn.max():+.2f}]  "
          f"U [{1e3*du.min():+.2f}, {1e3*du.max():+.2f}]")

    # axes for interp (need ascending)
    lat_axis = np.linspace(lat_max, lat_min, n_lat)[::-1]  # ascending
    lon_axis = np.linspace(lon_min, lon_max, n_lon)
    de_i = RegularGridInterpolator((lat_axis, lon_axis), de[::-1], bounds_error=False, fill_value=None)
    dn_i = RegularGridInterpolator((lat_axis, lon_axis), dn[::-1], bounds_error=False, fill_value=None)
    du_i = RegularGridInterpolator((lat_axis, lon_axis), du[::-1], bounds_error=False, fill_value=None)

    # ---- LOS unit vector at the AOI from radarGrid ----
    with h5py.File(GSLC_REF, "r") as f:
        rg = f["science/LSAR/GSLC/metadata/radarGrid"]
        losx_cube = rg["losUnitVectorX"][...]
        losy_cube = rg["losUnitVectorY"][...]
        x_rg = rg["xCoordinates"][...]
        y_rg = rg["yCoordinates"][...]
        center_freq = float(f["science/LSAR/GSLC/grids/frequencyA/centerFrequency"][()])
    wavelength = 2.99792458e8 / center_freq
    print(f"  lambda = {wavelength:.4f} m")

    mid = losx_cube.shape[0] // 2
    losx_mid = losx_cube[mid]
    losy_mid = losy_cube[mid]
    # radarGrid y is descending in array order. Reverse for interp.
    if y_rg[0] > y_rg[-1]:
        y_rg_a = y_rg[::-1]
        losx_mid = losx_mid[::-1]
        losy_mid = losy_mid[::-1]
    else:
        y_rg_a = y_rg
    losx_i = RegularGridInterpolator((y_rg_a, x_rg), losx_mid, bounds_error=False, fill_value=None)
    losy_i = RegularGridInterpolator((y_rg_a, x_rg), losy_mid, bounds_error=False, fill_value=None)

    # ---- Sample everything on the 80 m UTM grid ----
    xx, yy = np.meshgrid(x_utm, y_utm)
    lon_grid, lat_grid = tf.transform(xx, yy)
    pts_ll = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    pts_yx = np.stack([yy.ravel(), xx.ravel()], axis=-1)

    de_g = de_i(pts_ll).reshape(unw_pytools.shape)
    dn_g = dn_i(pts_ll).reshape(unw_pytools.shape)
    du_g = du_i(pts_ll).reshape(unw_pytools.shape)
    losx_g = losx_i(pts_yx).reshape(unw_pytools.shape)
    losy_g = losy_i(pts_yx).reshape(unw_pytools.shape)
    losz_g = np.sqrt(np.maximum(0.0, 1.0 - losx_g ** 2 - losy_g ** 2))

    # Differential SET projected onto LOS (positive = motion toward sat)
    delta_los = de_g * losx_g + dn_g * losy_g + du_g * losz_g
    print(f"  delta_LOS_set range (mm): [{1e3*delta_los.min():+.3f}, {1e3*delta_los.max():+.3f}]")

    # Phase contribution to the ifg: target moving toward sat between acqs
    # gives R_sec < R_ref, hence ifg_phase = phase_ref - phase_sec = -4pi/lambda * delta_LOS.
    phi_set = -4.0 * np.pi / wavelength * delta_los
    print(f"  phi_set range (rad):     [{phi_set.min():+.4f}, {phi_set.max():+.4f}]  "
          f"span = {phi_set.max() - phi_set.min():.4f}")

    # ---- Apply correction (try both signs) ----
    pytools_dm = unw_pytools - np.nanmedian(unw_pytools)
    jpl_dm = unw_jpl - np.nanmedian(unw_jpl)

    pytools_minus = (unw_pytools - phi_set) - np.nanmedian(unw_pytools - phi_set)
    pytools_plus  = (unw_pytools + phi_set) - np.nanmedian(unw_pytools + phi_set)

    print("\nMetrics:")
    for label, arr in [("no correction        ", pytools_dm),
                       ("pytools - phi_set    ", pytools_minus),
                       ("pytools + phi_set    ", pytools_plus)]:
        r2, rmse, pearson = _metrics(jpl_dm, arr)
        print(f"  {label}  R²={r2:>+7.4f}   "
              f"Pearson r={pearson:>+7.4f}   RMSE={rmse:.4f}")

    # ---- Plot scatter for the better-signed correction ----
    r2_minus = _metrics(jpl_dm, pytools_minus)[0]
    r2_plus = _metrics(jpl_dm, pytools_plus)[0]
    use_corrected = pytools_minus if r2_minus > r2_plus else pytools_plus
    used_sign = "−" if r2_minus > r2_plus else "+"

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for axi, arr, title in [
        (ax[0], pytools_dm, "Before SET correction"),
        (ax[1], use_corrected, f"After SET correction (sign: {used_sign})"),
    ]:
        valid = np.isfinite(jpl_dm) & np.isfinite(arr)
        x = jpl_dm[valid]
        y = arr[valid]
        axi.hexbin(x, y, gridsize=80, cmap="viridis", mincnt=1, bins="log")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        axi.plot([lo, hi], [lo, hi], "r-", lw=1, alpha=0.7, label="1:1")
        r2, rmse, pearson = _metrics(jpl_dm, arr)
        axi.text(0.04, 0.96,
                 f"R² = {r2:.4f}\nPearson r = {pearson:.4f}\nRMSE = {rmse:.4f} rad",
                 transform=axi.transAxes, va="top", ha="left", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5"))
        axi.set_xlabel("JPL unwrapped phase (rad)")
        axi.set_ylabel("nisar_pytools unwrapped phase (rad)")
        axi.set_title(title)
        axi.set_aspect("equal", adjustable="box")
        axi.set_xlim(lo, hi)
        axi.set_ylim(lo, hi)
        axi.legend(loc="lower right", fontsize=8)

    fig.suptitle("nisar_pytools GSLC × GSLC vs JPL GUNW — solid earth tide correction test")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"\nWrote {OUT_FIG}")


if __name__ == "__main__":
    main()
