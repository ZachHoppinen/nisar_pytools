"""Quick test: does a Goldstein-style filter on the pytools IFG improve R² vs JPL?

The GSLC->GUNW-like path skips the filter_interferogram step that JPL's
production pipeline runs between crossmul and unwrap. This test applies
a Goldstein adaptive filter to our wrapped multilooked IFG, re-unwraps,
and prints the R² / RMSE against JPL.

Run with the nisar_pytools env:
    /Users/zmhoppinen/miniforge3/envs/nisar_pytools/bin/python scripts/isce3/test_goldstein.py
"""

from pathlib import Path

import h5py
import numpy as np

from nisar_pytools.processing import unwrap

PYTOOLS = Path("scripts/isce3/output/gslc_path.h5")
JPL = Path(
    "local/gunws/"
    "NISAR_L2_PR_GUNW_004_077_A_024_005_4000_SH_"
    "20251103T124615_20251103T124650_"
    "20251115T124615_20251115T124650_"
    "X05010_N_F_J_001.h5"
)
JPL_BASE = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"


def goldstein_filter(ifg: np.ndarray, alpha: float = 0.5, patch: int = 32) -> np.ndarray:
    """Goldstein adaptive filter (Goldstein & Werner 1998).

    Patch-wise: F = FFT(patch); F_filt = |F|^alpha * exp(i*angle(F)); patch_filt = IFFT(F_filt).
    Patches blend with a Hann window of size ``patch`` and 50% overlap stride.
    """
    ny, nx = ifg.shape
    step = patch // 2
    win = np.outer(np.hanning(patch), np.hanning(patch))
    out = np.zeros_like(ifg, dtype=np.complex128)
    wsum = np.zeros((ny, nx), dtype=np.float64)

    for y0 in range(0, ny - patch + 1, step):
        for x0 in range(0, nx - patch + 1, step):
            blk = ifg[y0:y0 + patch, x0:x0 + patch].astype(np.complex128)
            F = np.fft.fft2(blk)
            mag = np.abs(F)
            # Smooth magnitude lightly so noise doesn't dominate filter response
            mag = np.clip(mag, 1e-12, None) ** alpha
            F_filt = mag * np.exp(1j * np.angle(F))
            blk_filt = np.fft.ifft2(F_filt)
            out[y0:y0 + patch, x0:x0 + patch] += blk_filt * win
            wsum[y0:y0 + patch, x0:x0 + patch] += win

    return (out / np.maximum(wsum, 1e-12)).astype(ifg.dtype)


def _crop(arr, x_full, y_full, x_sub, y_sub):
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
    with h5py.File(PYTOOLS, "r") as f:
        x = f["xCoordinates"][...]
        y = f["yCoordinates"][...]
        ml = f["wrappedInterferogram"][...]
        coh = f["coherenceMagnitude"][...]
        unw_orig = f["unwrappedPhase"][...].astype(np.float32)
    with h5py.File(JPL, "r") as f:
        x_j = f[f"{JPL_BASE}/xCoordinates"][...]
        y_j = f[f"{JPL_BASE}/yCoordinates"][...]
        unw_j = _crop(
            f[f"{JPL_BASE}/unwrappedPhase"][...].astype(np.float32),
            x_j, y_j, x, y,
        )

    print("Applying Goldstein filter (alpha=0.5, patch=32)...")
    ml_filt = goldstein_filter(ml, alpha=0.5, patch=32)

    print("Re-unwrapping the filtered IFG with snaphu...")
    import xarray as xr
    ml_filt_da = xr.DataArray(ml_filt, dims=["y", "x"], coords={"y": y, "x": x})
    coh_da = xr.DataArray(coh, dims=["y", "x"], coords={"y": y, "x": x})
    unw_filt, _ = unwrap(ml_filt_da, coh_da, nlooks=256.0)
    unw_filt = unw_filt.values.astype(np.float32)

    # Median-removed comparisons
    unw_orig_dm = unw_orig - np.nanmedian(unw_orig)
    unw_filt_dm = unw_filt - np.nanmedian(unw_filt)
    unw_j_dm = unw_j - np.nanmedian(unw_j)

    print("\nMetrics (vs JPL):")
    for label, arr in [
        ("no filter (baseline)", unw_orig_dm),
        ("with Goldstein α=0.5 ", unw_filt_dm),
    ]:
        r2, rmse, pearson = _metrics(unw_j_dm, arr)
        print(f"  {label}  R²={r2:>+7.4f}   "
              f"Pearson r={pearson:>+7.4f}   RMSE={rmse:.4f}")

    # Also try a couple of alpha values to see sensitivity
    print("\nAlpha sensitivity:")
    for alpha in (0.2, 0.5, 0.8):
        ml_f = goldstein_filter(ml, alpha=alpha, patch=32)
        ml_f_da = xr.DataArray(ml_f, dims=["y", "x"], coords={"y": y, "x": x})
        u, _ = unwrap(ml_f_da, coh_da, nlooks=256.0)
        u_dm = u.values.astype(np.float32) - np.nanmedian(u.values)
        r2, rmse, pearson = _metrics(unw_j_dm, u_dm)
        print(f"  α={alpha:.1f}  R²={r2:>+7.4f}   "
              f"Pearson r={pearson:>+7.4f}   RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
