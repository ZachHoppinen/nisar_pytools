"""Validate nisar_pytools processing against the official NISAR GUNW product.

Compares our GSLC-derived interferogram, coherence, and baselines against
the GUNW processor output for the same acquisition pair.

Files used:
- GSLC 1: track 77, frame 24, 2025-11-03
- GSLC 2: track 77, frame 24, 2025-11-15
- GUNW:   track 77, frame 24, 2025-11-03 / 2025-11-15
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nisar_pytools import open_nisar
from nisar_pytools.processing import interferogram, coherence, multilook
from nisar_pytools.processing.baseline import compute_baseline

GSLC1 = "local/gslcs/NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001.h5"
GSLC2 = "local/gslcs/NISAR_L2_PR_GSLC_005_077_A_024_4005_DHDH_A_20251115T124615_20251115T124650_X05009_N_F_J_001.h5"
GUNW = "local/gunws/NISAR_L2_PR_GUNW_004_077_A_024_005_4000_SH_20251103T124615_20251103T124650_20251115T124615_20251115T124650_X05010_N_F_J_001.h5"

# Subset: center 2000x2000 pixels from GSLC (5m) = 500x500 at 20m
CY, CX, HW = 35000, 36000, 1000


def interp_gunw_to_our_grid(gunw_da, our_y, our_x):
    """Interpolate a GUNW 2D DataArray to our coordinate grid."""
    gy = gunw_da.y.values
    gx = gunw_da.x.values
    gvals = gunw_da.values

    if gy[0] > gy[-1]:
        gy = gy[::-1]
        gvals = gvals[::-1, :]

    interp = RegularGridInterpolator(
        (gy, gx), gvals, bounds_error=False, fill_value=np.nan
    )
    yy, xx = np.meshgrid(our_y, our_x, indexing="ij")
    return interp(np.column_stack([yy.ravel(), xx.ravel()])).reshape(len(our_y), len(our_x))


def interp_gunw_complex_to_our_grid(gunw_da, our_y, our_x):
    """Interpolate a complex GUNW DataArray to our grid (real + imag separately)."""
    gy = gunw_da.y.values
    gx = gunw_da.x.values
    gvals = gunw_da.values

    if gy[0] > gy[-1]:
        gy = gy[::-1]
        gvals = gvals[::-1, :]

    real_interp = RegularGridInterpolator((gy, gx), gvals.real, bounds_error=False, fill_value=np.nan)
    imag_interp = RegularGridInterpolator((gy, gx), gvals.imag, bounds_error=False, fill_value=np.nan)

    yy, xx = np.meshgrid(our_y, our_x, indexing="ij")
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    return (real_interp(pts) + 1j * imag_interp(pts)).reshape(len(our_y), len(our_x))


def main():
    results = {}

    # Load data
    print("Loading GSLCs and GUNW...")
    gslc1 = open_nisar(GSLC1)
    gslc2 = open_nisar(GSLC2)
    gunw = open_nisar(GUNW)

    # Extract SLC subsets
    print(f"Extracting {2*HW}x{2*HW} pixel subset from GSLCs...")
    slc1 = gslc1["science/LSAR/GSLC/grids/frequencyA"].dataset["HH"].isel(
        y=slice(CY - HW, CY + HW), x=slice(CX - HW, CX + HW)
    ).compute()
    slc2 = gslc2["science/LSAR/GSLC/grids/frequencyA"].dataset["HH"].isel(
        y=slice(CY - HW, CY + HW), x=slice(CX - HW, CX + HW)
    ).compute()

    # =========================================================================
    # 1. INTERFEROGRAM (wrapped phase)
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. INTERFEROGRAM COMPARISON")
    print("=" * 70)

    our_ifg = interferogram(slc1, slc2)
    our_ml = multilook(our_ifg, looks_y=4, looks_x=4)
    our_phase = np.angle(our_ml.values)

    gunw_wifg = gunw["science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH"].dataset[
        "wrappedInterferogram"
    ].sel(
        x=slice(float(our_ml.x[0]) - 10, float(our_ml.x[-1]) + 10),
        y=slice(float(our_ml.y[0]) + 10, float(our_ml.y[-1]) - 10),
    ).compute()

    gunw_interp = interp_gunw_complex_to_our_grid(gunw_wifg, our_ml.y.values, our_ml.x.values)
    gunw_phase = np.angle(gunw_interp)

    valid = ~np.isnan(gunw_phase) & ~np.isnan(our_phase) & (np.abs(gunw_interp) > 0)
    phase_diff = np.angle(np.exp(1j * (our_phase[valid] - gunw_phase[valid])))
    phase_corr = np.abs(np.mean(np.exp(1j * phase_diff)))

    print(f"   Phase correlation:    {phase_corr:.4f}")
    print(f"   Phase bias (mean):    {np.mean(phase_diff):.4f} rad ({np.degrees(np.mean(phase_diff)):.2f} deg)")
    print(f"   Phase std:            {np.std(phase_diff):.4f} rad ({np.degrees(np.std(phase_diff)):.2f} deg)")
    results["ifg_phase_correlation"] = phase_corr
    results["ifg_phase_bias_rad"] = np.mean(phase_diff)

    # =========================================================================
    # 2. COHERENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. COHERENCE COMPARISON")
    print("=" * 70)

    gunw_coh = gunw["science/LSAR/GUNW/grids/frequencyA/wrappedInterferogram/HH"].dataset[
        "coherenceMagnitude"
    ].sel(
        x=slice(float(slc1.x[0]) - 10, float(slc1.x[-1]) + 10),
        y=slice(float(slc1.y[0]) + 10, float(slc1.y[-1]) - 10),
    ).compute()

    print(f"   GUNW coherence: shape={gunw_coh.shape}, mean={float(gunw_coh.mean()):.3f}")
    print()

    # Test multiple methods
    best_corr = 0
    best_method = ""
    print(f"   {'Method':<12} {'Window':>8} {'Our Mean':>10} {'GUNW Mean':>10} {'Corr':>8} {'RMSE':>8} {'Bias':>8}")
    print("   " + "-" * 68)

    for method, windows in [("boxcar", [5, 7, 11, 15]), ("gaussian", [2, 3, 5])]:
        for win in windows:
            our_coh = coherence(slc1, slc2, window_size=win, method=method)
            gunw_on_grid = interp_gunw_to_our_grid(gunw_coh, our_coh.y.values, our_coh.x.values)
            v = ~np.isnan(gunw_on_grid) & (gunw_on_grid > 0)
            our_v = our_coh.values[v]
            gunw_v = gunw_on_grid[v]
            corr = np.corrcoef(our_v, gunw_v)[0, 1]
            rmse = np.sqrt(np.mean((our_v - gunw_v) ** 2))
            bias = np.mean(our_v - gunw_v)
            marker = " <-- best" if corr > best_corr else ""
            if corr > best_corr:
                best_corr = corr
                best_method = f"{method} {win}"
            print(f"   {method:<12} {win:>8} {np.mean(our_v):>10.3f} {np.mean(gunw_v):>10.3f} {corr:>8.3f} {rmse:>8.3f} {bias:>8.3f}{marker}")

    print(f"\n   Best match: {best_method} (correlation {best_corr:.3f})")
    results["coh_best_method"] = best_method
    results["coh_best_correlation"] = best_corr

    # =========================================================================
    # 3. BASELINES
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. BASELINE COMPARISON")
    print("=" * 70)

    # Our baselines (secondary as reference to match GUNW sign convention)
    our_baselines = compute_baseline(gslc2, gslc1)

    # GUNW baselines (middle height layer)
    rg = gunw["science/LSAR/GUNW/metadata/radarGrid"].dataset
    mid = rg.sizes["z"] // 2
    gunw_bperp = np.asarray(rg["perpendicularBaseline"].isel(z=mid))
    gunw_bpar = np.asarray(rg["parallelBaseline"].isel(z=mid))

    our_bperp = our_baselines["perpendicular_baseline"].values
    our_bpar = our_baselines["parallel_baseline"].values

    print(f"   {'':>25} {'Our':>12} {'GUNW':>12} {'Diff':>12}")
    print("   " + "-" * 55)
    print(f"   {'B_perp mean (m)':>25} {np.nanmean(our_bperp):>12.1f} {np.nanmean(gunw_bperp):>12.1f} {np.nanmean(our_bperp) - np.nanmean(gunw_bperp):>12.1f}")
    print(f"   {'B_perp range (m)':>25} [{np.nanmin(our_bperp):.1f}, {np.nanmax(our_bperp):.1f}]   [{np.nanmin(gunw_bperp):.1f}, {np.nanmax(gunw_bperp):.1f}]")
    print(f"   {'B_par mean (m)':>25} {np.nanmean(our_bpar):>12.1f} {np.nanmean(gunw_bpar):>12.1f} {np.nanmean(our_bpar) - np.nanmean(gunw_bpar):>12.1f}")
    print(f"   {'B_par range (m)':>25} [{np.nanmin(our_bpar):.1f}, {np.nanmax(our_bpar):.1f}]   [{np.nanmin(gunw_bpar):.1f}, {np.nanmax(gunw_bpar):.1f}]")

    results["bperp_diff_m"] = np.nanmean(our_bperp) - np.nanmean(gunw_bperp)
    results["bpar_diff_m"] = np.nanmean(our_bpar) - np.nanmean(gunw_bpar)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    checks = [
        ("Interferogram phase correlation > 0.90", results["ifg_phase_correlation"] > 0.90),
        ("Interferogram phase bias < 0.1 rad", abs(results["ifg_phase_bias_rad"]) < 0.1),
        ("Coherence best correlation > 0.90", results["coh_best_correlation"] > 0.90),
        ("B_perp difference < 2 m", abs(results["bperp_diff_m"]) < 2),
        ("B_par difference < 2 m", abs(results["bpar_diff_m"]) < 2),
    ]

    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"   [{status}] {desc}")

    print()
    if all_pass:
        print("   ALL VALIDATION CHECKS PASSED")
    else:
        print("   SOME CHECKS FAILED — review results above")

    return results


if __name__ == "__main__":
    main()
