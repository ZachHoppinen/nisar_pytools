"""Three-way comparison: JPL GUNW vs isce3-RSLC GUNW vs nisar_pytools GSLC path.

Loads the same 80 x 80 km AOI from three independent processing paths and
plots them side by side. The pedagogical point: identical inputs (same SAR
acquisitions, same scene), three very different processor stacks, three
unwrapped phase maps. How close do they agree?

Saves: figures/isce3_comparisons/three_way.png
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

JPL = Path(
    "local/gunws/"
    "NISAR_L2_PR_GUNW_004_077_A_024_005_4000_SH_"
    "20251103T124615_20251103T124650_"
    "20251115T124615_20251115T124650_"
    "X05010_N_F_J_001.h5"
)
RSLC_GUNW = Path("scripts/isce3/output/product.h5")  # from nisar.workflows.insar
GSLC_PATH = Path("scripts/isce3/output/gslc_path.h5")  # from gslc_path.py

OUT = Path("figures/isce3_comparisons/three_way.png")
OUT_SCATTER = Path("figures/isce3_comparisons/three_way_scatter.png")

POL = "HH"
FREQ = "frequencyA"
BASE = f"science/LSAR/GUNW/grids/{FREQ}/unwrappedInterferogram/{POL}"


def _read(path, dataset):
    with h5py.File(path, "r") as f:
        return f[dataset][...]


def _grid(path, base=BASE):
    with h5py.File(path, "r") as f:
        x = f[f"{base}/xCoordinates"][...]
        y = f[f"{base}/yCoordinates"][...]
    return x, y


def _crop_to_bbox(arr, x_full, y_full, x_sub, y_sub):
    """Crop arr to the AOI defined by (x_sub, y_sub)."""
    ix0 = int(np.argmin(np.abs(x_full - x_sub[0])))
    ix1 = int(np.argmin(np.abs(x_full - x_sub[-1])))
    iy0 = int(np.argmin(np.abs(y_full - y_sub[0])))
    iy1 = int(np.argmin(np.abs(y_full - y_sub[-1])))
    ix0, ix1 = sorted((ix0, ix1))
    iy0, iy1 = sorted((iy0, iy1))
    return arr[iy0:iy1 + 1, ix0:ix1 + 1]


def _imshow(ax, arr, title, vlim=None, cmap="RdBu_r"):
    if vlim is None:
        finite = arr[np.isfinite(arr)]
        vlim = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    im = ax.imshow(arr, cmap=cmap, vmin=-vlim, vmax=vlim, origin="upper")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _fit_metrics(ref, ours):
    """R², RMSE, Pearson r between paired arrays. Ignores NaNs."""
    valid = np.isfinite(ref) & np.isfinite(ours)
    x = ref[valid]
    y = ours[valid]
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    ss_res = float(np.sum((y - x) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return r2, rmse, pearson


def _scatter_panel(ax, ref, ours, xlabel, ylabel, title, units=""):
    """Density scatter (hexbin) of (ref vs ours) with 1:1 line and metrics."""
    valid = np.isfinite(ref) & np.isfinite(ours)
    x = ref[valid]
    y = ours[valid]

    ax.hexbin(x, y, gridsize=80, cmap="viridis", mincnt=1, bins="log")

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], "r-", lw=1, alpha=0.7, label="1:1")

    r2, rmse, pearson = _fit_metrics(ref, ours)
    txt = f"R² = {r2:.4f}\nPearson r = {pearson:.4f}\nRMSE = {rmse:.4f} {units}"
    ax.text(
        0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5"),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="lower right", fontsize=8)


def main():
    if not GSLC_PATH.exists():
        raise SystemExit("GSLC-path output missing. Run scripts/isce3/gslc_path.py first.")
    if not RSLC_GUNW.exists():
        raise SystemExit("RSLC-path GUNW missing. Run scripts/isce3/run_insar.sh first.")

    # --- AOI defined by GSLC-path output (smallest, set by AOI bbox) ---
    x_gslc, y_gslc = _grid(GSLC_PATH, base="")  # gslc_path.h5 stores at top-level
    print(f"GSLC-path AOI grid: {len(y_gslc)} x {len(x_gslc)}  posting={abs(x_gslc[1] - x_gslc[0]):.1f} m")

    # --- Unwrapped phase from each path ---
    unw_gslc = _read(GSLC_PATH, "unwrappedPhase").astype(np.float32)

    x_rslc, y_rslc = _grid(RSLC_GUNW)
    unw_rslc = _read(RSLC_GUNW, f"{BASE}/unwrappedPhase").astype(np.float32)
    unw_rslc = _crop_to_bbox(unw_rslc, x_rslc, y_rslc, x_gslc, y_gslc)

    x_jpl, y_jpl = _grid(JPL)
    unw_jpl = _read(JPL, f"{BASE}/unwrappedPhase").astype(np.float32)
    unw_jpl = _crop_to_bbox(unw_jpl, x_jpl, y_jpl, x_gslc, y_gslc)

    # --- Coherence from each ---
    coh_gslc = _read(GSLC_PATH, "coherenceMagnitude").astype(np.float32)
    coh_rslc = _crop_to_bbox(
        _read(RSLC_GUNW, f"{BASE}/coherenceMagnitude").astype(np.float32),
        x_rslc, y_rslc, x_gslc, y_gslc,
    )
    coh_jpl = _crop_to_bbox(
        _read(JPL, f"{BASE}/coherenceMagnitude").astype(np.float32),
        x_jpl, y_jpl, x_gslc, y_gslc,
    )

    # Phase has a global integer-2pi ambiguity; remove per-product median
    # so colour scales emphasize spatial structure.
    for arr in (unw_gslc, unw_rslc, unw_jpl):
        arr -= np.nanmedian(arr)

    # --- Plot 2 x 3: rows = (unwrapped phase, coherence), cols = (GSLC-path, isce3-RSLC, JPL) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Use the JPL data range for consistent unwrap colour scaling across panels
    finite = unw_jpl[np.isfinite(unw_jpl)]
    unw_vlim = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0

    _imshow(axes[0, 0], unw_gslc, "Unwrapped phase: GSLC x GSLC\n(nisar_pytools, 80 m)", vlim=unw_vlim)
    _imshow(axes[0, 1], unw_rslc, "Unwrapped phase: RSLC -> GUNW\n(local isce3, 80 m)", vlim=unw_vlim)
    _imshow(axes[0, 2], unw_jpl, "Unwrapped phase: JPL GUNW\n(reference, 80 m)", vlim=unw_vlim)

    _imshow(axes[1, 0], coh_gslc, "Coherence: GSLC x GSLC", vlim=1.0, cmap="viridis")
    _imshow(axes[1, 1], coh_rslc, "Coherence: RSLC -> GUNW", vlim=1.0, cmap="viridis")
    _imshow(axes[1, 2], coh_jpl, "Coherence: JPL GUNW", vlim=1.0, cmap="viridis")

    # Override viridis vmin (we passed vlim=1.0 above; set vmin=0 explicitly)
    for ax in axes[1]:
        for im in ax.get_images():
            im.set_clim(0.0, 1.0)

    fig.suptitle("Three-way GUNW comparison (80 x 80 km AOI, frequencyA HH)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")

    # --- Print residual stats across all pairs ---
    def stats(label, a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        res = a[valid] - b[valid]
        print(f"  {label:36s}  mean={res.mean():+8.4f}  std={res.std():7.4f}  "
              f"rms={np.sqrt((res**2).mean()):7.4f}")

    print("\nResidual statistics (median-removed for unwrapped phase):")
    stats("unwrapped: GSLC-path vs JPL", unw_gslc, unw_jpl)
    stats("unwrapped: RSLC-path vs JPL", unw_rslc, unw_jpl)
    stats("unwrapped: GSLC-path vs RSLC-path", unw_gslc, unw_rslc)
    print()
    stats("coherence: GSLC-path vs JPL", coh_gslc, coh_jpl)
    stats("coherence: RSLC-path vs JPL", coh_rslc, coh_jpl)
    stats("coherence: GSLC-path vs RSLC-path", coh_gslc, coh_rslc)

    # --- scatter plots: each path vs JPL (the reference), 2 bands x 2 paths ---
    fig2, sax = plt.subplots(2, 2, figsize=(12, 12))
    _scatter_panel(
        sax[0, 0], unw_jpl.ravel(), unw_gslc.ravel(),
        xlabel="JPL unwrapped phase (rad)",
        ylabel="GSLC-path unwrapped phase (rad)",
        title="Unwrapped phase: GSLC-path vs JPL",
        units="rad",
    )
    _scatter_panel(
        sax[0, 1], unw_jpl.ravel(), unw_rslc.ravel(),
        xlabel="JPL unwrapped phase (rad)",
        ylabel="RSLC-path unwrapped phase (rad)",
        title="Unwrapped phase: RSLC-path vs JPL",
        units="rad",
    )
    _scatter_panel(
        sax[1, 0], coh_jpl.ravel(), coh_gslc.ravel(),
        xlabel="JPL coherence",
        ylabel="GSLC-path coherence",
        title="Coherence: GSLC-path vs JPL",
    )
    _scatter_panel(
        sax[1, 1], coh_jpl.ravel(), coh_rslc.ravel(),
        xlabel="JPL coherence",
        ylabel="RSLC-path coherence",
        title="Coherence: RSLC-path vs JPL",
    )
    fig2.suptitle("GSLC-path and RSLC-path vs JPL GUNW (median-removed phase)")
    fig2.tight_layout()
    fig2.savefig(OUT_SCATTER, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT_SCATTER}")

    # --- fit metrics summary ---
    print("\nFit metrics (each path vs JPL):")
    pairs = [
        ("unwrapped: GSLC-path vs JPL", unw_jpl, unw_gslc),
        ("unwrapped: RSLC-path vs JPL", unw_jpl, unw_rslc),
        ("coherence: GSLC-path vs JPL", coh_jpl, coh_gslc),
        ("coherence: RSLC-path vs JPL", coh_jpl, coh_rslc),
    ]
    for name, ref, ours in pairs:
        r2, rmse, pearson = _fit_metrics(ref, ours)
        print(f"  {name:34s}  R²={r2:>+7.4f}   "
              f"Pearson r={pearson:>+7.4f}   RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
