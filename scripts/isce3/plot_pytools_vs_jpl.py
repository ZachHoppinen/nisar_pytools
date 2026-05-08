"""Comparison plot: our nisar_pytools GSLC x GSLC -> "GUNW-like" vs JPL GUNW.

Produces:
    figures/isce3_comparisons/pytools_vs_jpl.png         (image side-by-side)
    figures/isce3_comparisons/pytools_vs_jpl_scatter.png (density scatter + metrics)

Bands compared: unwrapped phase + coherence. (Ionosphere is not produced by
the GSLC path -- split-spectrum needs the RSLC sub-bands.)

Run gslc_path.py first to produce scripts/isce3/output/gslc_path.h5.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

OURS = Path("scripts/isce3/output/gslc_path.h5")
JPL = Path(
    "local/gunws/"
    "NISAR_L2_PR_GUNW_004_077_A_024_005_4000_SH_"
    "20251103T124615_20251103T124650_"
    "20251115T124615_20251115T124650_"
    "X05010_N_F_J_001.h5"
)
OUT = Path("figures/isce3_comparisons/pytools_vs_jpl.png")
OUT_SCATTER = Path("figures/isce3_comparisons/pytools_vs_jpl_scatter.png")

POL = "HH"
FREQ = "frequencyA"
JPL_BASE = f"science/LSAR/GUNW/grids/{FREQ}/unwrappedInterferogram/{POL}"


def _read(path, dataset):
    with h5py.File(path, "r") as f:
        return f[dataset][...]


def _grid_pytools(path):
    with h5py.File(path, "r") as f:
        return f["xCoordinates"][...], f["yCoordinates"][...]


def _grid_jpl(path):
    with h5py.File(path, "r") as f:
        return (
            f[f"{JPL_BASE}/xCoordinates"][...],
            f[f"{JPL_BASE}/yCoordinates"][...],
        )


def _crop_to_bbox(arr, x_full, y_full, x_sub, y_sub):
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


def _plot_image_row(axes, ours, ref, title, vlim=None, cmap="RdBu_r"):
    if vlim is None:
        finite = np.concatenate([ours[np.isfinite(ours)], ref[np.isfinite(ref)]])
        vlim = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    axes[0].imshow(ours, cmap=cmap, vmin=-vlim, vmax=vlim, origin="upper")
    axes[0].set_title(f"{title}: nisar_pytools (GSLC × GSLC)")
    axes[1].imshow(ref, cmap=cmap, vmin=-vlim, vmax=vlim, origin="upper")
    axes[1].set_title(f"{title}: JPL GUNW")
    res = ours - ref
    finite_res = res[np.isfinite(res)]
    rlim = float(np.nanpercentile(np.abs(finite_res), 98)) if finite_res.size else 1.0
    im = axes[2].imshow(res, cmap=cmap, vmin=-rlim, vmax=rlim, origin="upper")
    axes[2].set_title(f"{title}: pytools − JPL")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)


def main():
    if not OURS.exists():
        raise SystemExit(f"GSLC-path output missing: {OURS}\nRun scripts/isce3/gslc_path.py first.")

    # AOI driven by ours (smaller); crop JPL to the same bbox.
    x_o, y_o = _grid_pytools(OURS)
    x_j, y_j = _grid_jpl(JPL)
    print(f"pytools grid: {len(y_o)} x {len(x_o)}  posting={abs(x_o[1]-x_o[0]):.1f} m")

    unw_o = _read(OURS, "unwrappedPhase").astype(np.float32)
    unw_j = _crop_to_bbox(_read(JPL, f"{JPL_BASE}/unwrappedPhase").astype(np.float32),
                          x_j, y_j, x_o, y_o)

    coh_o = _read(OURS, "coherenceMagnitude").astype(np.float32)
    coh_j = _crop_to_bbox(_read(JPL, f"{JPL_BASE}/coherenceMagnitude").astype(np.float32),
                          x_j, y_j, x_o, y_o)

    # Median-remove unwrapped phase (global 2π ambiguity)
    unw_o = unw_o - np.nanmedian(unw_o)
    unw_j = unw_j - np.nanmedian(unw_j)

    # --- image-domain comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    _plot_image_row(axes[0], unw_o, unw_j, "Unwrapped phase (rad, median-removed)")

    # Coherence row uses viridis [0,1], not divergent
    axes[1, 0].imshow(coh_o, cmap="viridis", vmin=0, vmax=1, origin="upper")
    axes[1, 0].set_title("Coherence: nisar_pytools (GSLC × GSLC)")
    axes[1, 1].imshow(coh_j, cmap="viridis", vmin=0, vmax=1, origin="upper")
    axes[1, 1].set_title("Coherence: JPL GUNW")
    res_coh = coh_o - coh_j
    rlim = float(np.nanpercentile(np.abs(res_coh[np.isfinite(res_coh)]), 98))
    im_r = axes[1, 2].imshow(res_coh, cmap="RdBu_r", vmin=-rlim, vmax=rlim, origin="upper")
    axes[1, 2].set_title("Coherence: pytools − JPL")
    for ax in axes[1]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.colorbar(im_r, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle("nisar_pytools GSLC × GSLC vs Production JPL GUNW (80 × 80 km AOI, frequencyA HH)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")

    # --- residual stats ---
    def stats(label, a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        res = a[valid] - b[valid]
        print(f"  {label:32s}  mean={res.mean():+8.4f}  std={res.std():7.4f}  "
              f"rms={np.sqrt((res**2).mean()):7.4f}")

    print("\nResidual statistics:")
    stats("unwrapped phase: pytools vs JPL", unw_o, unw_j)
    stats("coherence:       pytools vs JPL", coh_o, coh_j)

    # --- scatter + fit metrics ---
    fig2, sax = plt.subplots(1, 2, figsize=(12, 6))
    _scatter_panel(
        sax[0], unw_j.ravel(), unw_o.ravel(),
        xlabel="JPL unwrapped phase (rad)",
        ylabel="nisar_pytools unwrapped phase (rad)",
        title="Unwrapped phase",
        units="rad",
    )
    _scatter_panel(
        sax[1], coh_j.ravel(), coh_o.ravel(),
        xlabel="JPL coherence",
        ylabel="nisar_pytools coherence",
        title="Coherence",
    )
    fig2.suptitle("nisar_pytools GSLC × GSLC vs Production JPL GUNW")
    fig2.tight_layout()
    fig2.savefig(OUT_SCATTER, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT_SCATTER}")

    print("\nFit metrics (pytools vs JPL):")
    for name, ref, ours in [
        ("unwrapped phase", unw_j, unw_o),
        ("coherence",       coh_j, coh_o),
    ]:
        r2, rmse, pearson = _fit_metrics(ref, ours)
        print(f"  {name:24s}  R²={r2:>+7.4f}   "
              f"Pearson r={pearson:>+7.4f}   RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
