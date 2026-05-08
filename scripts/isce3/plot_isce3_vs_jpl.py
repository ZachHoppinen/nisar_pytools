"""Comparison plot: our local isce3 RSLC->GUNW vs production JPL GUNW.

Produces:
    figures/isce3_comparisons/isce3_vs_jpl.png         (image side-by-side)
    figures/isce3_comparisons/isce3_vs_jpl_scatter.png (density scatter + metrics)

Bands compared: unwrapped phase + ionosphere phase screen (both have a
global 2pi / zero-reference ambiguity, so per-product medians are
subtracted before comparison).
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

OURS = Path("scripts/isce3/output/product.h5")
JPL = Path(
    "local/gunws/"
    "NISAR_L2_PR_GUNW_004_077_A_024_005_4000_SH_"
    "20251103T124615_20251103T124650_"
    "20251115T124615_20251115T124650_"
    "X05010_N_F_J_001.h5"
)
OUT = Path("figures/isce3_comparisons/isce3_vs_jpl.png")
OUT_SCATTER = Path("figures/isce3_comparisons/isce3_vs_jpl_scatter.png")

POL = "HH"
FREQ = "frequencyA"
BASE = f"science/LSAR/GUNW/grids/{FREQ}/unwrappedInterferogram/{POL}"


def read(path, dataset):
    with h5py.File(path, "r") as f:
        return f[dataset][...]


def grid(path):
    with h5py.File(path, "r") as f:
        x = f[f"{BASE}/xCoordinates"][...]
        y = f[f"{BASE}/yCoordinates"][...]
    return x, y


def crop_to_bbox(arr, x_full, y_full, x_sub, y_sub):
    ix0 = int(np.argmin(np.abs(x_full - x_sub[0])))
    ix1 = int(np.argmin(np.abs(x_full - x_sub[-1])))
    iy0 = int(np.argmin(np.abs(y_full - y_sub[0])))
    iy1 = int(np.argmin(np.abs(y_full - y_sub[-1])))
    ix0, ix1 = sorted((ix0, ix1))
    iy0, iy1 = sorted((iy0, iy1))
    return arr[iy0:iy1 + 1, ix0:ix1 + 1]


def fit_metrics(ref, ours):
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


def scatter_panel(ax, ref, ours, xlabel, ylabel, title, units=""):
    """Density scatter (hexbin) of (ref vs ours) with 1:1 line and metrics."""
    valid = np.isfinite(ref) & np.isfinite(ours)
    x = ref[valid]
    y = ours[valid]

    ax.hexbin(x, y, gridsize=80, cmap="viridis", mincnt=1, bins="log")

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], "r-", lw=1, alpha=0.7, label="1:1")

    r2, rmse, pearson = fit_metrics(ref, ours)
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


def plot_row(axes, ours, ref, title, vlim=None, cmap="RdBu_r"):
    """Plot ours, ref, residual into three given axes."""
    if vlim is None:
        finite = np.concatenate([ours[np.isfinite(ours)], ref[np.isfinite(ref)]])
        vlim = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0

    axes[0].imshow(ours, cmap=cmap, vmin=-vlim, vmax=vlim, origin="upper")
    axes[0].set_title(f"{title}: ours")
    axes[1].imshow(ref, cmap=cmap, vmin=-vlim, vmax=vlim, origin="upper")
    axes[1].set_title(f"{title}: JPL")

    res = ours - ref
    finite_res = res[np.isfinite(res)]
    rlim = float(np.nanpercentile(np.abs(finite_res), 98)) if finite_res.size else 1.0
    im = axes[2].imshow(res, cmap=cmap, vmin=-rlim, vmax=rlim, origin="upper")
    axes[2].set_title(f"{title}: ours - JPL")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)


def main():
    x_o, y_o = grid(OURS)
    x_j, y_j = grid(JPL)

    unw_o = read(OURS, f"{BASE}/unwrappedPhase").astype(np.float32)
    unw_j = crop_to_bbox(read(JPL, f"{BASE}/unwrappedPhase").astype(np.float32),
                         x_j, y_j, x_o, y_o)

    iono_o = read(OURS, f"{BASE}/ionospherePhaseScreen").astype(np.float32)
    iono_j = crop_to_bbox(read(JPL, f"{BASE}/ionospherePhaseScreen").astype(np.float32),
                          x_j, y_j, x_o, y_o)

    # Both products carry a large constant offset on their phase screens
    # (unwrap is only defined up to a global integer multiple of 2pi;
    # ionosphere is referenced to an arbitrary zero). Subtract per-product
    # medians so the colour scales emphasize spatial structure, not offset.
    unw_o = unw_o - np.nanmedian(unw_o)
    unw_j = unw_j - np.nanmedian(unw_j)
    iono_o = iono_o - np.nanmedian(iono_o)
    iono_j = iono_j - np.nanmedian(iono_j)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plot_row(axes[0], unw_o, unw_j, "Unwrapped phase (rad, median-removed)")
    plot_row(axes[1], iono_o, iono_j, "Ionosphere screen (rad, median-removed)")

    fig.suptitle("Local isce3 GUNW vs JPL GUNW (80x80 km AOI, frequencyA HH)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")

    # Residual stats
    def stats(label, a, b):
        valid = np.isfinite(a) & np.isfinite(b)
        res = a[valid] - b[valid]
        print(f"  {label:32s}  mean={res.mean():+8.4f}  std={res.std():7.4f}  "
              f"rms={np.sqrt((res**2).mean()):7.4f}")

    print("\nResidual statistics (median-removed):")
    stats("unwrapped phase: ours vs JPL", unw_o, unw_j)
    stats("ionosphere:      ours vs JPL", iono_o, iono_j)

    # --- scatter plots: ours vs JPL with 1:1 line, hexbin density, metrics ---
    fig2, sax = plt.subplots(1, 2, figsize=(12, 6))
    scatter_panel(
        sax[0], unw_j.ravel(), unw_o.ravel(),
        xlabel="JPL unwrapped phase (rad)",
        ylabel="Ours unwrapped phase (rad)",
        title="Unwrapped phase",
        units="rad",
    )
    scatter_panel(
        sax[1], iono_j.ravel(), iono_o.ravel(),
        xlabel="JPL ionosphere screen (rad)",
        ylabel="Ours ionosphere screen (rad)",
        title="Ionosphere screen",
        units="rad",
    )
    fig2.suptitle("Local isce3 GUNW vs JPL GUNW (median-removed)")
    fig2.tight_layout()
    fig2.savefig(OUT_SCATTER, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT_SCATTER}")

    # --- fit metrics summary ---
    print("\nFit metrics (ours vs JPL):")
    for name, ref, ours in [
        ("unwrapped phase", unw_j, unw_o),
        ("ionosphere",      iono_j, iono_o),
    ]:
        r2, rmse, pearson = fit_metrics(ref, ours)
        print(f"  {name:24s}  R²={r2:>+7.4f}   "
              f"Pearson r={pearson:>+7.4f}   RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
