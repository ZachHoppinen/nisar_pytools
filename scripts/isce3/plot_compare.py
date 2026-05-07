"""Side-by-side plots of unwrapped phase + ionosphere screen: ours vs JPL.

Produces a 2x3 figure:
    row 1: unwrapped phase  -- ours / JPL / residual
    row 2: ionosphere screen -- ours / JPL / residual

Saves to scripts/isce3/output/compare.png
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
OUT = Path("scripts/isce3/output/compare.png")

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

    fig.suptitle("Local GUNW vs JPL GUNW (80x80 km AOI, frequencyA HH)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
