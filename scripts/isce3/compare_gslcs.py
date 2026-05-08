"""Compare our locally-processed GSLC against the JPL X05009 GSLC, pixel-by-pixel.

Loads both GSLCs at their shared 80x80 km AOI, then quantifies amplitude
agreement, phase agreement, and complex coherence between them. The
question: does the JPL GSLC processing (X05009, with iono geolocation
correction) produce different complex pixels than our local processing
(no TEC, so no iono geolocation correction) at the same UTM (x, y)?

If amplitude + phase agree → JPL's GSLC processing is reproducible from
RSLCs with the same isce3 binaries, and the pytools-vs-JPL residual
is structural to the GSLC vs RSLC pipeline order (not GSLC processing
specifics).

Saves:
    figures/isce3_comparisons/gslc_local_vs_jpl.png
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

LOC = Path("scripts/isce3/output_gslc/gslc.h5")
JPL = Path(
    "local/gslcs/"
    "NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_"
    "20251103T124615_20251103T124650_X05009_N_F_J_001.h5"
)
OUT = Path("figures/isce3_comparisons/gslc_local_vs_jpl.png")

POL = "HH"
SLC_PATH = f"science/LSAR/GSLC/grids/frequencyA/{POL}"
GRID = "science/LSAR/GSLC/grids/frequencyA"


def _coh_window(s1: np.ndarray, s2: np.ndarray, win: int = 11) -> np.ndarray:
    """Boxcar-windowed complex coherence: |<s1 conj(s2)>| / sqrt(<|s1|²> <|s2|²>)."""
    from scipy.ndimage import uniform_filter

    num_re = uniform_filter((s1 * np.conj(s2)).real, size=win)
    num_im = uniform_filter((s1 * np.conj(s2)).imag, size=win)
    num = np.sqrt(num_re ** 2 + num_im ** 2)
    p1 = uniform_filter(np.abs(s1) ** 2, size=win)
    p2 = uniform_filter(np.abs(s2) ** 2, size=win)
    denom = np.sqrt(p1 * p2)
    return np.where(denom > 0, num / denom, 0.0)


def main():
    # Local GSLC: covers the 80 x 80 km AOI directly
    with h5py.File(LOC, "r") as f:
        x_loc = f[f"{GRID}/xCoordinates"][...]
        y_loc = f[f"{GRID}/yCoordinates"][...]
        slc_loc = f[SLC_PATH][...].astype(np.complex64)
    print(f"Local GSLC: shape={slc_loc.shape}  posting={abs(x_loc[1]-x_loc[0]):.2f} m")

    # JPL GSLC: covers the full 360 x 355 km scene; crop to local AOI
    with h5py.File(JPL, "r") as f:
        x_jpl = f[f"{GRID}/xCoordinates"][...]
        y_jpl = f[f"{GRID}/yCoordinates"][...]
        # Crop indices
        ix0 = int(np.argmin(np.abs(x_jpl - x_loc[0])))
        ix1 = int(np.argmin(np.abs(x_jpl - x_loc[-1])))
        iy0 = int(np.argmin(np.abs(y_jpl - y_loc[0])))
        iy1 = int(np.argmin(np.abs(y_jpl - y_loc[-1])))
        ix0, ix1 = sorted((ix0, ix1))
        iy0, iy1 = sorted((iy0, iy1))
        # Verify alignment
        if abs(x_jpl[ix0] - x_loc[0]) > 1.0 or abs(y_jpl[iy0] - y_loc[0]) > 1.0:
            print(f"  WARNING: corner offset > 1 m: x={x_jpl[ix0] - x_loc[0]:.2f} y={y_jpl[iy0] - y_loc[0]:.2f}")
        slc_jpl = f[SLC_PATH][iy0:iy1 + 1, ix0:ix1 + 1].astype(np.complex64)
    print(f"JPL GSLC (cropped): shape={slc_jpl.shape}")

    if slc_loc.shape != slc_jpl.shape:
        # truncate to common area
        ny = min(slc_loc.shape[0], slc_jpl.shape[0])
        nx = min(slc_loc.shape[1], slc_jpl.shape[1])
        slc_loc = slc_loc[:ny, :nx]
        slc_jpl = slc_jpl[:ny, :nx]
        print(f"  Truncated both to common shape {slc_loc.shape}")

    # ------- amplitude comparison -------
    amp_loc = np.abs(slc_loc)
    amp_jpl = np.abs(slc_jpl)
    valid = np.isfinite(amp_loc) & np.isfinite(amp_jpl) & (amp_jpl > 0)
    print(f"\nValid pixels: {valid.sum():,} / {valid.size:,}  ({valid.mean()*100:.1f}%)")

    rel_amp = amp_loc[valid] / amp_jpl[valid]
    print("\nAmplitude ratio (local/JPL):")
    print(f"  median   {np.median(rel_amp):.6f}")
    print(f"  mean     {rel_amp.mean():.6f}")
    print(f"  std      {rel_amp.std():.6f}")
    print(f"  5%-95%   [{np.percentile(rel_amp, 5):.4f}, {np.percentile(rel_amp, 95):.4f}]")

    # ------- phase comparison -------
    # phase difference of complex ratio: angle(s_loc * conj(s_jpl))
    phase_diff = np.angle(slc_loc * np.conj(slc_jpl))
    pd_valid = phase_diff[valid]
    print("\nPhase difference (rad), pixel-by-pixel:")
    print(f"  median       {np.median(pd_valid):+.6f}")
    print(f"  mean         {pd_valid.mean():+.6f}")
    print(f"  std          {pd_valid.std():.6f}")
    print(f"  5%-95%       [{np.percentile(pd_valid, 5):+.4f}, {np.percentile(pd_valid, 95):+.4f}]")
    print(f"  |median|/2pi {abs(np.median(pd_valid))/(2*np.pi)*100:.2f}% of a cycle")

    # ------- complex coherence over a small window -------
    print("\nComputing windowed complex coherence (boxcar 11x11)...")
    coh = _coh_window(slc_loc, slc_jpl, win=11)
    coh_valid = coh[valid]
    print(f"  median   {np.median(coh_valid):.4f}")
    print(f"  mean     {coh_valid.mean():.4f}")
    print(f"  5%-95%   [{np.percentile(coh_valid, 5):.4f}, {np.percentile(coh_valid, 95):.4f}]")

    # ------- plot -------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # row 1: amplitude
    pmin, pmax = np.percentile(amp_jpl[valid], [2, 98])
    axes[0, 0].imshow(amp_loc, vmin=pmin, vmax=pmax, cmap="gray")
    axes[0, 0].set_title("Amplitude: local")
    axes[0, 1].imshow(amp_jpl, vmin=pmin, vmax=pmax, cmap="gray")
    axes[0, 1].set_title("Amplitude: JPL")
    rat = np.where(valid, amp_loc / np.maximum(amp_jpl, 1e-12), np.nan)
    im_r = axes[0, 2].imshow(rat, vmin=0.5, vmax=1.5, cmap="RdBu_r")
    axes[0, 2].set_title("Amplitude ratio (local / JPL)")
    plt.colorbar(im_r, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # row 2: phase + coherence
    axes[1, 0].imshow(phase_diff, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    axes[1, 0].set_title("Phase difference (local − JPL, rad)")
    im_c = axes[1, 1].imshow(coh, vmin=0, vmax=1, cmap="viridis")
    axes[1, 1].set_title("Complex coherence (11×11 window)")
    plt.colorbar(im_c, ax=axes[1, 1], fraction=0.046, pad=0.04)
    # phase histogram
    axes[1, 2].hist(pd_valid, bins=200, range=(-np.pi, np.pi), color="steelblue")
    axes[1, 2].set_xlabel("Phase difference (rad)")
    axes[1, 2].set_ylabel("Pixel count")
    axes[1, 2].set_title(f"Phase diff distribution\n(median = {np.median(pd_valid):+.3f} rad)")

    for ax in axes[:, :2].ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Local GSLC vs JPL X05009 GSLC (HH, 80×80 km AOI)\n"
        f"phase median = {np.median(pd_valid):+.4f} rad  "
        f"phase std = {pd_valid.std():.4f} rad  "
        f"coh = {np.median(coh_valid):.3f}"
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
