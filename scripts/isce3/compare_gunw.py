"""Compare the locally-produced GUNW against the JPL-produced reference GUNW.

Compares unwrapped phase, coherence, and split-spectrum ionosphere phase
screen on the matched valid mask. Prints per-pixel mean / std / rms residuals.

Usage:
    /Users/zmhoppinen/miniforge3/envs/nisar_pytools/bin/python scripts/isce3/compare_gunw.py
"""

from pathlib import Path

import h5py
import numpy as np

OURS = Path("scripts/isce3/output/product.h5")
JPL = Path(
    "local/gunws/"
    "NISAR_L2_PR_GUNW_004_077_A_024_005_4000_SH_"
    "20251103T124615_20251103T124650_"
    "20251115T124615_20251115T124650_"
    "X05010_N_F_J_001.h5"
)

POL = "HH"
FREQ = "frequencyA"
UNWRAPPED_BASE = f"science/LSAR/GUNW/grids/{FREQ}/unwrappedInterferogram/{POL}"


def _read_band(h5_path, dataset_path):
    with h5py.File(h5_path, "r") as f:
        return f[dataset_path][...]


def _read_grid(h5_path, freq):
    """Return x, y coords of the unwrappedInterferogram grid for one product."""
    base = f"science/LSAR/GUNW/grids/{freq}/unwrappedInterferogram"
    with h5py.File(h5_path, "r") as f:
        x = f[f"{base}/xCoordinates"][...]
        y = f[f"{base}/yCoordinates"][...]
    return x, y


def _crop_to_bbox(arr, x_full, y_full, x_sub, y_sub, atol=1.0):
    """Crop arr (on full-grid x/y) to the sub-grid (x_sub/y_sub)."""
    ix_lo = int(np.argmin(np.abs(x_full - x_sub[0])))
    ix_hi = int(np.argmin(np.abs(x_full - x_sub[-1])))
    iy_lo = int(np.argmin(np.abs(y_full - y_sub[0])))
    iy_hi = int(np.argmin(np.abs(y_full - y_sub[-1])))
    ix0, ix1 = sorted((ix_lo, ix_hi))
    iy0, iy1 = sorted((iy_lo, iy_hi))
    sub = arr[iy0:iy1 + 1, ix0:ix1 + 1]
    # Sanity: matched corner positions
    if abs(x_full[ix_lo] - x_sub[0]) > atol or abs(y_full[iy_lo] - y_sub[0]) > atol:
        print(f"    (warning: corner mismatch >{atol} m -- grids may be offset)")
    return sub


def _residual_summary(label, ours, ref):
    valid = np.isfinite(ours) & np.isfinite(ref)
    if not valid.any():
        print(f"  {label:30s}  no overlapping valid pixels")
        return
    res = ours[valid] - ref[valid]
    print(f"  {label:30s}  n={valid.sum():>8,d}  "
          f"mean={res.mean():+8.4f}  std={res.std():7.4f}  "
          f"rms={np.sqrt((res**2).mean()):7.4f}")


def main():
    if not OURS.exists():
        raise SystemExit(f"Local GUNW not found: {OURS}")
    if not JPL.exists():
        raise SystemExit(f"JPL GUNW not found: {JPL}")

    print(f"Local GUNW: {OURS}")
    print(f"JPL  GUNW : {JPL}\n")

    # Local product is cropped to AOI; JPL is full scene. Use ours to drive the crop.
    x_ours, y_ours = _read_grid(OURS, FREQ)
    x_jpl, y_jpl = _read_grid(JPL, FREQ)
    print(f"  ours grid: {len(y_ours)} x {len(x_ours)} pixels  "
          f"x=[{x_ours[0]:.0f},{x_ours[-1]:.0f}]  y=[{y_ours[0]:.0f},{y_ours[-1]:.0f}]")
    print(f"  JPL  grid: {len(y_jpl)} x {len(x_jpl)} pixels  "
          f"x=[{x_jpl[0]:.0f},{x_jpl[-1]:.0f}]  y=[{y_jpl[0]:.0f},{y_jpl[-1]:.0f}]\n")

    bands = [
        ("unwrapped_phase",   f"{UNWRAPPED_BASE}/unwrappedPhase"),
        ("coherence",         f"{UNWRAPPED_BASE}/coherenceMagnitude"),
        ("connectedComps",    f"{UNWRAPPED_BASE}/connectedComponents"),
        ("ionosphere",        f"{UNWRAPPED_BASE}/ionospherePhaseScreen"),
        ("ionosphere_uncert", f"{UNWRAPPED_BASE}/ionospherePhaseScreenUncertainty"),
    ]

    for name, path in bands:
        try:
            ours = _read_band(OURS, path).astype(np.float32)
            ref_full = _read_band(JPL, path).astype(np.float32)
        except KeyError:
            print(f"  {name:30s}  (missing in one of the products)")
            continue

        ref = _crop_to_bbox(ref_full, x_jpl, y_jpl, x_ours, y_ours)
        if ours.shape != ref.shape:
            print(f"  {name:30s}  shape mismatch after crop: "
                  f"ours={ours.shape}, ref={ref.shape}")
            continue

        _residual_summary(name, ours, ref)


if __name__ == "__main__":
    main()
