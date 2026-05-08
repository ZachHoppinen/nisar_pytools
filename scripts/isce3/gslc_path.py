"""GSLC x GSLC -> GUNW-like product on the same AOI as the isce3 RSLC run.

Pedagogical contrast script. Where the RSLC->GUNW pipeline (run via
nisar.workflows.insar) goes through 16+ stages of geometry, coregistration,
multilook, filter, snaphu, ionosphere split-spectrum, and geocoding,
the GSLC->GUNW path collapses to:

    interferogram = antialiased_crossmul(slc1, slc2)   # both already on same UTM grid
    multilook by 16x16                                  # match the GUNW 80 m posting
    coherence over a small window
    snaphu unwrap

That's it. No coregistration (already geocoded), no ionosphere correction
(needs RSLC sub-bands), no troposphere, no SET. Runs in seconds on the
80 x 80 km AOI.

Antialias choice: ``antialias='2d'`` is the geometrically correct option
for GSLCs. The SAR spectrum's bandlimit is rotated diagonally in (kx, ky)
because geocoding rotates radar (range, azimuth) onto map (x, y), so
neither x nor y is privileged. The ISCE3 production crossmul is range-only
(``antialias='range'``), which is right for RSLCs but only partially
suppresses aliasing on a GSLC. After 16x16 multilooking the difference
is below noise floor either way.

No flat-earth removal step: NISAR GSLCs are already deramped during
geocoding (the geocoder subtracts the carrier phase ``-4 pi R / lambda``
at each ground point). So the conjugate product of two GSLCs has the
orbit-induced flat-earth phase already cancelled -- subtracting an
additional ``4 pi / lambda * B_par`` would over-correct. The small
residual versus the JPL GUNW (~0.3 rad N/S ramp) is from solid-earth
tides, troposphere, and ionosphere -- corrections that JPL applies and
the GSLC path can't reach.

Output: scripts/isce3/output/gslc_path.h5
"""

from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from nisar_pytools import open_nisar
from nisar_pytools.processing import coherence, interferogram, multilook, unwrap

# Same AOI as the isce3 runconfig's geocode bbox (UTM 11N, 80 km square).
TOP_LEFT_X, TOP_LEFT_Y = 464000.0, 4964080.0
BOT_RIGHT_X, BOT_RIGHT_Y = 544000.0, 4884080.0

# Match the JPL GUNW unwrappedInterferogram grid: 80 m posting.
# GSLC native posting is 5 m, so multilook 16 x 16 -> 80 m.
LOOKS = 16

# Coherence window (boxcar in native GSLC pixels); 11 = 55 m, ~half a look cell.
COH_WINDOW = 11

GSLC_REF = "local/gslcs/NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001.h5"
GSLC_SEC = "local/gslcs/NISAR_L2_PR_GSLC_005_077_A_024_4005_DHDH_A_20251115T124615_20251115T124650_X05009_N_F_J_001.h5"

OUT_PATH = Path("scripts/isce3/output/gslc_path.h5")


def _open_chip(path: str, pol: str = "HH") -> xr.DataArray:
    """Open a GSLC and crop to the shared AOI, materializing the chip
    before the DataTree (which owns the h5py file handle) goes out of
    scope. Returns an in-memory DataArray with no h5py references.
    """
    dt = open_nisar(path)
    slc = (
        dt["science/LSAR/GSLC/grids/frequencyA"]
        .dataset[pol]
        .sel(x=slice(TOP_LEFT_X, BOT_RIGHT_X), y=slice(TOP_LEFT_Y, BOT_RIGHT_Y))
    )
    # Compute now while the underlying h5py file handle is still alive.
    arr = np.asarray(slc.values)
    return xr.DataArray(
        arr,
        dims=slc.dims,
        coords={d: slc.coords[d].values for d in slc.dims if d in slc.coords},
        name=slc.name,
        attrs=dict(slc.attrs),
    )


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Opening GSLCs and cropping to AOI...")
    slc1 = _open_chip(GSLC_REF)
    slc2 = _open_chip(GSLC_SEC)
    print(f"  chip shape: {slc1.shape}  posting: 5 m")
    if slc1.shape != slc2.shape:
        raise SystemExit(f"GSLCs are not co-registered: {slc1.shape} vs {slc2.shape}")

    print("Forming interferogram (antialias='2d', symmetric for GSLCs)...")
    ifg = interferogram(slc1, slc2, antialias="2d")

    print(f"Multilooking by {LOOKS} x {LOOKS} (-> {5 * LOOKS} m posting)...")
    ml = multilook(ifg, looks_y=LOOKS, looks_x=LOOKS)
    print(f"  multilooked shape: {ml.shape}")

    print(f"Estimating coherence (boxcar, window={COH_WINDOW})...")
    coh = coherence(slc1, slc2, window_size=COH_WINDOW, method="boxcar")
    coh_ml = multilook(coh, looks_y=LOOKS, looks_x=LOOKS)

    print("Unwrapping with snaphu...")
    # nlooks roughly = product of multilook factors x coherence-window factor
    unw, conncomp = unwrap(ml, coh_ml, nlooks=float(LOOKS * LOOKS))

    print(f"Writing {OUT_PATH}")
    with h5py.File(OUT_PATH, "w") as f:
        f.attrs["source"] = "GSLC x GSLC -> GUNW-like, nisar_pytools"
        f.attrs["aoi_top_left"] = (TOP_LEFT_X, TOP_LEFT_Y)
        f.attrs["aoi_bot_right"] = (BOT_RIGHT_X, BOT_RIGHT_Y)
        f.attrs["multilook"] = LOOKS
        f.attrs["posting_m"] = 5 * LOOKS

        f.create_dataset("xCoordinates", data=np.asarray(ml.x))
        f.create_dataset("yCoordinates", data=np.asarray(ml.y))
        f.create_dataset("wrappedInterferogram", data=np.asarray(ml).astype(np.complex64))
        f.create_dataset("unwrappedPhase", data=np.asarray(unw).astype(np.float32))
        f.create_dataset("coherenceMagnitude", data=np.asarray(coh_ml).astype(np.float32))
        f.create_dataset("connectedComponents", data=np.asarray(conncomp).astype(np.uint16))

    print("Done.")


if __name__ == "__main__":
    main()
