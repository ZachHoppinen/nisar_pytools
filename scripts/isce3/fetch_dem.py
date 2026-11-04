"""Fetch the NISAR DEM covering the JPL GUNW geocode bbox.

The bbox comes from the official runconfig
(``processing.geocode.{top_left,bottom_right}``) which is in
EPSG:32611 (UTM 11N). ``fetch_dem`` reprojects the corners to WGS84.

Output: scripts/isce3/inputs/dem.tif
"""

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from nisar_pytools.utils.dem import fetch_dem

OUT_DIR = Path("scripts/isce3/inputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "dem.tif"

# JPL runconfig bbox (UTM 11N)
TOP_LEFT_X, TOP_LEFT_Y = 324000.0, 5101920.0
BOT_RIGHT_X, BOT_RIGHT_Y = 684000.0, 4746240.0
SRC_EPSG = 32611


def main():
    # Corner-only DataArray so fetch_dem can read the bbox and its CRS
    corners = xr.DataArray(
        np.zeros((2, 2)),
        dims=["y", "x"],
        coords={"y": [TOP_LEFT_Y, BOT_RIGHT_Y], "x": [TOP_LEFT_X, BOT_RIGHT_X]},
    ).rio.write_crs(SRC_EPSG)

    # Pad by ~0.05deg so geocoding edges aren't NaN
    print("Fetching NISAR DEM...")
    dem = fetch_dem(corners, out_path=OUT_PATH, buffer=0.05)

    print(f"DEM shape: {dem.shape}, dtype: {dem.dtype}")
    print(f"Valid range: [{float(dem.min()):.1f}, {float(dem.max()):.1f}] m")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
