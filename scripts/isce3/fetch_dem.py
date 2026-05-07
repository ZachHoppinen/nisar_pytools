"""Fetch a Copernicus 30m DEM VRT covering the JPL GUNW geocode bbox.

The bbox comes from the official runconfig
(``processing.geocode.{top_left,bottom_right}``) which is in
EPSG:32611 (UTM 11N). dem_stitcher needs WGS84 lon/lat, so we
reproject the corners first.

Output: scripts/isce3/inputs/dem.tif
"""

from pathlib import Path

import numpy as np
from dem_stitcher.stitcher import stitch_dem
from pyproj import Transformer
import rasterio

OUT_DIR = Path("scripts/isce3/inputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "dem.tif"

# JPL runconfig bbox (UTM 11N)
TOP_LEFT_X, TOP_LEFT_Y = 324000.0, 5101920.0
BOT_RIGHT_X, BOT_RIGHT_Y = 684000.0, 4746240.0
SRC_EPSG = 32611


def main():
    # Reproject UTM corners to WGS84 lon/lat
    tf = Transformer.from_crs(f"EPSG:{SRC_EPSG}", "EPSG:4326", always_xy=True)
    lon_min, lat_max = tf.transform(TOP_LEFT_X, TOP_LEFT_Y)
    lon_max, lat_min = tf.transform(BOT_RIGHT_X, BOT_RIGHT_Y)
    # Pad by ~0.05deg so geocoding edges aren't NaN
    pad = 0.05
    bounds = [lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad]
    print(f"WGS84 bbox: {bounds}")

    print("Fetching glo_30 (Copernicus 30m) DEM tiles...")
    arr, p = stitch_dem(
        bounds, dem_name="glo_30", dst_ellipsoidal_height=True, dst_area_or_point="Point"
    )

    print(f"DEM shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Valid range: [{np.nanmin(arr):.1f}, {np.nanmax(arr):.1f}] m")

    with rasterio.open(
        OUT_PATH,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=p["crs"],
        transform=p["transform"],
        nodata=p.get("nodata"),
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(arr, 1)

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
