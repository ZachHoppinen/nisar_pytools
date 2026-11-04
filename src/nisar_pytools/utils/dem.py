"""DEM fetching utilities for NISAR data."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer

# Public NISAR DEM v1.2 mosaic hosted by ASF (Earthdata Login required).
NISAR_DEM_URL = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/DEM/v1.2/EPSG4326/EPSG4326.vrt"


def fetch_dem(
    source: str | os.PathLike | xr.DataTree | xr.DataArray,
    out_path: str | os.PathLike | None = None,
    dem_name: str = "nisar_dem",
    buffer: float = 0.05,
) -> xr.DataArray:
    """Download a DEM covering a NISAR product's extent.

    Parameters
    ----------
    source : str, Path, DataTree, or DataArray
        One of:
        - Path to a NISAR HDF5 file
        - An opened DataTree from :func:`open_nisar`
        - A DataArray with ``x``/``y`` coordinates and CRS
    out_path : str or Path, optional
        If provided, save the DEM as a GeoTIFF at this path.
    dem_name : str
        ``"nisar_dem"`` (default) reads the NISAR DEM v1.2 from ASF, the
        Copernicus-derived DEM used in NISAR production, with heights above
        the WGS84 ellipsoid. Needs Earthdata Login credentials in
        ``~/.netrc``. Any other name (e.g. ``"glo_30"``) is passed to
        ``dem_stitcher`` and returns heights above the geoid.
    buffer : float
        Buffer in degrees around the extent. Default 0.05.

    Returns
    -------
    xr.DataArray
        DEM elevation data with CRS set.
    """
    bounds = _get_bounds_latlon(source, buffer)

    if dem_name == "nisar_dem":
        dem_arr, profile = _read_nisar_dem(bounds)
        long_name = "Elevation above WGS84 ellipsoid"
    else:
        from dem_stitcher import stitch_dem

        dem_arr, profile = stitch_dem(
            bounds,
            dem_name=dem_name,
            dst_ellipsoidal_height=False,
            dst_area_or_point="Point",
        )
        long_name = "Elevation above geoid"

    # Build DataArray from the returned numpy array + rasterio profile
    ny, nx = dem_arr.shape[-2:]
    transform = profile["transform"]
    x = np.arange(nx) * transform.a + transform.c + transform.a / 2
    y = np.arange(ny) * transform.e + transform.f + transform.e / 2

    # dem_arr may be (1, ny, nx) or (ny, nx)
    if dem_arr.ndim == 3:
        dem_arr = dem_arr[0]

    da = xr.DataArray(
        dem_arr.astype(np.float32),
        dims=["y", "x"],
        coords={"y": y, "x": x},
        name="elevation",
        attrs={"units": "m", "long_name": long_name, "dem_name": dem_name},
    )
    epsg = profile.get("crs", "EPSG:4326")
    da = da.rio.write_crs(epsg)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        da.rio.to_raster(str(out_path))

    return da


def _read_nisar_dem(bounds: list[float]) -> tuple[np.ndarray, dict]:
    """Read the NISAR DEM over [lon_min, lat_min, lon_max, lat_max].

    The window is snapped outward to whole DEM pixels, so the native grid is
    kept with no resampling.
    """
    import rasterio
    from rasterio.windows import Window

    lon_min, lat_min, lon_max, lat_max = bounds
    # Earthdata Login redirects need a cookie jar to persist the session.
    cookies = os.path.join(tempfile.gettempdir(), "nisar_pytools_edl_cookies.txt")
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_COOKIEFILE=cookies,
        GDAL_HTTP_COOKIEJAR=cookies,
    )
    try:
        with env, rasterio.open(f"/vsicurl/{NISAR_DEM_URL}") as src:
            # Row/col of the pixels holding the top-left and bottom-right corners
            row0, col0 = src.index(lon_min, lat_max, op=math.floor)
            row1, col1 = src.index(lon_max, lat_min, op=math.floor)
            window = Window.from_slices((row0, row1 + 1), (col0, col1 + 1))
            dem_arr = src.read(1, window=window)
            profile = {"transform": src.window_transform(window), "crs": src.crs}
    except rasterio.errors.RasterioIOError as e:
        raise RuntimeError(
            f"Could not read the NISAR DEM from {NISAR_DEM_URL}. It needs "
            "Earthdata Login credentials for urs.earthdata.nasa.gov in ~/.netrc."
        ) from e
    return dem_arr, profile


def _get_bounds_latlon(
    source: str | os.PathLike | xr.DataTree | xr.DataArray,
    buffer: float,
) -> list[float]:
    """Extract [lon_min, lat_min, lon_max, lat_max] from various source types."""
    if isinstance(source, (str, os.PathLike)):
        return _bounds_from_h5(Path(source), buffer)
    elif isinstance(source, xr.DataTree):
        return _bounds_from_datatree(source, buffer)
    elif isinstance(source, xr.DataArray):
        return _bounds_from_dataarray(source, buffer)
    else:
        raise TypeError(
            f"source must be a file path, DataTree, or DataArray, got {type(source)}"
        )


def _bounds_from_h5(path: Path, buffer: float) -> list[float]:
    """Extract bounds from an HDF5 file.

    Tries two sources in order:

    1. ``science/LSAR/identification/boundingPolygon`` -- a WGS84 WKT polygon
       present in every NISAR product family (RSLC, GSLC, GUNW, ...). This is
       what we want for radar-grid products that have no projected x/y coords.
    2. The first group containing ``xCoordinates`` and ``yCoordinates``
       (geocoded products only: GSLC, GUNW). Used as a fallback so existing
       behaviour on geocoded inputs is unchanged.
    """
    bounds = _bounds_from_h5_polygon(path, buffer)
    if bounds is not None:
        return bounds

    with h5py.File(path, "r") as f:
        x, y, epsg = _find_coords_in_h5(f)
    return _project_bounds(x, y, epsg, buffer)


def _bounds_from_h5_polygon(path: Path, buffer: float) -> list[float] | None:
    """Read NISAR ``identification/boundingPolygon`` (WGS84 WKT) and return
    [lon_min, lat_min, lon_max, lat_max] padded by ``buffer`` degrees.

    Returns ``None`` if the field is absent (e.g. non-NISAR HDF5 files).
    """
    from shapely import wkt

    candidate_paths = (
        "science/LSAR/identification/boundingPolygon",
        "science/SSAR/identification/boundingPolygon",
    )
    with h5py.File(path, "r") as f:
        for p in candidate_paths:
            if p in f:
                raw = f[p][()]
                wkt_str = raw.decode() if isinstance(raw, bytes) else raw
                poly = wkt.loads(wkt_str)
                lon_min, lat_min, lon_max, lat_max = poly.bounds
                return [
                    lon_min - buffer, lat_min - buffer,
                    lon_max + buffer, lat_max + buffer,
                ]
    return None


def _find_coords_in_h5(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, int]:
    """Recursively find xCoordinates/yCoordinates and projection in an HDF5 file."""
    if "xCoordinates" in group and "yCoordinates" in group:
        x = group["xCoordinates"][()]
        y = group["yCoordinates"][()]
        epsg = 4326
        if "projection" in group:
            proj = group["projection"]
            if "epsg_code" in proj.attrs:
                epsg = int(proj.attrs["epsg_code"])
            else:
                epsg = int(proj[()])
        return x, y, epsg

    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            try:
                return _find_coords_in_h5(item)
            except ValueError:
                continue

    raise ValueError("No xCoordinates/yCoordinates found in HDF5 file")


def _bounds_from_datatree(dt: xr.DataTree, buffer: float) -> list[float]:
    """Extract bounds from a DataTree by finding the first node with x/y coords."""
    for node in dt.subtree:
        ds = node.dataset
        if "x" in ds.coords and "y" in ds.coords and len(ds.coords["x"]) > 1:
            da = next(iter(ds.data_vars.values())) if ds.data_vars else None
            if da is not None and hasattr(da, "rio") and da.rio.crs is not None:
                epsg = da.rio.crs.to_epsg()
            else:
                epsg = int(ds.attrs.get("projection", 4326))
            return _project_bounds(
                ds.coords["x"].values, ds.coords["y"].values, epsg, buffer
            )
    raise ValueError("No node with x/y coordinates found in DataTree")


def _bounds_from_dataarray(da: xr.DataArray, buffer: float) -> list[float]:
    """Extract bounds from a DataArray with x/y coords and CRS."""
    if "x" not in da.coords or "y" not in da.coords:
        raise ValueError("DataArray must have 'x' and 'y' coordinates")

    epsg = 4326
    if da.rio.crs is not None:
        epsg = da.rio.crs.to_epsg()

    return _project_bounds(da.x.values, da.y.values, epsg, buffer)


def _project_bounds(
    x: np.ndarray, y: np.ndarray, epsg: int, buffer: float
) -> list[float]:
    """Project x/y bounds to lat/lon and add buffer."""
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(float(x.min()), float(y.min()))
    lon_max, lat_max = transformer.transform(float(x.max()), float(y.max()))

    # Ensure min < max
    lon_min, lon_max = min(lon_min, lon_max), max(lon_min, lon_max)
    lat_min, lat_max = min(lat_min, lat_max), max(lat_min, lat_max)

    return [lon_min - buffer, lat_min - buffer, lon_max + buffer, lat_max + buffer]
