"""DEM fetching utilities for NISAR data."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer


def fetch_dem(
    source: str | os.PathLike | xr.DataTree | xr.DataArray,
    out_path: str | os.PathLike | None = None,
    dem_name: str = "glo_30",
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
        DEM product name for ``dem_stitcher``. Default ``"glo_30"``
        (Copernicus GLO-30).
    buffer : float
        Buffer in degrees around the extent. Default 0.05.

    Returns
    -------
    xr.DataArray
        DEM elevation data with CRS set.
    """
    from dem_stitcher import stitch_dem

    bounds = _get_bounds_latlon(source, buffer)

    dem_arr, profile = stitch_dem(
        bounds,
        dem_name=dem_name,
        dst_ellipsoidal_height=False,
        dst_area_or_point="Point",
    )

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
        attrs={"units": "m", "long_name": "Elevation above geoid", "dem_name": dem_name},
    )
    epsg = profile.get("crs", "EPSG:4326")
    da = da.rio.write_crs(epsg)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        da.rio.to_raster(str(out_path))

    return da


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
    """Extract bounds from an HDF5 file by finding the first group with coordinates."""
    with h5py.File(path, "r") as f:
        x, y, epsg = _find_coords_in_h5(f)

    return _project_bounds(x, y, epsg, buffer)


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
