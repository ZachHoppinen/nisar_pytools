"""Compute local incidence angle from LOS vectors and a DEM.

Local incidence angle is the angle between the radar line-of-sight vector
(target-to-sensor) and the local surface normal derived from a DEM.
"""

from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


def compute_surface_normal(
    dem: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unit surface normal vectors from a DEM.

    Uses ``np.gradient`` on the elevation to get slope in x and y,
    then constructs and normalizes the surface normal in an
    east-north-up (ENU) frame: ``n = (-dz/dx, -dz/dy, 1)``.

    Parameters
    ----------
    dem : xr.DataArray
        2D elevation array with ``x`` and ``y`` coordinates.
        ``y`` may be ascending or descending.

    Returns
    -------
    nx, ny, nz : np.ndarray
        Unit surface normal components (east, north, up).
    """
    elev = dem.values
    x = dem.x.values
    y = dem.y.values

    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])

    # np.gradient: axis 0 = y (rows), axis 1 = x (cols)
    # Negate dy because y is typically descending (north-to-south)
    dz_dy, dz_dx = np.gradient(elev, -dy, dx)

    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(elev)

    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= mag
    ny /= mag
    nz /= mag

    return nx, ny, nz


def interpolate_los_to_dem(
    dem: xr.DataArray,
    los_x: np.ndarray,
    los_y: np.ndarray,
    los_z: np.ndarray,
    heights: np.ndarray,
    x_rg: np.ndarray,
    y_rg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate 3D LOS vectors from the radar grid to the DEM grid.

    The LOS vectors are defined on a ``(height, y, x)`` grid. This function
    interpolates them to the DEM pixel locations using the DEM elevation
    for the height axis.

    Parameters
    ----------
    dem : xr.DataArray
        2D elevation DataArray with ``x`` and ``y`` coordinates.
    los_x, los_y, los_z : np.ndarray
        LOS unit vector components, shape ``(n_heights, n_y, n_x)``.
    heights : np.ndarray
        Height values for the LOS grid, shape ``(n_heights,)``.
    x_rg, y_rg : np.ndarray
        Coordinate arrays for the LOS grid.

    Returns
    -------
    los_e, los_n, los_u : np.ndarray
        Interpolated LOS components on the DEM grid (east, north, up).
    """
    dem_x = dem.x.values
    dem_y = dem.y.values
    dem_elev = dem.values

    # RegularGridInterpolator requires ascending axes
    y_sorted = y_rg if y_rg[0] < y_rg[-1] else y_rg[::-1]
    flip_y = y_rg[0] > y_rg[-1]

    yy, xx = np.meshgrid(dem_y, dem_x, indexing="ij")
    pts = np.column_stack([dem_elev.ravel(), yy.ravel(), xx.ravel()])

    results = {}
    for name, data in [("x", los_x), ("y", los_y), ("z", los_z)]:
        if flip_y:
            data = data[:, ::-1, :]
        interp = RegularGridInterpolator(
            (heights, y_sorted, x_rg),
            data,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        results[name] = interp(pts).reshape(dem_elev.shape)

    return results["x"], results["y"], results["z"]


def local_incidence_angle(
    dem: xr.DataArray,
    los_x: np.ndarray,
    los_y: np.ndarray,
    los_z: np.ndarray,
    heights: np.ndarray,
    x_rg: np.ndarray,
    y_rg: np.ndarray,
    epsg: int | None = None,
) -> xr.DataArray:
    """Compute local incidence angle from LOS vectors and a DEM.

    Parameters
    ----------
    dem : xr.DataArray
        2D elevation DataArray with ``x`` and ``y`` coordinates.
        Should be in the same CRS as the LOS grid.
    los_x, los_y, los_z : np.ndarray
        LOS unit vector components (east, north, up),
        shape ``(n_heights, n_y, n_x)``.
    heights : np.ndarray
        Height values for the LOS grid.
    x_rg, y_rg : np.ndarray
        Coordinate arrays for the LOS grid.
    epsg : int, optional
        EPSG code to assign to the output. If ``None``, uses the DEM's CRS.

    Returns
    -------
    xr.DataArray
        Local incidence angle in degrees on the DEM grid.
    """
    nx, ny, nz = compute_surface_normal(dem)
    los_e, los_n, los_u = interpolate_los_to_dem(
        dem, los_x, los_y, los_z, heights, x_rg, y_rg
    )

    dot = los_e * nx + los_n * ny + los_u * nz
    dot = np.clip(dot, -1, 1)
    lia = np.degrees(np.arccos(dot)).astype(np.float32)

    lia_da = xr.DataArray(
        lia,
        dims=["y", "x"],
        coords={"y": dem.y.values, "x": dem.x.values},
        name="local_incidence_angle",
        attrs={"units": "degrees", "long_name": "Local incidence angle"},
    )

    crs = epsg if epsg is not None else dem.rio.crs
    if crs is not None:
        lia_da = lia_da.rio.write_crs(crs)
        lia_da = lia_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    return lia_da
