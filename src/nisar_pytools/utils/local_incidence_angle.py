"""Compute local incidence angle from LOS vectors and a DEM.

Local incidence angle is the angle between the radar line-of-sight vector
(target-to-sensor) and the local surface normal derived from a DEM.

.. note::
    The DEM must be in a **projected CRS** (metres) for the surface normal
    computation to be physically correct. Geographic CRS (degrees) will
    produce wrong normals because the gradient mixes elevation (metres)
    with horizontal (degrees). Reproject to UTM first if needed.
"""

from __future__ import annotations

import logging

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

from nisar_pytools.processing.atmospheric import _interpolate_3d_to_2d

log = logging.getLogger(__name__)


def compute_surface_normal(
    dem: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unit surface normal vectors from a DEM.

    Uses ``np.gradient`` on the elevation to get slope in x and y,
    then constructs and normalizes the surface normal in an
    east-north-up (ENU) frame: ``n = (-dz/dx, -dz/dy, 1)``.

    The DEM must be in a projected CRS (metres) for correct results.

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

    # Use signed spacing so gradient handles both ascending and descending y
    dy_signed = y[1] - y[0]  # negative if descending, positive if ascending
    dx_signed = x[1] - x[0]

    # np.gradient: axis 0 = y (rows), axis 1 = x (cols)
    dz_dy, dz_dx = np.gradient(elev, dy_signed, dx_signed)

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

    Uses the shared ``_interpolate_3d_to_2d`` utility which handles
    monotonicity sorting for all three axes (height, y, x).

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
    results = {}
    for name, data in [("x", los_x), ("y", los_y), ("z", los_z)]:
        interp_da = _interpolate_3d_to_2d(data, heights, y_rg, x_rg, dem)
        results[name] = np.asarray(interp_da)

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
        Must be in a **projected CRS** (metres) matching the LOS grid.
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
    # Validate LOS vectors are approximately unit length
    los_mag = np.sqrt(los_x**2 + los_y**2 + los_z**2)
    if not np.allclose(los_mag[np.isfinite(los_mag)], 1.0, atol=0.05):
        log.warning(
            "LOS vectors are not unit length (mean magnitude: %.3f). "
            "Results may be inaccurate.",
            np.nanmean(los_mag),
        )

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

    # Normalize CRS type
    if epsg is not None:
        lia_da = lia_da.rio.write_crs(int(epsg))
    elif dem.rio.crs is not None:
        lia_da = lia_da.rio.write_crs(dem.rio.crs)

    if lia_da.rio.crs is not None:
        lia_da = lia_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    return lia_da
