"""Perpendicular and parallel baseline computation between two GSLC acquisitions.

Computes the spatial baseline between two SAR acquisitions from their
orbit state vectors, decomposed into components perpendicular and
parallel to the line of sight.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import interp1d


def compute_baseline(
    dt_reference: xr.DataTree,
    dt_secondary: xr.DataTree,
) -> xr.Dataset:
    """Compute perpendicular and parallel baselines between two GSLC DataTrees.

    Uses orbit state vectors to compute the baseline in ECEF, then
    decomposes into perpendicular and parallel components relative to
    the line of sight (computed geometrically from satellite position
    and ground point location).

    Parameters
    ----------
    dt_reference : xr.DataTree
        Reference (primary) GSLC DataTree from :func:`open_nisar`.
    dt_secondary : xr.DataTree
        Secondary GSLC DataTree.

    Returns
    -------
    xr.Dataset
        Contains ``perpendicular_baseline`` and ``parallel_baseline``
        DataArrays in meters, on the radarGrid coordinates.
    """
    # Extract orbit data
    ref_pos, ref_vel, ref_time = _extract_orbit(dt_reference)
    sec_pos, sec_vel, sec_time = _extract_orbit(dt_secondary)

    # Extract radarGrid metadata from reference
    rg = dt_reference["science/LSAR/GSLC/metadata/radarGrid"].dataset

    # Select middle height layer
    if "z" in rg.dims:
        mid = rg.sizes["z"] // 2
        rg_2d = rg.isel(z=mid)
    else:
        rg_2d = rg

    az_time = np.asarray(rg_2d["zeroDopplerAzimuthTime"])
    x_rg = np.asarray(rg.coords["x"])
    y_rg = np.asarray(rg.coords["y"])

    # Get EPSG for coordinate conversion
    epsg = int(rg.attrs.get("projection", 32611))

    # Convert ground points from projected coords to ECEF
    ground_ecef = _grid_to_ecef(x_rg, y_rg, epsg)  # (ny, nx, 3)

    # Interpolate satellite positions to azimuth times
    ref_sat = _interpolate_orbit(ref_pos, ref_time, az_time)  # (ny, nx, 3)
    sec_sat = _interpolate_orbit(sec_pos, sec_time, az_time)  # (ny, nx, 3)

    # Baseline vector in ECEF
    baseline = sec_sat - ref_sat  # (ny, nx, 3)

    # LOS direction: ground - satellite (target-to-sensor would be satellite - ground,
    # but the sign doesn't matter for the decomposition magnitude)
    los_vec = ref_sat - ground_ecef  # (ny, nx, 3)
    los_mag = np.sqrt(np.sum(los_vec**2, axis=-1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        los_hat = np.where(los_mag > 0, los_vec / los_mag, 0)

    # Along-track direction from velocity
    ref_vel_interp = _interpolate_orbit(ref_vel, ref_time, az_time)
    vel_mag = np.sqrt(np.sum(ref_vel_interp**2, axis=-1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        along_hat = np.where(vel_mag > 0, ref_vel_interp / vel_mag, 0)

    # Cross-track direction = along × LOS (perpendicular to both)
    cross_hat = np.cross(along_hat, los_hat)
    cross_mag = np.sqrt(np.sum(cross_hat**2, axis=-1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_hat = np.where(cross_mag > 0, cross_hat / cross_mag, 0)

    # Perpendicular baseline: component of B perpendicular to LOS, in the cross-track plane
    b_perp = np.sum(baseline * cross_hat, axis=-1)

    # Parallel baseline: component along LOS
    b_par = np.sum(baseline * los_hat, axis=-1)

    coords = {"y": y_rg, "x": x_rg}
    return xr.Dataset(
        {
            "perpendicular_baseline": xr.DataArray(
                b_perp.astype(np.float32),
                dims=["y", "x"],
                coords=coords,
                attrs={"units": "meters", "long_name": "Perpendicular baseline"},
            ),
            "parallel_baseline": xr.DataArray(
                b_par.astype(np.float32),
                dims=["y", "x"],
                coords=coords,
                attrs={"units": "meters", "long_name": "Parallel baseline"},
            ),
        }
    )


def _extract_orbit(dt: xr.DataTree) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract orbit position, velocity, and time arrays from a DataTree."""
    orbit = dt["science/LSAR/GSLC/metadata/orbit"].dataset

    # Position and velocity are 2D data vars (n_epochs, 3)
    position = np.asarray(orbit["position"]) if "position" in orbit else None
    velocity = np.asarray(orbit["velocity"]) if "velocity" in orbit else None

    # Time may be a 1D data var or stored in attrs (if 1D, it goes to attrs)
    if "time" in orbit:
        time = np.asarray(orbit["time"])
    elif "time" in orbit.attrs:
        time = np.asarray(orbit.attrs["time"])
    else:
        time = None

    if position is None or time is None:
        raise ValueError("Could not extract orbit data from DataTree")
    if velocity is None:
        raise ValueError("Could not extract velocity from DataTree")

    return position, velocity, time


def _interpolate_orbit(
    data: np.ndarray,
    time: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Interpolate orbit data (position or velocity) to query times.

    Returns array of shape (ny, nx, 3).
    """
    ny, nx = query_times.shape
    result = np.zeros((ny, nx, 3), dtype=np.float64)

    for i in range(3):
        f = interp1d(time, data[:, i], kind="cubic", fill_value="extrapolate")
        result[..., i] = f(query_times)

    return result


def _grid_to_ecef(
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    height: float = 0.0,
) -> np.ndarray:
    """Convert a projected coordinate grid to ECEF XYZ.

    Returns array of shape (ny, nx, 3).
    """
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4978", always_xy=True)

    xx, yy = np.meshgrid(x, y)
    hh = np.full_like(xx, height)

    # Transform projected → ECEF (EPSG:4978 is WGS84 ECEF)
    ecef_x, ecef_y, ecef_z = transformer.transform(xx, yy, hh)

    return np.stack([ecef_x, ecef_y, ecef_z], axis=-1)
