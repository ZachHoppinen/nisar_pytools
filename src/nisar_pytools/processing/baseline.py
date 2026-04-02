"""Perpendicular and parallel baseline computation between two GSLC acquisitions.

Computes the spatial baseline between two SAR acquisitions from their
orbit state vectors, decomposed into components perpendicular and
parallel to the line of sight.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import make_interp_spline

log = logging.getLogger(__name__)


def compute_baseline(
    dt_reference: xr.DataTree,
    dt_secondary: xr.DataTree,
    dem: xr.DataArray | None = None,
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
    dem : xr.DataArray, optional
        DEM on the radarGrid for accurate ground point heights. If
        ``None``, height=0 is used (introduces error in mountainous
        terrain).

    Returns
    -------
    xr.Dataset
        Contains ``perpendicular_baseline`` and ``parallel_baseline``
        DataArrays in meters, on the radarGrid coordinates.
    """
    # Extract orbit data
    ref_pos, ref_vel, ref_time = _extract_orbit(dt_reference)
    sec_pos, _, sec_time = _extract_orbit(dt_secondary)

    # Extract radarGrid metadata from reference
    rg = dt_reference["science/LSAR/GSLC/metadata/radarGrid"].dataset

    # Select middle height layer if 3D
    if "z" in rg.dims:
        mid = rg.sizes["z"] // 2
        rg_2d = rg.isel(z=mid)
    else:
        rg_2d = rg

    az_time = np.asarray(rg_2d["zeroDopplerAzimuthTime"])

    # Ensure az_time is 2D (ny, nx)
    x_rg = np.asarray(rg.coords["x"])
    y_rg = np.asarray(rg.coords["y"])
    if az_time.ndim == 1:
        # 1D azimuth time → broadcast to (ny, nx)
        az_time = np.broadcast_to(az_time[:, np.newaxis], (len(y_rg), len(x_rg)))

    # Get EPSG for coordinate conversion
    epsg = _extract_epsg_from_dataset(rg)

    # Convert ground points from projected coords to ECEF
    height = 0.0
    if dem is not None:
        height = dem
    ground_ecef = _grid_to_ecef(x_rg, y_rg, epsg, height=height)

    # Build interpolators once, reuse for all components
    ref_pos_interp = _make_orbit_interpolator(ref_pos, ref_time)
    sec_pos_interp = _make_orbit_interpolator(sec_pos, sec_time)
    ref_vel_interp = _make_orbit_interpolator(ref_vel, ref_time)

    # Interpolate satellite positions to azimuth times
    ref_sat = _eval_orbit_interpolator(ref_pos_interp, az_time)
    sec_sat = _eval_orbit_interpolator(sec_pos_interp, az_time)
    ref_vel_at = _eval_orbit_interpolator(ref_vel_interp, az_time)

    # Baseline vector in ECEF
    baseline = sec_sat - ref_sat

    # LOS direction: satellite to ground
    los_vec = ref_sat - ground_ecef
    los_mag = np.sqrt(np.sum(los_vec**2, axis=-1, keepdims=True))
    if np.any(los_mag == 0):
        log.warning("Zero-magnitude LOS vectors detected — satellite at ground point?")
    los_hat = np.where(los_mag > 0, los_vec / los_mag, 0)

    # Along-track direction from velocity
    vel_mag = np.sqrt(np.sum(ref_vel_at**2, axis=-1, keepdims=True))
    along_hat = np.where(vel_mag > 0, ref_vel_at / vel_mag, 0)

    # Cross-track direction = along-track × LOS (perpendicular to both)
    # This gives the direction perpendicular to the LOS in the plane
    # normal to the flight path — the standard B_perp direction.
    cross_hat = np.cross(along_hat, los_hat)
    cross_mag = np.sqrt(np.sum(cross_hat**2, axis=-1, keepdims=True))
    if np.any(cross_mag == 0):
        log.warning("Zero-magnitude cross-track vectors — LOS parallel to velocity?")
    cross_hat = np.where(cross_mag > 0, cross_hat / cross_mag, 0)

    # B_perp: component of baseline along the cross-track direction
    # (perpendicular to LOS, in the along-track/LOS plane)
    b_perp = np.sum(baseline * cross_hat, axis=-1)

    # B_par: component along LOS
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

    position = np.asarray(orbit["position"]) if "position" in orbit else None
    velocity = np.asarray(orbit["velocity"]) if "velocity" in orbit else None

    # Time may be a data var or stored in attrs (1D arrays go to attrs)
    if "time" in orbit:
        time = np.asarray(orbit["time"])
    elif "time" in orbit.attrs:
        raw = orbit.attrs["time"]
        if isinstance(raw, (list, np.ndarray)):
            time = np.asarray(raw, dtype=np.float64)
        else:
            raise ValueError(
                f"Orbit time in attrs is not an array: {type(raw)}. "
                f"Expected a list of epoch times."
            )
    else:
        time = None

    if position is None or time is None:
        raise ValueError("Could not extract orbit data from DataTree")
    if velocity is None:
        raise ValueError("Could not extract velocity from DataTree")

    if time.ndim == 0:
        raise ValueError("Orbit time is scalar — expected 1D array of epoch times")

    return position, velocity, time


def _extract_epsg_from_dataset(rg: xr.Dataset) -> int:
    """Extract EPSG from a radarGrid dataset, raising on failure."""
    epsg = rg.attrs.get("projection")
    if epsg is not None:
        try:
            return int(epsg)
        except (ValueError, TypeError):
            pass

    # Check for spatial_ref coordinate (set by rioxarray)
    if "spatial_ref" in rg.coords:
        try:
            import rioxarray  # noqa: F401

            for var in rg.data_vars:
                crs = rg[var].rio.crs
                if crs is not None:
                    return crs.to_epsg()
        except Exception:
            pass

    raise ValueError(
        "Could not determine EPSG code from radarGrid. "
        "Ensure the DataTree was opened with open_nisar()."
    )


def _make_orbit_interpolator(
    data: np.ndarray,
    time: np.ndarray,
) -> list:
    """Build cubic spline interpolators for orbit XYZ components.

    Returns a list of 3 spline objects (one per component).
    """
    splines = []
    for i in range(3):
        splines.append(make_interp_spline(time, data[:, i], k=3))
    return splines


def _eval_orbit_interpolator(
    splines: list,
    query_times: np.ndarray,
) -> np.ndarray:
    """Evaluate pre-built orbit interpolators at query times.

    Parameters
    ----------
    splines : list of 3 spline objects
    query_times : np.ndarray, shape (ny, nx)

    Returns
    -------
    np.ndarray, shape (ny, nx, 3)
    """
    shape = query_times.shape
    flat = query_times.ravel()

    # Warn if extrapolating
    t_min, t_max = splines[0].t[0], splines[0].t[-1]
    if np.any(flat < t_min) or np.any(flat > t_max):
        warnings.warn(
            f"Azimuth times [{flat.min():.1f}, {flat.max():.1f}] exceed orbit "
            f"time bounds [{t_min:.1f}, {t_max:.1f}] — extrapolating.",
            UserWarning,
            stacklevel=3,
        )

    result = np.zeros((*shape, 3), dtype=np.float64)
    for i, spline in enumerate(splines):
        result[..., i] = spline(flat).reshape(shape)

    return result


def _grid_to_ecef(
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    height: float | xr.DataArray = 0.0,
) -> np.ndarray:
    """Convert a projected coordinate grid to ECEF XYZ.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays.
    epsg : int
        EPSG code for the projected CRS.
    height : float or xr.DataArray
        Height above ellipsoid. Scalar or 2D array matching the grid.

    Returns
    -------
    np.ndarray, shape (ny, nx, 3)
    """
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4978", always_xy=True)

    xx, yy = np.meshgrid(x, y)
    if isinstance(height, xr.DataArray):
        hh = height.values
    elif isinstance(height, np.ndarray):
        hh = height
    else:
        hh = np.full_like(xx, float(height))

    ecef_x, ecef_y, ecef_z = transformer.transform(xx, yy, hh)
    return np.stack([ecef_x, ecef_y, ecef_z], axis=-1)
