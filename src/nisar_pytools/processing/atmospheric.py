"""Atmospheric phase correction for NISAR GUNW products.

Applies tropospheric (hydrostatic + wet) and ionospheric phase screen
corrections to unwrapped interferograms.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

log = logging.getLogger(__name__)


def correct_troposphere(
    unwrapped_phase: xr.DataArray,
    hydrostatic: np.ndarray,
    wet: np.ndarray,
    heights: np.ndarray,
    x_rg: np.ndarray,
    y_rg: np.ndarray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """Remove tropospheric phase from an unwrapped interferogram.

    Interpolates the 3D hydrostatic and wet tropospheric phase screens
    from the GUNW radarGrid to the unwrapped phase grid using a DEM
    for the height axis, then subtracts the combined correction.

    The hydrostatic and wet components are interpolated separately and
    summed afterward to reduce peak memory usage.

    Parameters
    ----------
    unwrapped_phase : xr.DataArray
        Unwrapped phase in radians, 2D with ``y``/``x`` coordinates.
    hydrostatic : np.ndarray
        Hydrostatic tropospheric phase screen, shape ``(n_heights, n_y, n_x)``,
        in radians.
    wet : np.ndarray
        Wet tropospheric phase screen, same shape as ``hydrostatic``, in radians.
    heights : np.ndarray
        Height values for the phase screen grid, shape ``(n_heights,)``.
    x_rg, y_rg : np.ndarray
        Coordinate arrays for the radarGrid.
    dem : xr.DataArray
        DEM on the same grid as ``unwrapped_phase``, used to select
        the correct height layer for interpolation.

    Returns
    -------
    xr.DataArray
        Corrected unwrapped phase with tropospheric contribution removed.
    """
    _check_dem_grid(unwrapped_phase, dem)

    # Interpolate separately to reduce peak memory
    hydro_2d = _interpolate_3d_to_2d(hydrostatic, heights, y_rg, x_rg, dem)
    wet_2d = _interpolate_3d_to_2d(wet, heights, y_rg, x_rg, dem)
    tropo_interp = hydro_2d + wet_2d

    corrected = unwrapped_phase - tropo_interp
    return _finalize_corrected(corrected, unwrapped_phase, "tropospheric_correction")


def correct_ionosphere(
    unwrapped_phase: xr.DataArray,
    ionosphere_screen: xr.DataArray,
) -> xr.DataArray:
    """Remove ionospheric phase from an unwrapped interferogram.

    If the ionosphere screen has slightly different coordinates,
    it is interpolated to the unwrapped phase grid before subtraction.

    Parameters
    ----------
    unwrapped_phase : xr.DataArray
        Unwrapped phase in radians.
    ionosphere_screen : xr.DataArray
        Ionosphere phase screen in radians, same or similar grid as
        the unwrapped phase.

    Returns
    -------
    xr.DataArray
        Corrected unwrapped phase with ionospheric contribution removed.
    """
    # Align grids if coordinates don't match exactly
    if not (
        np.array_equal(unwrapped_phase.x.values, ionosphere_screen.x.values)
        and np.array_equal(unwrapped_phase.y.values, ionosphere_screen.y.values)
    ):
        log.info("Interpolating ionosphere screen to unwrapped phase grid")
        ionosphere_screen = ionosphere_screen.interp_like(
            unwrapped_phase, method="linear"
        )

    corrected = unwrapped_phase - ionosphere_screen
    return _finalize_corrected(corrected, unwrapped_phase, "ionospheric_correction")


def correct_atmosphere(
    unwrapped_phase: xr.DataArray,
    hydrostatic: np.ndarray,
    wet: np.ndarray,
    heights: np.ndarray,
    x_rg: np.ndarray,
    y_rg: np.ndarray,
    dem: xr.DataArray,
    ionosphere_screen: xr.DataArray | None = None,
) -> xr.DataArray:
    """Apply both tropospheric and ionospheric corrections.

    Convenience function that combines :func:`correct_troposphere`
    and optionally :func:`correct_ionosphere`.

    Parameters
    ----------
    unwrapped_phase : xr.DataArray
        Unwrapped phase in radians.
    hydrostatic, wet : np.ndarray
        Tropospheric phase screens, shape ``(n_heights, n_y, n_x)``.
    heights : np.ndarray
        Height values for the tropospheric grid.
    x_rg, y_rg : np.ndarray
        Coordinate arrays for the radarGrid.
    dem : xr.DataArray
        DEM for tropospheric height interpolation.
    ionosphere_screen : xr.DataArray, optional
        Ionosphere phase screen. If ``None``, only tropospheric correction
        is applied.

    Returns
    -------
    xr.DataArray
        Corrected unwrapped phase.
    """
    result = correct_troposphere(
        unwrapped_phase, hydrostatic, wet, heights, x_rg, y_rg, dem
    )
    if ionosphere_screen is not None:
        result = correct_ionosphere(result, ionosphere_screen)
    return result


def _finalize_corrected(
    corrected: xr.DataArray,
    source: xr.DataArray,
    correction_key: str,
) -> xr.DataArray:
    """Apply consistent dtype, name, and attrs to a corrected DataArray."""
    result = source.copy(data=corrected.values.astype(np.float32))
    result.attrs[correction_key] = "applied"
    return result


def _check_dem_grid(phase: xr.DataArray, dem: xr.DataArray) -> None:
    """Verify the DEM and unwrapped phase share the same grid."""
    if phase.shape != dem.shape:
        raise ValueError(
            f"DEM shape {dem.shape} does not match unwrapped phase shape {phase.shape}. "
            f"Reproject or resample the DEM to match the unwrapped phase grid."
        )


def _interpolate_3d_to_2d(
    data_3d: np.ndarray,
    heights: np.ndarray,
    y_rg: np.ndarray,
    x_rg: np.ndarray,
    dem: xr.DataArray,
) -> xr.DataArray:
    """Interpolate a 3D (height, y, x) field to a 2D grid using DEM elevations.

    Returns an xr.DataArray with the DEM's coordinates so that xarray
    alignment works correctly when subtracting from the unwrapped phase.
    """
    dem_x = dem.x.values
    dem_y = dem.y.values
    dem_elev = dem.values

    # Make working copies for sorting
    data = data_3d.copy()
    h = heights.copy()
    y = y_rg.copy()
    x = x_rg.copy()

    # RegularGridInterpolator requires all axes strictly ascending
    if h[0] > h[-1]:
        h = h[::-1]
        data = data[::-1, :, :]

    if y[0] > y[-1]:
        y = y[::-1]
        data = data[:, ::-1, :]

    if x[0] > x[-1]:
        x = x[::-1]
        data = data[:, :, ::-1]

    interp = RegularGridInterpolator(
        (h, y, x),
        data,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    yy, xx = np.meshgrid(dem_y, dem_x, indexing="ij")
    pts = np.column_stack([dem_elev.ravel(), yy.ravel(), xx.ravel()])
    result = interp(pts).reshape(dem_elev.shape)

    # Warn if significant fraction of pixels are NaN from extrapolation
    nan_frac = np.isnan(result).sum() / result.size
    if nan_frac > 0.05:
        log.warning(
            "%.1f%% of interpolated pixels are NaN (outside radarGrid extent)",
            nan_frac * 100,
        )

    return xr.DataArray(
        result,
        dims=dem.dims,
        coords=dem.coords,
        attrs={"units": "radians"},
    )
