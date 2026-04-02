"""Atmospheric phase correction for NISAR GUNW products.

Applies tropospheric (hydrostatic + wet) and ionospheric phase screen
corrections to unwrapped interferograms.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


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
    tropo = hydrostatic + wet
    tropo_interp = _interpolate_3d_to_2d(
        tropo, heights, y_rg, x_rg, dem
    )

    corrected = unwrapped_phase - tropo_interp

    result = corrected.astype(np.float32)
    result.name = unwrapped_phase.name
    result.attrs = dict(unwrapped_phase.attrs)
    result.attrs["tropospheric_correction"] = "applied"
    return result


def correct_ionosphere(
    unwrapped_phase: xr.DataArray,
    ionosphere_screen: xr.DataArray,
) -> xr.DataArray:
    """Remove ionospheric phase from an unwrapped interferogram.

    The ionosphere phase screen is on the same grid as the unwrapped
    phase and is subtracted directly.

    Parameters
    ----------
    unwrapped_phase : xr.DataArray
        Unwrapped phase in radians.
    ionosphere_screen : xr.DataArray
        Ionosphere phase screen in radians, same grid as the unwrapped phase.

    Returns
    -------
    xr.DataArray
        Corrected unwrapped phase with ionospheric contribution removed.
    """
    corrected = unwrapped_phase - ionosphere_screen

    result = corrected.astype(np.float32)
    result.name = unwrapped_phase.name
    result.attrs = dict(unwrapped_phase.attrs)
    result.attrs["ionospheric_correction"] = "applied"
    return result


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


def _interpolate_3d_to_2d(
    data_3d: np.ndarray,
    heights: np.ndarray,
    y_rg: np.ndarray,
    x_rg: np.ndarray,
    dem: xr.DataArray,
) -> np.ndarray:
    """Interpolate a 3D (height, y, x) field to a 2D grid using DEM elevations."""
    dem_x = dem.x.values
    dem_y = dem.y.values
    dem_elev = dem.values

    # RegularGridInterpolator needs ascending axes
    y_sorted = y_rg if y_rg[0] < y_rg[-1] else y_rg[::-1]
    flip_y = y_rg[0] > y_rg[-1]

    if flip_y:
        data_3d = data_3d[:, ::-1, :]

    interp = RegularGridInterpolator(
        (heights, y_sorted, x_rg),
        data_3d,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    yy, xx = np.meshgrid(dem_y, dem_x, indexing="ij")
    pts = np.column_stack([dem_elev.ravel(), yy.ravel(), xx.ravel()])
    result = interp(pts).reshape(dem_elev.shape)

    return result
