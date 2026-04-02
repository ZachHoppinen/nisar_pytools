"""Core SAR processing functions: interferogram, coherence, and unwrapping.

All functions operate on xarray DataArrays and preserve coordinates/CRS.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, uniform_filter

log = logging.getLogger(__name__)


def calculate_phase(data: xr.DataArray) -> xr.DataArray:
    """Extract the phase angle from a complex-valued DataArray.

    Parameters
    ----------
    data : xr.DataArray
        Complex-valued input.

    Returns
    -------
    xr.DataArray
        Phase in radians (float32), same coordinates as input.
    """
    # Use np.angle directly on DataArray to preserve dask graphs if present
    phase = np.angle(data)
    return xr.DataArray(
        phase.values.astype(np.float32) if hasattr(phase, "values") else phase.astype(np.float32),
        dims=data.dims,
        coords=data.coords,
        name="phase",
        attrs={"units": "radians", "long_name": "Phase angle"},
    )


def _check_matching_grids(
    a: xr.DataArray, b: xr.DataArray, rtol: float = 1e-9
) -> None:
    """Raise ValueError if two DataArrays have mismatched x/y coordinates.

    Uses approximate comparison (``allclose``) to handle floating-point
    rounding differences between processing paths.
    """
    if a.x.shape != b.x.shape or not np.allclose(a.x.values, b.x.values, rtol=rtol, atol=0):
        raise ValueError(
            f"x coordinates do not match: "
            f"shapes {a.x.shape} vs {b.x.shape}, "
            f"range [{a.x.values[0]}, {a.x.values[-1]}] vs "
            f"[{b.x.values[0]}, {b.x.values[-1]}]"
        )
    if a.y.shape != b.y.shape or not np.allclose(a.y.values, b.y.values, rtol=rtol, atol=0):
        raise ValueError(
            f"y coordinates do not match: "
            f"shapes {a.y.shape} vs {b.y.shape}, "
            f"range [{a.y.values[0]}, {a.y.values[-1]}] vs "
            f"[{b.y.values[0]}, {b.y.values[-1]}]"
        )


def interferogram(slc1: xr.DataArray, slc2: xr.DataArray) -> xr.DataArray:
    """Generate an interferogram from two co-registered SLC images.

    Computes ``slc1 * conj(slc2)``.

    Parameters
    ----------
    slc1, slc2 : xr.DataArray
        Complex-valued SLC images with matching coordinates.

    Returns
    -------
    xr.DataArray
        Complex interferogram with the same coordinates as the inputs.

    Raises
    ------
    ValueError
        If x or y coordinates do not match, or inputs are not complex.
    """
    if not np.iscomplexobj(slc1) or not np.iscomplexobj(slc2):
        raise ValueError("SLC inputs must be complex-valued")
    _check_matching_grids(slc1, slc2)
    ifg = slc1 * np.conj(slc2)
    ifg.name = "interferogram"
    ifg.attrs = {"units": "1", "long_name": "Complex interferogram"}
    return ifg


def multilook(
    data: xr.DataArray,
    looks_y: int = 1,
    looks_x: int = 1,
) -> xr.DataArray:
    """Multilook (spatially average and downsample) a 2D array.

    Averages non-overlapping blocks of ``(looks_y, looks_x)`` pixels.
    The output grid is downsampled accordingly, with coordinates
    taken from the block centers.

    If the array dimensions are not exact multiples of the look factors,
    the trailing rows/columns are trimmed (with a debug log message).

    Parameters
    ----------
    data : xr.DataArray
        2D input array with ``y`` and ``x`` dimensions.
    looks_y : int
        Number of looks (pixels to average) in the y direction.
    looks_x : int
        Number of looks (pixels to average) in the x direction.

    Returns
    -------
    xr.DataArray
        Multilooked array with reduced dimensions.
    """
    if looks_y < 1 or looks_x < 1:
        raise ValueError(f"looks must be >= 1, got looks_y={looks_y}, looks_x={looks_x}")
    if looks_y == 1 and looks_x == 1:
        return data.copy()

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"multilook requires 2D input, got {arr.ndim}D")

    ny, nx = arr.shape

    ny_trim = (ny // looks_y) * looks_y
    nx_trim = (nx // looks_x) * looks_x

    trimmed_y = ny - ny_trim
    trimmed_x = nx - nx_trim
    if trimmed_y > 0 or trimmed_x > 0:
        log.debug(
            "Trimming %d rows and %d columns for multilook (%d×%d)",
            trimmed_y, trimmed_x, looks_y, looks_x,
        )

    arr = arr[:ny_trim, :nx_trim]

    # Reshape and average — works for both real and complex dtypes
    out = arr.reshape(ny_trim // looks_y, looks_y, nx_trim // looks_x, looks_x).mean(axis=(1, 3))

    y_out = data.y.values[:ny_trim].reshape(-1, looks_y).mean(axis=1)
    x_out = data.x.values[:nx_trim].reshape(-1, looks_x).mean(axis=1)

    result = xr.DataArray(
        out.astype(arr.dtype),
        dims=["y", "x"],
        coords={"y": y_out, "x": x_out},
        name=data.name,
        attrs=dict(data.attrs),
    )
    return result


def multilook_interferogram(
    slc1: xr.DataArray,
    slc2: xr.DataArray,
    looks_y: int = 1,
    looks_x: int = 1,
) -> xr.DataArray:
    """Generate a multilooked interferogram from two SLC images.

    Computes ``slc1 * conj(slc2)`` then averages over non-overlapping
    blocks of ``(looks_y, looks_x)`` pixels. Averaging the complex
    interferogram (rather than the phase) preserves coherence information.

    This is a convenience wrapper around :func:`interferogram` +
    :func:`multilook` that sets appropriate output metadata.

    Parameters
    ----------
    slc1, slc2 : xr.DataArray
        Complex-valued SLC images with matching coordinates.
    looks_y : int
        Number of looks in the y direction.
    looks_x : int
        Number of looks in the x direction.

    Returns
    -------
    xr.DataArray
        Multilooked complex interferogram on the downsampled grid.
    """
    ifg = interferogram(slc1, slc2)
    ml = multilook(ifg, looks_y=looks_y, looks_x=looks_x)
    ml.name = "interferogram"
    ml.attrs = {"units": "1", "long_name": "Multilooked complex interferogram"}
    return ml


def coherence(
    slc1: xr.DataArray,
    slc2: xr.DataArray,
    window_size: int = 5,
    method: str = "boxcar",
) -> xr.DataArray:
    """Estimate interferometric coherence magnitude over a spatial window.

    Computes::

        |⟨s1 * conj(s2)⟩| / sqrt(⟨|s1|²⟩ * ⟨|s2|²⟩)

    where ⟨·⟩ denotes spatial averaging over the window.

    Parameters
    ----------
    slc1, slc2 : xr.DataArray
        Complex-valued SLC images with matching coordinates.
    window_size : int
        For ``"boxcar"``: side length of the square averaging window.
        Must be odd and >= 1.
        For ``"gaussian"``: sigma (standard deviation) of the Gaussian
        kernel in pixels. Note: the effective kernel radius is ~4×sigma,
        so ``sigma=3`` gives an effective window of ~25 pixels.
    method : str
        Averaging method: ``"boxcar"`` (default) for uniform weighting,
        or ``"gaussian"`` for Gaussian-weighted averaging.

    Returns
    -------
    xr.DataArray
        Coherence magnitude in [0, 1], same coordinates as inputs.
    """
    if not np.iscomplexobj(slc1) or not np.iscomplexobj(slc2):
        raise ValueError("SLC inputs must be complex-valued")
    _check_matching_grids(slc1, slc2)

    if method not in ("boxcar", "gaussian"):
        raise ValueError(f"method must be 'boxcar' or 'gaussian', got '{method}'")

    if method == "boxcar":
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"window_size must be odd and >= 1, got {window_size}")
    else:
        if window_size < 1:
            raise ValueError(f"window_size (sigma) must be >= 1, got {window_size}")

    s1 = np.asarray(slc1)
    s2 = np.asarray(slc2)

    ifg = s1 * np.conj(s2)
    pow1 = np.abs(s1) ** 2
    pow2 = np.abs(s2) ** 2

    def _avg(a):
        if method == "boxcar":
            return uniform_filter(a, size=window_size)
        return gaussian_filter(a, sigma=window_size)

    # scipy filters don't support complex input, so we average real and
    # imaginary parts separately. This is mathematically equivalent since
    # averaging is a linear operation.
    avg_ifg = _avg(ifg.real) + 1j * _avg(ifg.imag)
    avg_pow1 = _avg(pow1)
    avg_pow2 = _avg(pow2)

    denom = np.sqrt(avg_pow1 * avg_pow2)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.where(denom > 0, np.abs(avg_ifg) / denom, 0.0)

    coh = np.clip(coh, 0.0, 1.0).astype(np.float32)

    result = xr.DataArray(
        coh,
        dims=slc1.dims,
        coords=slc1.coords,
        name="coherence",
        attrs={"units": "1", "long_name": "Interferometric coherence magnitude"},
    )
    return result


def unwrap(
    igram: xr.DataArray,
    corr: xr.DataArray,
    nlooks: float,
    mask: xr.DataArray | None = None,
    cost: str = "smooth",
    init: str = "mcf",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Unwrap an interferogram using SNAPHU.

    Parameters
    ----------
    igram : xr.DataArray
        Complex interferogram (2D). SNAPHU accepts complex input
        directly and extracts the wrapped phase internally.
    corr : xr.DataArray
        Coherence magnitude in [0, 1], same shape as ``igram``.
    nlooks : float
        Equivalent number of independent looks.
    mask : xr.DataArray, optional
        Binary mask of valid pixels (1 = valid, 0 = masked).
    cost : str
        SNAPHU cost mode: ``"smooth"`` or ``"defo"``. Default ``"smooth"``.
    init : str
        Initialization method: ``"mcf"`` or ``"mst"``. Default ``"mcf"``.

    Returns
    -------
    unwrapped_phase : xr.DataArray
        Unwrapped phase in radians.
    connected_components : xr.DataArray
        Connected component labels (0 = not connected).
    """
    import snaphu

    igram_vals = np.asarray(igram)
    corr_vals = np.asarray(corr)
    mask_arr = np.asarray(mask) if mask is not None else None

    unw_arr, conncomp_arr = snaphu.unwrap(
        igram_vals,
        corr_vals,
        nlooks,
        cost=cost,
        init=init,
        mask=mask_arr,
    )

    unw = xr.DataArray(
        unw_arr.astype(np.float32),
        dims=igram.dims,
        coords=igram.coords,
        name="unwrapped_phase",
        attrs={"units": "radians", "long_name": "Unwrapped phase"},
    )

    conncomp = xr.DataArray(
        conncomp_arr,
        dims=igram.dims,
        coords=igram.coords,
        name="connected_components",
        attrs={"long_name": "Connected component labels"},
    )

    return unw, conncomp
