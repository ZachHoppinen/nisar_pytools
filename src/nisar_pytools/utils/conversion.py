"""Unit conversion utilities for SAR data."""

from __future__ import annotations

import numpy as np
import xarray as xr


def to_db(data: xr.DataArray, power: bool = True) -> xr.DataArray:
    """Convert linear values to decibels.

    Parameters
    ----------
    data : xr.DataArray
        Linear-scale data (power or amplitude).
    power : bool
        If ``True`` (default), assume power values: ``10 * log10(data)``.
        If ``False``, assume amplitude values: ``20 * log10(data)``.

    Returns
    -------
    xr.DataArray
        Values in dB.
    """
    scale = 10.0 if power else 20.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = scale * np.log10(np.abs(data))
    result.attrs = dict(data.attrs)
    result.attrs["units"] = "dB"
    result.name = data.name
    return result


def from_db(data: xr.DataArray, power: bool = True) -> xr.DataArray:
    """Convert decibel values to linear scale.

    Parameters
    ----------
    data : xr.DataArray
        Data in dB.
    power : bool
        If ``True`` (default), convert to power: ``10^(data/10)``.
        If ``False``, convert to amplitude: ``10^(data/20)``.

    Returns
    -------
    xr.DataArray
        Linear-scale values.
    """
    scale = 10.0 if power else 20.0
    result = 10.0 ** (data / scale)
    result.attrs = dict(data.attrs)
    result.attrs["units"] = "1"
    result.name = data.name
    return result
