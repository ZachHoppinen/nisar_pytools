"""Masking utilities for NISAR data products."""

from __future__ import annotations

import numpy as np
import xarray as xr


def apply_mask(
    data: xr.DataArray,
    mask: xr.DataArray,
    valid_value: int = 0,
    fill: float = np.nan,
) -> xr.DataArray:
    """Apply a NISAR mask to a data array.

    In NISAR products, the ``mask`` dataset uses 0 for valid pixels
    and nonzero values for various invalid/flagged conditions.

    Parameters
    ----------
    data : xr.DataArray
        Data to mask.
    mask : xr.DataArray
        NISAR mask array (same spatial grid as data).
    valid_value : int
        Value in the mask that indicates valid pixels. Default 0.
    fill : float
        Fill value for masked pixels. Default ``np.nan``.

    Returns
    -------
    xr.DataArray
        Masked data with invalid pixels set to ``fill``.
    """
    return data.where(mask == valid_value, other=fill)


def get_mask(
    dt: xr.DataTree,
    group: str,
) -> xr.DataArray:
    """Extract a mask DataArray from a DataTree node.

    Parameters
    ----------
    dt : xr.DataTree
        Opened NISAR DataTree.
    group : str
        Path to the group containing the mask, e.g.
        ``"science/LSAR/GSLC/grids/frequencyA"``.

    Returns
    -------
    xr.DataArray
    """
    ds = dt[group].dataset
    if "mask" not in ds:
        raise KeyError(f"No 'mask' variable found in {group}")
    return ds["mask"]
