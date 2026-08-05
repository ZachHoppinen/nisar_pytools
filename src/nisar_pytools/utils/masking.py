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
    """Apply a mask to a data array, keeping pixels equal to ``valid_value``.

    The default suits ``inputDataExceptionMask``, a bitfield where 0 means no
    anomaly. It does **not** suit the GSLC ``mask`` layer, whose encoding is the
    other way round: 0 is invalid, 1..N is the valid subswath number, and 255 is
    fill outside the acquisition. Use :func:`nisar_pytools.utils.metadata.get_slc`
    with ``valid_mask=True`` for that layer rather than calling this directly.

    Parameters
    ----------
    data : xr.DataArray
        Data to mask.
    mask : xr.DataArray
        Mask array on the same grid as ``data``.
    valid_value : int
        Value in the mask that indicates pixels to keep. Default 0.
    fill : float
        Fill value for masked pixels. Default ``np.nan``.

    Returns
    -------
    xr.DataArray
        Masked data with invalid pixels set to ``fill``.

    Raises
    ------
    ValueError
        If ``mask`` has dimensions ``data`` does not. Without this check xarray
        would broadcast the two into an outer product rather than masking, which
        is silently wrong and can be enormous.
    """
    extra = set(mask.dims) - set(data.dims)
    if extra:
        raise ValueError(
            f"mask has dimensions {sorted(extra)} that data does not "
            f"(mask dims {mask.dims}, data dims {data.dims}); "
            "masking would broadcast instead of aligning."
        )
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
