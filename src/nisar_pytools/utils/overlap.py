"""Utilities for handling partially overlapping NISAR products."""

from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import Transformer


def crop_to_overlap(
    da1: xr.DataArray,
    da2: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Crop two DataArrays to their spatial overlap region.

    Finds the intersection of the ``x`` and ``y`` coordinate ranges
    and selects the overlapping subset from each array.

    Parameters
    ----------
    da1, da2 : xr.DataArray
        2D (or 3D with time) DataArrays with ``x`` and ``y`` coordinates.

    Returns
    -------
    (da1_cropped, da2_cropped) : tuple of xr.DataArray
        Cropped arrays covering only the overlapping region.

    Raises
    ------
    ValueError
        If the arrays have no spatial overlap.
    """
    x1, x2 = da1.x.values, da2.x.values
    y1, y2 = da1.y.values, da2.y.values

    # Find overlap bounds
    x_min = max(x1.min(), x2.min())
    x_max = min(x1.max(), x2.max())
    y_min = max(y1.min(), y2.min())
    y_max = min(y1.max(), y2.max())

    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"No spatial overlap between arrays. "
            f"da1 x=[{x1.min():.1f}, {x1.max():.1f}], y=[{y1.min():.1f}, {y1.max():.1f}]; "
            f"da2 x=[{x2.min():.1f}, {x2.max():.1f}], y=[{y2.min():.1f}, {y2.max():.1f}]"
        )

    # Handle ascending or descending y
    if y1[0] <= y1[-1]:
        y_slice = slice(y_min, y_max)
    else:
        y_slice = slice(y_max, y_min)

    da1_crop = da1.sel(x=slice(x_min, x_max), y=y_slice)
    da2_crop = da2.sel(x=slice(x_min, x_max), y=y_slice)

    return da1_crop, da2_crop


def overlap_fraction(
    da1: xr.DataArray,
    da2: xr.DataArray,
) -> float:
    """Compute the fraction of spatial overlap between two DataArrays.

    Returns the overlap area as a fraction of the smaller array's extent.

    Parameters
    ----------
    da1, da2 : xr.DataArray
        DataArrays with ``x`` and ``y`` coordinates.

    Returns
    -------
    float
        Overlap fraction in [0, 1]. 0 = no overlap, 1 = fully contained.
    """
    x1, x2 = da1.x.values, da2.x.values
    y1, y2 = da1.y.values, da2.y.values

    x_overlap = max(0, min(x1.max(), x2.max()) - max(x1.min(), x2.min()))
    y_overlap = max(0, min(y1.max(), y2.max()) - max(y1.min(), y2.min()))

    overlap_area = x_overlap * y_overlap

    area1 = (x1.max() - x1.min()) * (y1.max() - y1.min())
    area2 = (x2.max() - x2.min()) * (y2.max() - y2.min())
    smaller_area = min(area1, area2)

    if smaller_area == 0:
        return 0.0

    return float(np.clip(overlap_area / smaller_area, 0, 1))


def reproject_bbox(
    bbox: tuple[float, float, float, float],
    src_crs: int | str = 4326,
    dst_crs: int | str = 32612,
) -> tuple[float, float, float, float]:
    """Reproject a bounding box between coordinate reference systems.

    Transforms all four corners and returns the enclosing bbox in the
    target CRS. Handles CRS specified as EPSG integers or strings
    like ``"EPSG:4326"``.

    Parameters
    ----------
    bbox : tuple of float
        (xmin, ymin, xmax, ymax) in the source CRS. For WGS84 this is
        (lon_min, lat_min, lon_max, lat_max).
    src_crs : int or str
        Source CRS as EPSG code (e.g. ``4326``) or string (``"EPSG:4326"``).
    dst_crs : int or str
        Target CRS.

    Returns
    -------
    tuple of float
        (xmin, ymin, xmax, ymax) in the target CRS.
    """
    if isinstance(src_crs, int):
        src_crs = f"EPSG:{src_crs}"
    if isinstance(dst_crs, int):
        dst_crs = f"EPSG:{dst_crs}"
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xmin, ymin, xmax, ymax = bbox
    xs, ys = [], []
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            tx, ty = t.transform(x, y)
            xs.append(tx)
            ys.append(ty)
    return min(xs), min(ys), max(xs), max(ys)
