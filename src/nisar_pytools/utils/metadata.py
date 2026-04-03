"""Extract common metadata from NISAR DataTrees."""

from __future__ import annotations

import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from shapely import wkt


def get_product_type(dt: xr.DataTree) -> str:
    """Get the product type (e.g. "GSLC", "GUNW")."""
    return dt.attrs.get("product_type", "")


def get_acquisition_time(dt: xr.DataTree) -> pd.Timestamp:
    """Get the acquisition start time from identification metadata.

    Returns
    -------
    pd.Timestamp
    """
    ident = dt["science/LSAR/identification"].dataset
    time_str = ident.attrs.get("zeroDopplerStartTime", "")
    return pd.Timestamp(time_str)


def get_orbit_info(dt: xr.DataTree) -> dict:
    """Get orbit information from identification metadata.

    Returns
    -------
    dict
        Keys: ``track_number``, ``frame_number``, ``orbit_direction``,
        ``absolute_orbit_number``.
    """
    ident = dt["science/LSAR/identification"].dataset.attrs
    return {
        "track_number": ident.get("trackNumber"),
        "frame_number": ident.get("frameNumber"),
        "orbit_direction": ident.get("orbitPassDirection"),
        "absolute_orbit_number": ident.get("absoluteOrbitNumber"),
    }


def get_slc(
    dt: xr.DataTree,
    polarization: str = "HH",
    frequency: str = "frequencyA",
) -> xr.DataArray:
    """Extract a polarization channel from a GSLC DataTree.

    Parameters
    ----------
    dt : xr.DataTree
        Opened GSLC DataTree from :func:`open_nisar`.
    polarization : str
        Polarization to extract (e.g. ``"HH"``, ``"HV"``). Default ``"HH"``.
    frequency : str
        Frequency group (default ``"frequencyA"``).

    Returns
    -------
    xr.DataArray
        Complex SLC DataArray with ``y``/``x`` coordinates and CRS set.

    Raises
    ------
    ValueError
        If the product type is not GSLC, or the requested frequency/polarization
        is not available.
    """
    product_type = get_product_type(dt)
    if product_type and product_type != "GSLC":
        raise ValueError(f"get_slc expects a GSLC DataTree, got '{product_type}'")

    grids_path = f"science/LSAR/GSLC/grids/{frequency}"
    try:
        ds = dt[grids_path].dataset
    except KeyError:
        available = [
            k for k in dt["science/LSAR/GSLC/grids"].children
        ]
        raise ValueError(
            f"Frequency '{frequency}' not found. Available: {available}"
        )

    if polarization not in ds:
        available = [k for k in ds.data_vars if k not in ("mask",)]
        raise ValueError(
            f"Polarization '{polarization}' not found in {frequency}. "
            f"Available: {available}"
        )

    return ds[polarization]


def get_bounding_polygon(dt: xr.DataTree) -> Polygon:
    """Get the bounding polygon from identification metadata.

    Returns
    -------
    shapely.geometry.Polygon
    """
    ident = dt["science/LSAR/identification"].dataset.attrs
    poly_wkt = ident.get("boundingPolygon", "")
    if not poly_wkt:
        raise ValueError("No boundingPolygon found in identification metadata")
    return wkt.loads(poly_wkt)
