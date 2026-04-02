"""Validation utilities for search and download parameters."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

import numpy as np

NISAR_LAUNCH = pd.Timestamp("2024-02-03")


def validate_dates(
    start_date: str | datetime | pd.Timestamp,
    end_date: str | datetime | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize start/end dates for NISAR data search.

    Parameters
    ----------
    start_date, end_date : str, datetime, or pd.Timestamp
        Search date bounds. Strings are parsed as ISO format.

    Returns
    -------
    (start, end) : tuple of pd.Timestamp
        Normalized timestamps.

    Raises
    ------
    ValueError
        If dates cannot be parsed, are out of order, or predate NISAR launch.
    """
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")

    if start >= end:
        raise ValueError(f"start_date ({start}) must be before end_date ({end})")

    if start < NISAR_LAUNCH:
        raise ValueError(
            f"start_date ({start}) is before NISAR launch ({NISAR_LAUNCH.date()}). "
            f"No data available before this date."
        )

    return start, end


def _parse_date(
    date: str | datetime | pd.Timestamp | np.datetime64,
    param_name: str,
) -> pd.Timestamp:
    """Parse a date input to pd.Timestamp."""
    try:
        if isinstance(date, pd.Timestamp):
            ts = date
        elif isinstance(date, np.datetime64):
            ts = pd.Timestamp(date)
        elif isinstance(date, datetime):
            ts = pd.Timestamp(date)
        elif isinstance(date, str):
            ts = pd.to_datetime(date)
        else:
            raise TypeError(
                f"{param_name} must be str, datetime, or pd.Timestamp, "
                f"got {type(date)}"
            )
    except (ValueError, pd.errors.ParserError) as e:
        raise ValueError(f"Could not parse {param_name}='{date}': {e}") from e

    if pd.isna(ts):
        raise ValueError(f"{param_name} is NaT")

    # Strip timezone for consistency
    if ts.tz is not None:
        ts = ts.tz_localize(None)

    return ts


def validate_aoi(aoi) -> BaseGeometry:
    """Validate and normalize an area of interest.

    Parameters
    ----------
    aoi : shapely geometry, list/tuple, or dict
        Area of interest. Accepts:
        - Shapely geometry (Polygon, Point, etc.)
        - ``[xmin, ymin, xmax, ymax]`` bounding box
        - ``(lon, lat)`` point
        - dict with keys ``{"west", "south", "east", "north"}``
          or ``{"xmin", "ymin", "xmax", "ymax"}``

    Returns
    -------
    shapely.geometry.BaseGeometry
        Validated geometry.

    Raises
    ------
    ValueError
        If AOI is empty, has zero area (for polygons), or cannot be parsed.
    TypeError
        If AOI type is not recognized.
    """
    geom = None

    if isinstance(aoi, BaseGeometry):
        geom = aoi

    elif isinstance(aoi, (list, tuple, np.ndarray)):
        if len(aoi) == 4:
            xmin, ymin, xmax, ymax = map(float, aoi)
            xmin, xmax = sorted((xmin, xmax))
            ymin, ymax = sorted((ymin, ymax))
            geom = box(xmin, ymin, xmax, ymax)
        elif len(aoi) == 2:
            geom = Point(float(aoi[0]), float(aoi[1]))
        else:
            raise ValueError(f"AOI list/tuple must have 2 or 4 elements, got {len(aoi)}")

    elif isinstance(aoi, dict):
        key_sets = [
            ("xmin", "ymin", "xmax", "ymax"),
            ("west", "south", "east", "north"),
            ("minx", "miny", "maxx", "maxy"),
        ]
        for keys in key_sets:
            if all(k in aoi for k in keys):
                xmin, ymin, xmax, ymax = (float(aoi[k]) for k in keys)
                xmin, xmax = sorted((xmin, xmax))
                ymin, ymax = sorted((ymin, ymax))
                geom = box(xmin, ymin, xmax, ymax)
                break
        if geom is None:
            raise ValueError(f"AOI dict keys not recognized: {list(aoi.keys())}")

    else:
        raise TypeError(f"AOI must be geometry, list/tuple, or dict; got {type(aoi)}")

    if geom.is_empty:
        raise ValueError("AOI geometry is empty")
    if isinstance(geom, Polygon) and geom.area == 0:
        raise ValueError("AOI polygon has zero area")

    return geom


def validate_urls(urls: list[str]) -> list[str]:
    """Validate a list of download URLs.

    Parameters
    ----------
    urls : list of str
        URLs to validate.

    Returns
    -------
    list of str
        Cleaned URLs.

    Raises
    ------
    ValueError
        If list is empty or any URL is invalid.
    TypeError
        If any element is not a string.
    """
    if not urls:
        raise ValueError("No URLs provided")

    valid = []
    for u in urls:
        if not isinstance(u, str):
            raise TypeError(f"URL must be a string, got {type(u)}")
        url = u.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"Invalid URL (must start with http/https): {url}")
        valid.append(url)

    return valid


def validate_path(
    filepath: str | Path,
    should_exist: bool | None = None,
    make_directory: bool = False,
) -> Path:
    """Validate and normalize a filesystem path.

    Parameters
    ----------
    filepath : str or Path
        Input path.
    should_exist : bool or None
        If ``True``, path must exist. If ``False``, must not exist.
        If ``None``, no check.
    make_directory : bool
        If ``True``, create the directory (and parents) if needed.

    Returns
    -------
    Path
        Normalized path.
    """
    path = Path(filepath).expanduser()

    if should_exist is True and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if should_exist is False and path.exists():
        raise ValueError(f"Path already exists: {path}")
    if make_directory:
        path.mkdir(parents=True, exist_ok=True)

    return path
