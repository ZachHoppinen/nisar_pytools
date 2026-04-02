"""Validation utilities for search and download parameters."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

log = logging.getLogger(__name__)

NISAR_LAUNCH = pd.Timestamp("2024-02-03")


def validate_dates(
    start_date: str | datetime | pd.Timestamp,
    end_date: str | datetime | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize start/end dates for NISAR data search.

    Timezone-aware inputs are converted to UTC before stripping the
    timezone, ensuring consistent comparison.

    Parameters
    ----------
    start_date, end_date : str, datetime, or pd.Timestamp
        Search date bounds. Strings are parsed as ISO format.

    Returns
    -------
    (start, end) : tuple of pd.Timestamp
        Normalized naive (UTC) timestamps.

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
    """Parse a date input to pd.Timestamp (naive UTC)."""
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
            raise ValueError(
                f"{param_name} must be str, datetime, or pd.Timestamp, "
                f"got {type(date)}"
            )
    except (ValueError, pd.errors.ParserError) as e:
        raise ValueError(f"Could not parse {param_name}='{date}': {e}") from e

    if pd.isna(ts):
        raise ValueError(f"{param_name} is NaT")

    # Convert to UTC before stripping timezone to avoid offset errors
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    return ts


def validate_aoi(aoi) -> BaseGeometry:
    """Validate and normalize an area of interest.

    Bounding box coordinates are expected in ``[xmin, ymin, xmax, ymax]``
    order. Swapped min/max values are auto-corrected with a warning.

    For geographic coordinates (used by ASF), valid ranges are
    longitude [-180, 180] and latitude [-90, 90]. Coordinates outside
    these ranges raise a ValueError.

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
        If AOI is empty, has zero area, coordinates are out of range,
        or cannot be parsed.
    """
    geom = None

    if isinstance(aoi, BaseGeometry):
        geom = aoi

    elif isinstance(aoi, (list, tuple, np.ndarray)):
        if len(aoi) == 4:
            xmin, ymin, xmax, ymax = map(float, aoi)
            if xmin > xmax or ymin > ymax:
                log.warning(
                    "AOI bounds appear swapped — auto-correcting. "
                    "Expected [xmin, ymin, xmax, ymax]."
                )
            xmin, xmax = sorted((xmin, xmax))
            ymin, ymax = sorted((ymin, ymax))
            _check_geographic_bounds(xmin, ymin, xmax, ymax)
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
                _check_geographic_bounds(xmin, ymin, xmax, ymax)
                geom = box(xmin, ymin, xmax, ymax)
                break
        if geom is None:
            raise ValueError(f"AOI dict keys not recognized: {list(aoi.keys())}")

    else:
        raise ValueError(f"AOI must be geometry, list/tuple, or dict; got {type(aoi)}")

    if geom.is_empty:
        raise ValueError("AOI geometry is empty")
    if isinstance(geom, Polygon) and geom.area == 0:
        raise ValueError("AOI polygon has zero area")

    return geom


def _check_geographic_bounds(
    xmin: float, ymin: float, xmax: float, ymax: float
) -> None:
    """Warn if coordinates look like they're not in geographic (lon/lat) range."""
    if xmin < -180 or xmax > 180 or ymin < -90 or ymax > 90:
        log.warning(
            "AOI bounds [%.1f, %.1f, %.1f, %.1f] are outside geographic "
            "coordinate range (lon: -180..180, lat: -90..90). "
            "ASF search expects geographic coordinates.",
            xmin, ymin, xmax, ymax,
        )


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
        If list is empty, any element is not a string, or any URL is invalid.
    """
    if not urls:
        raise ValueError("No URLs provided")

    valid = []
    for u in urls:
        if not isinstance(u, str):
            raise ValueError(f"URL must be a string, got {type(u)}: {u}")
        url = u.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid URL (must be http/https with host): {url}")
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
        If ``None``, no check. Incompatible with ``make_directory=True``.
    make_directory : bool
        If ``True``, create the directory (and parents) if needed.
        Cannot be used with ``should_exist=False``.

    Returns
    -------
    Path
        Normalized path.
    """
    if should_exist is False and make_directory:
        raise ValueError("should_exist=False and make_directory=True are contradictory")

    path = Path(filepath).expanduser()

    if should_exist is True and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if should_exist is False and path.exists():
        raise ValueError(f"Path already exists: {path}")
    if make_directory:
        path.mkdir(parents=True, exist_ok=True)

    return path
