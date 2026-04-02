"""Search for NISAR data products on ASF."""

from __future__ import annotations

import logging
from datetime import datetime
from urllib.parse import urlparse

import asf_search as asf

from nisar_pytools.utils.search_validation import validate_aoi, validate_dates

log = logging.getLogger(__name__)

# Mapping from friendly names to asf_search constants
PRODUCT_TYPES = {
    "GSLC": asf.PRODUCT_TYPE.GSLC,
    "GUNW": asf.PRODUCT_TYPE.GUNW,
    "RSLC": asf.PRODUCT_TYPE.RSLC,
    "GCOV": asf.PRODUCT_TYPE.GCOV,
    "RIFG": asf.PRODUCT_TYPE.RIFG,
    "RUNW": asf.PRODUCT_TYPE.RUNW,
    "ROFF": asf.PRODUCT_TYPE.ROFF,
    "GOFF": asf.PRODUCT_TYPE.GOFF,
}


def _url_filename(url: str) -> str:
    """Extract the filename from a URL, stripping query strings."""
    return urlparse(url).path.rsplit("/", 1)[-1]


def find_nisar(
    aoi,
    start_date: str | datetime,
    end_date: str | datetime,
    product_type: str = "GSLC",
    path_number: int | None = None,
    frame: int | None = None,
    direction: str | None = None,
    max_results: int | None = None,
    include_qa: bool = False,
) -> list[str]:
    """Search ASF for NISAR product download URLs.

    Parameters
    ----------
    aoi : shapely geometry, list, or dict
        Area of interest. Accepts:
        - Shapely geometry (Polygon, Point, etc.)
        - ``[xmin, ymin, xmax, ymax]`` bounding box
        - dict with keys like ``{"west", "south", "east", "north"}``
    start_date, end_date : str or datetime
        Temporal search bounds (ISO format strings or datetime objects).
    product_type : str
        NISAR product type. One of: ``"GSLC"``, ``"GUNW"``, ``"RSLC"``,
        ``"GCOV"``, ``"RIFG"``, ``"RUNW"``, ``"ROFF"``, ``"GOFF"``.
        Default ``"GSLC"``.
    path_number : int, optional
        Relative orbit / track number to filter by.
    frame : int, optional
        Frame number to filter by.
    direction : str, optional
        Flight direction: ``"ASCENDING"`` or ``"DESCENDING"``.
    max_results : int, optional
        Hint to ASF for maximum number of search results. Note that
        post-search filtering (QA exclusion, .h5 filter) may reduce
        the final count below this number.
    include_qa : bool
        If ``True``, include QA files (``_QA_STATS.h5``). Default ``False``.

    Returns
    -------
    list of str
        Download URLs for matching ``.h5`` product files.
        Returns an empty list if no results match.

    Raises
    ------
    ValueError
        If ``product_type`` or ``direction`` is not recognized.
    """
    aoi_geom = validate_aoi(aoi)
    start, end = validate_dates(start_date, end_date)

    pt_upper = product_type.upper()
    if pt_upper not in PRODUCT_TYPES:
        raise ValueError(
            f"Unknown product_type '{product_type}'. "
            f"Supported: {sorted(PRODUCT_TYPES.keys())}"
        )

    # Validate direction before building search kwargs
    if direction is not None:
        direction_upper = direction.upper()
        if direction_upper not in ("ASCENDING", "DESCENDING"):
            raise ValueError(
                f"direction must be 'ASCENDING' or 'DESCENDING', got '{direction}'"
            )
    else:
        direction_upper = None

    search_kwargs = dict(
        platform=asf.PLATFORM.NISAR,
        intersectsWith=aoi_geom.wkt,
        start=start,
        end=end,
        processingLevel=PRODUCT_TYPES[pt_upper],
    )
    if path_number is not None:
        search_kwargs["relativeOrbit"] = path_number
    if frame is not None:
        search_kwargs["frame"] = frame
    if direction_upper is not None:
        search_kwargs["flightDirection"] = direction_upper
    if max_results is not None:
        search_kwargs["maxResults"] = max_results

    log.info(
        "Searching ASF: product_type=%s, path=%s, direction=%s, frame=%s",
        pt_upper, path_number, direction, frame,
    )

    results = asf.search(**search_kwargs)
    urls = results.find_urls()

    # Filter to .h5 product files, optionally excluding QA
    # Use urlparse to handle URLs with query strings
    urls = [u for u in urls if _url_filename(u).endswith(".h5")]
    if not include_qa:
        urls = [u for u in urls if "_QA_" not in _url_filename(u)]

    log.info("Found %d URLs after filtering", len(urls))

    return urls
