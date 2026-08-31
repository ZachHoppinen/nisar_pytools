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


#: processing maturities NISAR collections are published at, most mature first
MATURITIES = ("validated", "provisional", "beta")

#: the operational tier carries no maturity infix: ``NISAR_L<n>_<TYPE>_V1``
_MATURITY_INFIX = {"validated": "", "provisional": "_PROVISIONAL", "beta": "_BETA"}


def _url_filename(url: str) -> str:
    """Extract the filename from a URL, stripping query strings."""
    return urlparse(url).path.rsplit("/", 1)[-1]


def _url_collection(url: str) -> str | None:
    """Collection name from an ASF NISAR URL, or None if not encoded in it."""
    _, _, tail = urlparse(url).path.partition("/NISAR/")
    return tail.split("/", 1)[0] if tail else None


def _collection_names(product_type: str, maturity: str) -> list[str]:
    """CMR collection short names for one product type at one maturity.

    Collections are named ``NISAR_L<n>_<TYPE>_<MATURITY>_V1``, so the level is
    matched rather than hardcoded (RSLC is L1, GSLC is L2, and so on). Matching
    the whole suffix rather than the maturity alone keeps ``validated``, which
    has no infix, from also selecting the beta and provisional collections.

    asf_search 13 dropped the concept IDs this used to return and made
    ``dataset_collections`` a set of short names; iterating it yields those
    names under both layouts, since the old mapping was keyed on them.
    """
    from asf_search.CMR.datasets import dataset_collections

    if maturity not in _MATURITY_INFIX:
        raise ValueError(
            f"Unknown maturity '{maturity}'. Supported: {sorted(MATURITIES)}"
        )
    suffix = f"_{product_type}{_MATURITY_INFIX[maturity]}_V1"
    names = sorted(n for n in dataset_collections["NISAR"] if n.endswith(suffix))
    if not names:
        raise ValueError(
            f"No NISAR {product_type} collection at maturity '{maturity}'. "
            f"Supported: {sorted(MATURITIES)}"
        )
    return names


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
    maturity: str | None = None,
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
    maturity : str, optional
        Restrict to one processing baseline: ``"validated"``, ``"provisional"``
        or ``"beta"``.
        ``None`` (default) searches every baseline and a warning names them when
        more than one comes back.

        Note that restricting narrows the *time range*, it does not deduplicate.
        The baselines partition the mission rather than overlapping: over a wide
        Alaskan AOI across the whole mission, beta covers 2025-10-18 to
        2026-01-20 and provisional 2026-06-17 to 2026-07-27, with no acquisition
        appearing under more than one composite release ID. So asking for one
        maturity silently drops the other's dates. Mix them only if you accept
        that the products were processed differently.

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
        intersectsWith=aoi_geom.wkt,
        start=start,
        end=end,
        processingLevel=PRODUCT_TYPES[pt_upper],
    )
    if maturity is None:
        search_kwargs["platform"] = asf.PLATFORM.NISAR
    else:
        # asf_search ORs platform with the collection keywords, so passing both
        # would widen the search back to every maturity instead of narrowing it.
        search_kwargs["shortName"] = _collection_names(pt_upper, maturity)
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

    # The baselines cover different date ranges rather than reprocessing the
    # same acquisitions, so a mixed result is real extra coverage -- but the
    # products either side of the boundary were processed differently.
    found = sorted({c for c in map(_url_collection, urls) if c})
    if len(found) > 1:
        log.warning(
            "Results span %d processing baselines (%s). These cover different "
            "dates and were processed differently; pass maturity= to restrict "
            "to one, at the cost of the other's date range.",
            len(found), ", ".join(found),
        )

    log.info("Found %d URLs after filtering", len(urls))

    return urls
