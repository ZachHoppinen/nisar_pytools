"""Test script to verify find_nisar() can locate the files we have locally.

Searches ASF for the same track/frame/dates as our local test files:
- 2 GSLCs: track 77, frame 24, ascending, Nov 2025
- 1 GUNW: track 149, frame 24, ascending, Dec 2025 - Jan 2026

Usage:
    python scripts/test_search.py
"""

from nisar_pytools import find_nisar

# Bounding box covering our local files (Idaho/Montana area)
AOI = [-119.15, 42.85, -114.62, 46.05]


def search_gslcs():
    print("=" * 60)
    print("Searching for GSLCs: track 77, frame 24, Nov 2025")
    print("=" * 60)

    urls = find_nisar(
        aoi=AOI,
        start_date="2025-11-01",
        end_date="2025-11-30",
        product_type="GSLC",
        path_number=77,
        frame=24,
        direction="ASCENDING",
    )

    print(f"Found {len(urls)} URLs:")
    for url in urls:
        print(f"  {url.split('/')[-1]}")

    # Check that our local files show up
    expected = [
        "NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650",
        "NISAR_L2_PR_GSLC_005_077_A_024_4005_DHDH_A_20251115T124615_20251115T124650",
    ]
    for name in expected:
        found = any(name in url for url in urls)
        status = "FOUND" if found else "MISSING"
        print(f"  [{status}] {name}")

    return urls


def search_gunws():
    print()
    print("=" * 60)
    print("Searching for GUNWs: track 149, frame 24, Dec 2025 - Jan 2026")
    print("=" * 60)

    urls = find_nisar(
        aoi=AOI,
        start_date="2025-12-01",
        end_date="2026-01-31",
        product_type="GUNW",
        path_number=149,
        frame=24,
        direction="ASCENDING",
    )

    print(f"Found {len(urls)} URLs:")
    for url in urls:
        print(f"  {url.split('/')[-1]}")

    expected = "NISAR_L2_PR_GUNW_006_149_A_024_009_4000_SH_20251202T123756"
    found = any(expected in url for url in urls)
    status = "FOUND" if found else "MISSING"
    print(f"  [{status}] {expected}")

    return urls


if __name__ == "__main__":
    gslc_urls = search_gslcs()
    gunw_urls = search_gunws()

    print()
    print("=" * 60)
    total = len(gslc_urls) + len(gunw_urls)
    print(f"Total URLs found: {total}")
    print("=" * 60)
