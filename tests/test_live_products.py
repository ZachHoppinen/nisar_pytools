"""Regression tests against real, recent NISAR products on ASF.

These hit the network. They search for products acquired in the last few
months and read one GSLC and one RSLC over byte-range, so nothing large is
downloaded (a full product is 8-25 GB; these tests move tens of MB).

A failure here usually means something moved upstream rather than a
regression in this repo: a new processing maturity, an ASF/CMR API change, a
filename convention change, or a product layout change. That is the point of
running them weekly instead of on every push.

The region and the granule are chosen at random, seeded on the run date, so
each run exercises different products rather than passing forever on one that
already worked. A failing run prints its seed; set NISAR_WEEKLY_SEED to that
value to replay the same choices.

Run by .github/workflows/weekly.yml. The remote-read tests need Earthdata
credentials in ~/.netrc and skip without them.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dask.array as da
import h5py
import pytest

from nisar_pytools.io.h5_to_datatree import h5_to_datatree
from nisar_pytools.io.search import (
    MATURITIES,
    _collection_ids,
    _url_collection,
    _url_filename,
    find_nisar,
)
from nisar_pytools.io.stream_rslc import open_remote_rslc, write_skeleton
from nisar_pytools.utils.filename import parse_filename
from nisar_pytools.utils.validation import detect_product_type

pytestmark = pytest.mark.integration

# Wide boxes over land NISAR images regularly. A run picks between them rather
# than pinning one region, so the tests see different tracks, frames and
# acquisition modes over time instead of passing on the same granule forever.
AOIS = {
    "alaska": [-165, 55, -135, 70],
    "western_us": [-125, 32, -103, 49],
    "india": [68, 8, 89, 30],
    "patagonia": [-75, -55, -65, -40],
}
if os.environ.get("NISAR_WEEKLY_AOI"):
    AOIS = {"custom": [float(v) for v in os.environ["NISAR_WEEKLY_AOI"].split(",")]}

LOOKBACK_DAYS = int(os.environ.get("NISAR_WEEKLY_DAYS", "120"))

# Which region and which granule get pulled is seeded on the run date, so a run
# varies week to week but is reproducible: a failure prints its seed, and
# NISAR_WEEKLY_SEED=<that value> replays the same choices.
SEED = os.environ.get("NISAR_WEEKLY_SEED", datetime.now(timezone.utc).strftime("%Y%m%d"))

requires_earthdata = pytest.mark.skipif(
    not (Path.home() / ".netrc").exists(),
    reason="needs Earthdata credentials in ~/.netrc",
)


def _window() -> tuple[str, str]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _search(product_type: str) -> list[str]:
    """Search AOIs in a seeded order, returning the first region with products.

    Regions differ in how often they are published, so trying several keeps the
    run varied without turning a thin region into a false alarm.
    """
    start, end = _window()
    regions = list(AOIS)
    random.Random(f"{SEED}-{product_type}-aoi").shuffle(regions)
    for region in regions:
        urls = find_nisar(AOIS[region], start, end, product_type, max_results=50)
        if urls:
            print(f"[seed={SEED}] {product_type}: {len(urls)} products over {region}")
            return urls
    pytest.fail(
        f"no {product_type} published over any of {regions} between {start} and "
        f"{end} (seed={SEED})"
    )


def _pick(urls: list[str], product_type: str) -> str:
    """One granule out of the search results, chosen by seed rather than newest."""
    url = random.Random(f"{SEED}-{product_type}-granule").choice(urls)
    print(f"[seed={SEED}] {product_type} granule: {_url_filename(url)}")
    return url


def _remote_handle(url: str) -> h5py.File:
    """Open a granule over byte-range from whichever collection served it.

    ``open_remote_rslc`` is RSLC-specific only in its default collection list;
    the open itself is product-agnostic, so passing the collection parsed off
    the URL works for GSLC too.
    """
    collection = _url_collection(url)
    assert collection, f"could not parse a collection out of {url}"
    return open_remote_rslc(_url_filename(url), short_names=(collection,))


@pytest.fixture(scope="session")
def gslc_urls() -> list[str]:
    return _search("GSLC")


@pytest.fixture(scope="session")
def rslc_urls() -> list[str]:
    return _search("RSLC")


@pytest.fixture(scope="session")
def remote_gslc(gslc_urls):
    h5file = _remote_handle(_pick(gslc_urls, "GSLC"))
    yield h5file
    h5file.close()


@pytest.fixture(scope="session")
def remote_rslc(rslc_urls):
    h5file = _remote_handle(_pick(rslc_urls, "RSLC"))
    yield h5file
    h5file.close()


class TestCollections:
    """Catch the mission publishing at a maturity the package cannot select."""

    @pytest.mark.parametrize("product_type", ["GSLC", "RSLC"])
    def test_nothing_published_outside_known_maturities(self, product_type):
        """The unsuffixed ``NISAR_L<n>_<TYPE>_V1`` collections are the operational
        tier. ASF has them registered but empty while the mission is still
        provisional, so ``MATURITIES`` covering only beta and provisional is
        correct today. Granules turning up there is the signal that
        ``_collection_ids`` and ``find_nisar(maturity=)`` can no longer reach
        everything that is published.
        """
        import asf_search as asf
        from asf_search.CMR.datasets import dataset_collections

        reachable = {cid for m in MATURITIES for cid in _collection_ids(product_type, m)}
        for name, ids in dataset_collections["NISAR"].items():
            if f"_{product_type}_" not in name or set(ids) & reachable:
                continue
            assert not asf.search(collections=ids, maxResults=1), (
                f"{name} now holds granules, but "
                f"nisar_pytools.io.search.MATURITIES only knows {sorted(MATURITIES)}."
            )

    @pytest.mark.parametrize("product_type", ["GSLC", "RSLC"])
    @pytest.mark.parametrize("maturity", MATURITIES)
    def test_collection_ids_resolve(self, product_type, maturity):
        assert _collection_ids(product_type, maturity)


class TestSearch:
    def test_recent_gslcs_published(self, gslc_urls):
        assert all(u.endswith(".h5") for u in gslc_urls)
        assert not any("_QA_" in _url_filename(u) for u in gslc_urls)

    def test_recent_rslcs_published(self, rslc_urls):
        assert all(u.endswith(".h5") for u in rslc_urls)

    def test_filenames_still_parse(self, gslc_urls):
        """The filename convention is unversioned, so it can change under us."""
        start, _ = _window()
        for url in gslc_urls:
            info = parse_filename(_url_filename(url))
            assert info.product_type == "GSLC"
            assert info.direction in ("Ascending", "Descending")
            assert str(info.start_time.date()) >= start


@requires_earthdata
class TestRemoteGslc:
    def test_product_type_detected(self, remote_gslc):
        assert detect_product_type(remote_gslc) == "GSLC"

    def test_datatree_builds(self, remote_gslc):
        dt = h5_to_datatree(remote_gslc)
        assert "science" in dt.children
        assert dt["science/LSAR/GSLC/grids/frequencyA"].dataset.rio.crs is not None

    def test_imagery_lazy_and_readable(self, remote_gslc):
        dt = h5_to_datatree(remote_gslc)
        grids = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        pol = next(v for v in ("HH", "VV", "HV", "VH") if v in grids)
        image = grids[pol]
        assert isinstance(image.data, da.Array)
        assert image.dtype.kind == "c"
        # Read a window from the middle; geocoded corners are usually nodata.
        ny, nx = image.shape
        window = image.isel(y=slice(ny // 2, ny // 2 + 64), x=slice(nx // 2, nx // 2 + 64))
        assert window.values.shape == (64, 64)


@requires_earthdata
class TestRemoteRslc:
    def test_swath_layout(self, remote_rslc):
        # detect_product_type only accepts the reader's GSLC/GUNW, so read raw.
        raw = remote_rslc["science/LSAR/identification/productType"][()]
        assert (raw.decode() if isinstance(raw, bytes) else str(raw)).strip() == "RSLC"
        freq_a = remote_rslc["science/LSAR/RSLC/swaths/frequencyA"]
        assert any(pol in freq_a for pol in ("HH", "VV", "HV", "VH"))

    def test_skeleton_roundtrips(self, remote_rslc, tmp_path):
        """The path crop_streamed takes: structure without the imagery."""
        skeleton = write_skeleton(remote_rslc, tmp_path / "skeleton.h5")
        with h5py.File(skeleton, "r") as out:
            src_time = remote_rslc["science/LSAR/RSLC/swaths/zeroDopplerTime"]
            assert out["science/LSAR/RSLC/swaths/zeroDopplerTime"].shape == src_time.shape
            assert out["science/LSAR/RSLC/metadata/orbit/position"].shape[1] == 3
        assert skeleton.stat().st_size < 1_000_000_000
