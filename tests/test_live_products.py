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
value to replay the same window and the same region. It will not return the
same granule: ASF publishes into a past acquisition window as processing
catches up, so the result list the seed indexes into keeps changing. A failing
run prints the granule it used; pass that to NISAR_WEEKLY_GRANULE to open the
exact product that failed.

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
    _collection_names,
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
# NISAR_WEEKLY_SEED=<that value> replays the same choices. The seed also fixes
# the search window (see _window), without which a replay would draw from a
# different set of products than the run it is meant to reproduce.
SEED = os.environ.get("NISAR_WEEKLY_SEED", datetime.now(timezone.utc).strftime("%Y%m%d"))

# Pin an exact granule, as ``NISAR_WEEKLY_GRANULE="RSLC=<name>.h5"`` (comma
# separated for both types). The seed alone cannot reproduce a past run's
# granule: ASF keeps publishing into an acquisition window after the fact, so
# the result list a seed indexes into is not the same list a week later. This
# is the only way to get back to the exact product that failed.
PINNED = dict(
    entry.split("=", 1)
    for entry in os.environ.get("NISAR_WEEKLY_GRANULE", "").split(",")
    if entry
)

requires_earthdata = pytest.mark.skipif(
    not (Path.home() / ".netrc").exists(),
    reason="needs Earthdata credentials in ~/.netrc",
)


def _window() -> tuple[str, str]:
    """Search window, anchored on the seed rather than on now().

    Anchoring on now() would slide the window between runs, so replaying a seed
    would search a different date range, get a different result set back, and
    then index into it with the same choice, landing on a different granule
    than the run being replayed.
    """
    try:
        end = datetime.strptime(SEED, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        # A seed that is not a run date carries no window with it.
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


def _frequency(names, where: str) -> str:
    """The frequency group a granule actually carries.

    Single-pol products are published with only one of the two: an RSLC whose
    polarization code is ``NASV`` has frequencyB and no frequencyA.
    """
    for freq in ("frequencyA", "frequencyB"):
        if freq in names:
            return freq
    pytest.fail(f"no frequency group in {where} (seed={SEED})")


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


def _open(urls: list[str], product_type: str) -> h5py.File:
    """The granule under test: pinned by env if set, else the seeded choice."""
    if product_type not in PINNED:
        return _remote_handle(_pick(urls, product_type))
    name = PINNED[product_type]
    print(f"[pinned] {product_type} granule: {name}")
    short_names = tuple(
        n for m in MATURITIES for n in _collection_names(product_type, m)
    )
    return open_remote_rslc(name, short_names=short_names)


@pytest.fixture(scope="session")
def remote_gslc(gslc_urls):
    h5file = _open(gslc_urls, "GSLC")
    yield h5file
    h5file.close()


@pytest.fixture(scope="session")
def remote_rslc(rslc_urls):
    h5file = _open(rslc_urls, "RSLC")
    yield h5file
    h5file.close()


class TestCollections:
    """Catch the mission publishing at a maturity the package cannot select."""

    @pytest.mark.parametrize("product_type", ["GSLC", "RSLC"])
    def test_nothing_published_outside_known_maturities(self, product_type):
        """``MATURITIES`` now covers all three tiers ASF publishes NISAR under,
        so every collection carrying this product type should be reachable. One
        turning up that is not is the signal that the mission added a tier and
        ``_collection_names`` and ``find_nisar(maturity=)`` can no longer reach
        everything that is published.
        """
        import asf_search as asf
        from asf_search.CMR.datasets import dataset_collections

        reachable = {n for m in MATURITIES for n in _collection_names(product_type, m)}
        for name in dataset_collections["NISAR"]:
            if f"_{product_type}_" not in name or name in reachable:
                continue
            assert not asf.search(shortName=name, maxResults=1), (
                f"{name} now holds granules, but "
                f"nisar_pytools.io.search.MATURITIES only knows {sorted(MATURITIES)}."
            )

    @pytest.mark.parametrize("product_type", ["GSLC", "RSLC"])
    @pytest.mark.parametrize("maturity", MATURITIES)
    def test_collection_names_resolve(self, product_type, maturity):
        assert _collection_names(product_type, maturity)


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
        grids = dt["science/LSAR/GSLC/grids"]
        freq = _frequency(grids.children, "GSLC grids")
        assert grids[freq].dataset.rio.crs is not None

    def test_imagery_lazy_and_readable(self, remote_gslc):
        dt = h5_to_datatree(remote_gslc)
        node = dt["science/LSAR/GSLC/grids"]
        grids = node[_frequency(node.children, "GSLC grids")].dataset
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
        swaths = remote_rslc["science/LSAR/RSLC/swaths"]
        freq = swaths[_frequency(swaths, "RSLC swaths")]
        assert any(pol in freq for pol in ("HH", "VV", "HV", "VH"))

    def test_skeleton_roundtrips(self, remote_rslc, tmp_path):
        """The path crop_streamed takes: structure without the imagery."""
        skeleton = write_skeleton(remote_rslc, tmp_path / "skeleton.h5")
        with h5py.File(skeleton, "r") as out:
            src_time = remote_rslc["science/LSAR/RSLC/swaths/zeroDopplerTime"]
            assert out["science/LSAR/RSLC/swaths/zeroDopplerTime"].shape == src_time.shape
            assert out["science/LSAR/RSLC/metadata/orbit/position"].shape[1] == 3
        assert skeleton.stat().st_size < 1_000_000_000
