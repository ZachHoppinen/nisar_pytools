"""Tests for nisar_pytools.utils.metadata."""

import pandas as pd
from shapely.geometry import Polygon

from nisar_pytools import open_nisar
from nisar_pytools.utils.metadata import (
    get_acquisition_time,
    get_bounding_polygon,
    get_orbit_info,
    get_product_type,
)


class TestMetadata:
    def test_product_type(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        assert get_product_type(dt) == "GSLC"

    def test_acquisition_time(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        ts = get_acquisition_time(dt)
        assert isinstance(ts, pd.Timestamp)

    def test_orbit_info(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        info = get_orbit_info(dt)
        assert "track_number" in info
        assert "frame_number" in info
        assert "orbit_direction" in info
        assert info["track_number"] == 77

    def test_bounding_polygon_from_fixture(self, gslc_h5):
        # Fixture doesn't have boundingPolygon, so we add one
        import h5py
        with h5py.File(gslc_h5, "a") as f:
            ident = f["science/LSAR/identification"]
            ident.create_dataset(
                "boundingPolygon",
                data=b"POLYGON ((-115 43, -114 43, -114 44, -115 44, -115 43))",
            )

        dt = open_nisar(gslc_h5)
        poly = get_bounding_polygon(dt)
        assert isinstance(poly, Polygon)
        assert poly.area > 0
