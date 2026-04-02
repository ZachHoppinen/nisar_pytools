"""Tests for nisar_pytools.io.search (find_nisar).

Note: Tests that hit the ASF API are marked as integration tests.
Unit tests mock asf_search to avoid network calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from nisar_pytools.io.search import find_nisar, PRODUCT_TYPES


class TestFindNisarValidation:
    def test_unknown_product_type_raises(self):
        with pytest.raises(ValueError, match="Unknown product_type"):
            find_nisar(
                aoi=[-115, 43, -114, 44],
                start_date="2025-06-01",
                end_date="2025-07-01",
                product_type="FAKE",
            )

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="ASCENDING.*DESCENDING"):
            find_nisar(
                aoi=[-115, 43, -114, 44],
                start_date="2025-06-01",
                end_date="2025-07-01",
                direction="SIDEWAYS",
            )

    def test_product_types_mapping(self):
        assert "GSLC" in PRODUCT_TYPES
        assert "GUNW" in PRODUCT_TYPES
        assert "RSLC" in PRODUCT_TYPES

    @patch("nisar_pytools.io.search.asf")
    def test_returns_h5_urls_only(self, mock_asf):
        mock_results = MagicMock()
        mock_results.find_urls.return_value = [
            "https://asf.alaska.edu/data/file.h5",
            "https://asf.alaska.edu/data/file.xml",
            "https://asf.alaska.edu/data/file.png",
            "https://asf.alaska.edu/data/other.h5",
        ]
        mock_asf.search.return_value = mock_results
        mock_asf.PLATFORM.NISAR = "NISAR"

        urls = find_nisar(
            aoi=[-115, 43, -114, 44],
            start_date="2025-06-01",
            end_date="2025-07-01",
        )
        assert len(urls) == 2
        assert all(u.endswith(".h5") for u in urls)

    @patch("nisar_pytools.io.search.asf")
    def test_passes_path_and_frame(self, mock_asf):
        mock_results = MagicMock()
        mock_results.find_urls.return_value = []
        mock_asf.search.return_value = mock_results
        mock_asf.PLATFORM.NISAR = "NISAR"

        find_nisar(
            aoi=[-115, 43, -114, 44],
            start_date="2025-06-01",
            end_date="2025-07-01",
            path_number=77,
            frame=24,
            direction="ASCENDING",
        )

        call_kwargs = mock_asf.search.call_args.kwargs
        assert call_kwargs["relativeOrbit"] == 77
        assert call_kwargs["frame"] == 24
        assert call_kwargs["flightDirection"] == "ASCENDING"

    @patch("nisar_pytools.io.search.asf")
    def test_empty_results(self, mock_asf):
        mock_results = MagicMock()
        mock_results.find_urls.return_value = []
        mock_asf.search.return_value = mock_results
        mock_asf.PLATFORM.NISAR = "NISAR"

        urls = find_nisar(
            aoi=[-115, 43, -114, 44],
            start_date="2025-06-01",
            end_date="2025-07-01",
        )
        assert urls == []
