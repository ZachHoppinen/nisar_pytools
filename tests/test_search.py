"""Tests for nisar_pytools.io.search (find_nisar).

Note: Tests that hit the ASF API are marked as integration tests.
Unit tests mock asf_search to avoid network calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from nisar_pytools.io.search import (
    PRODUCT_TYPES,
    _collection_names,
    _url_collection,
    find_nisar,
)

_BETA = ("https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GSLC_BETA_V1/"
         "G/NISAR_L2_PR_GSLC_010_145_D_055_2005_QPDH_A_20260119T061609"
         "_20260119T061629_X05010_N_P_J_001.h5")
_PROV = ("https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GSLC_PROVISIONAL_V1/"
         "G/NISAR_L2_PR_GSLC_026_092_A_034_2005_QPDH_A_20260726T135041"
         "_20260726T135053_P05023_N_P_J_001.h5")


class TestMaturity:
    """Beta and provisional are reprocessings of the same data, not extra scenes."""

    def test_collection_parsed_from_url(self):
        assert _url_collection(_BETA) == "NISAR_L2_GSLC_BETA_V1"
        assert _url_collection(_PROV) == "NISAR_L2_GSLC_PROVISIONAL_V1"
        assert _url_collection("https://example.com/no/nisar/segment.h5") is None

    def test_collection_names_match_level_and_maturity(self):
        # RSLC is L1 and GSLC is L2, so the level must not be hardcoded.
        assert _collection_names("GSLC", "provisional") == [
            "NISAR_L2_GSLC_PROVISIONAL_V1"]
        assert _collection_names("RSLC", "beta") == ["NISAR_L1_RSLC_BETA_V1"]
        assert not set(_collection_names("GSLC", "beta")) & set(
            _collection_names("GSLC", "provisional"))

    def test_validated_excludes_the_other_maturities(self):
        """Validated has no infix, so a bare ``_V1`` suffix would catch them all."""
        assert _collection_names("GSLC", "validated") == ["NISAR_L2_GSLC_V1"]

    def test_unknown_maturity_raises(self):
        with pytest.raises(ValueError, match="maturity"):
            _collection_names("GSLC", "operational")

    @patch("nisar_pytools.io.search.asf")
    def test_mixed_baselines_warn(self, mock_asf, caplog):
        mock_results = MagicMock()
        mock_results.find_urls.return_value = [_BETA, _PROV]
        mock_asf.search.return_value = mock_results
        find_nisar([-115, 43, -114, 44], "2026-01-01", "2026-08-01")
        assert "processing baselines" in caplog.text

    @patch("nisar_pytools.io.search.asf")
    def test_single_baseline_does_not_warn(self, mock_asf, caplog):
        mock_results = MagicMock()
        mock_results.find_urls.return_value = [_PROV]
        mock_asf.search.return_value = mock_results
        find_nisar([-115, 43, -114, 44], "2026-01-01", "2026-08-01")
        assert "processing baselines" not in caplog.text

    @patch("nisar_pytools.io.search.asf")
    def test_maturity_replaces_platform(self, mock_asf):
        """Passing both would OR them and widen the search back out."""
        mock_results = MagicMock()
        mock_results.find_urls.return_value = [_PROV]
        mock_asf.search.return_value = mock_results
        find_nisar([-115, 43, -114, 44], "2026-01-01", "2026-08-01",
                   maturity="provisional")
        kwargs = mock_asf.search.call_args.kwargs
        assert "shortName" in kwargs
        assert "platform" not in kwargs


class TestFindNisarValidation:
    def test_unknown_product_type_raises(self):
        with pytest.raises(ValueError, match="Unknown product_type"):
            find_nisar(
                aoi=[-115, 43, -114, 44],
                start_date="2025-09-01",
                end_date="2025-10-01",
                product_type="FAKE",
            )

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="ASCENDING.*DESCENDING"):
            find_nisar(
                aoi=[-115, 43, -114, 44],
                start_date="2025-09-01",
                end_date="2025-10-01",
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
            start_date="2025-09-01",
            end_date="2025-10-01",
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
            start_date="2025-09-01",
            end_date="2025-10-01",
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
            start_date="2025-09-01",
            end_date="2025-10-01",
        )
        assert urls == []

    @patch("nisar_pytools.io.search.asf")
    def test_handles_query_strings_in_urls(self, mock_asf):
        mock_results = MagicMock()
        mock_results.find_urls.return_value = [
            "https://asf.alaska.edu/data/product.h5?token=abc123",
            "https://asf.alaska.edu/data/product_QA_STATS.h5?token=abc123",
        ]
        mock_asf.search.return_value = mock_results
        mock_asf.PLATFORM.NISAR = "NISAR"

        urls = find_nisar(
            aoi=[-115, 43, -114, 44],
            start_date="2025-09-01",
            end_date="2025-10-01",
        )
        assert len(urls) == 1
        assert "QA" not in urls[0]

    @patch("nisar_pytools.io.search.asf")
    def test_include_qa(self, mock_asf):
        mock_results = MagicMock()
        mock_results.find_urls.return_value = [
            "https://asf.alaska.edu/data/product.h5",
            "https://asf.alaska.edu/data/product_QA_STATS.h5",
        ]
        mock_asf.search.return_value = mock_results
        mock_asf.PLATFORM.NISAR = "NISAR"

        urls = find_nisar(
            aoi=[-115, 43, -114, 44],
            start_date="2025-09-01",
            end_date="2025-10-01",
            include_qa=True,
        )
        assert len(urls) == 2
