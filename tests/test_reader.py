"""Tests for nisar_pytools.io._reader (open_nisar entry point)."""

import os

import dask.array as da
import h5py
import pytest
import xarray as xr

from nisar_pytools import open_nisar


class TestOpenNisar:
    def test_returns_datatree_gslc(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        assert isinstance(dt, xr.DataTree)

    def test_returns_datatree_gunw(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        assert isinstance(dt, xr.DataTree)

    def test_product_type_attr_gslc(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        assert dt.attrs["product_type"] == "GSLC"

    def test_product_type_attr_gunw(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        assert dt.attrs["product_type"] == "GUNW"

    def test_lazy_by_default(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        assert isinstance(freq_a["HH"].data, da.Array)

    def test_coordinates_present(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        assert "x" in freq_a.coords
        assert "y" in freq_a.coords

    def test_file_handle_kept_alive(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        h5file = dt.__dict__["_h5file"]
        assert h5file.id.valid

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            open_nisar("/nonexistent/path/file.h5")

    def test_not_hdf5_raises(self, tmp_path):
        path = tmp_path / "fake.h5"
        path.write_text("not hdf5")
        with pytest.raises(ValueError, match="Not a valid HDF5 file"):
            open_nisar(path)

    def test_accepts_string_path(self, gslc_h5):
        dt = open_nisar(str(gslc_h5))
        assert dt.attrs["product_type"] == "GSLC"

    def test_data_computable(self, gslc_h5):
        """Verify lazy arrays can actually be computed."""
        dt = open_nisar(gslc_h5)
        freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
        result = freq_a["HH"].values
        assert result.shape == (8, 10)

    def test_gunw_multi_resolution_accessible(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        base = "science/LSAR/GUNW/grids/frequencyA"
        unwrapped = dt[f"{base}/unwrappedInterferogram/HH"].dataset
        wrapped = dt[f"{base}/wrappedInterferogram/HH"].dataset
        assert unwrapped["unwrappedPhase"].shape == (6, 8)
        assert wrapped["wrappedInterferogram"].shape == (12, 16)


class TestIntegration:
    @pytest.mark.integration
    def test_real_gslc(self):
        path = os.environ.get("NISAR_TEST_GSLC")
        if not path:
            pytest.skip("Set NISAR_TEST_GSLC env var to a real GSLC .h5 file")
        dt = open_nisar(path)
        assert dt.attrs["product_type"] == "GSLC"
        assert "science" in dt.children

    @pytest.mark.integration
    def test_real_gunw(self):
        path = os.environ.get("NISAR_TEST_GUNW")
        if not path:
            pytest.skip("Set NISAR_TEST_GUNW env var to a real GUNW .h5 file")
        dt = open_nisar(path)
        assert dt.attrs["product_type"] == "GUNW"
        assert "science" in dt.children
