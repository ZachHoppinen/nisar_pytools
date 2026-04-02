"""Tests for nisar_pytools.io.export."""

import numpy as np
import pytest
import xarray as xr

from nisar_pytools.io.export import read_netcdf, to_netcdf, to_zarr

try:
    import zarr as _zarr  # noqa: F401

    _has_zarr = True
except ImportError:
    _has_zarr = False


def _make_dataset():
    x = np.arange(10, dtype="f8")
    y = np.arange(8, dtype="f8")
    return xr.Dataset(
        {"temp": xr.DataArray(np.random.rand(8, 10).astype("f4"), dims=["y", "x"])},
        coords={"y": y, "x": x},
    )


def _make_complex_da():
    x = np.arange(10, dtype="f8")
    y = np.arange(8, dtype="f8")
    data = (np.random.randn(8, 10) + 1j * np.random.randn(8, 10)).astype(np.complex64)
    return xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x}, name="slc")


def _make_mixed_dataset():
    """Dataset with both complex and real variables."""
    x = np.arange(5, dtype="f8")
    y = np.arange(4, dtype="f8")
    return xr.Dataset(
        {
            "real_var": xr.DataArray(np.ones((4, 5), dtype="f4"), dims=["y", "x"]),
            "complex_var": xr.DataArray(
                (np.ones((4, 5)) + 1j * np.ones((4, 5))).astype(np.complex64),
                dims=["y", "x"],
            ),
            "another_real": xr.DataArray(np.zeros((4, 5), dtype="f4"), dims=["y", "x"]),
        },
        coords={"y": y, "x": x},
    )


@pytest.mark.skipif(not _has_zarr, reason="zarr not installed")
class TestToZarr:
    def test_dataset(self, tmp_path):
        ds = _make_dataset()
        out = to_zarr(ds, tmp_path / "test.zarr")
        assert out.exists()
        loaded = xr.open_zarr(out)
        np.testing.assert_allclose(loaded["temp"].values, ds["temp"].values)

    def test_dataarray(self, tmp_path):
        da = _make_dataset()["temp"]
        out = to_zarr(da, tmp_path / "test.zarr")
        loaded = xr.open_zarr(out)
        assert "temp" in loaded

    def test_overwrite(self, tmp_path):
        ds = _make_dataset()
        to_zarr(ds, tmp_path / "test.zarr")
        to_zarr(ds, tmp_path / "test.zarr", mode="w")
        loaded = xr.open_zarr(tmp_path / "test.zarr")
        assert "temp" in loaded

    def test_creates_parent_dirs(self, tmp_path):
        ds = _make_dataset()
        out = to_zarr(ds, tmp_path / "sub" / "dir" / "test.zarr")
        assert out.exists()


class TestToNetcdf:
    def test_dataset(self, tmp_path):
        ds = _make_dataset()
        out = to_netcdf(ds, tmp_path / "test.nc")
        assert out.exists()
        loaded = read_netcdf(out)
        np.testing.assert_allclose(loaded["temp"].values, ds["temp"].values)

    def test_dataarray(self, tmp_path):
        da = _make_dataset()["temp"]
        out = to_netcdf(da, tmp_path / "test.nc")
        loaded = read_netcdf(out)
        assert "temp" in loaded

    def test_complex_roundtrip(self, tmp_path):
        da = _make_complex_da()
        out = to_netcdf(da, tmp_path / "complex.nc")
        # Should have split into real/imag
        raw = read_netcdf(out, merge_complex=False)
        assert "slc_real" in raw
        assert "slc_imag" in raw
        # read_netcdf recombines
        loaded = read_netcdf(out)
        assert "slc" in loaded
        np.testing.assert_allclose(loaded["slc"].values, da.values, atol=1e-6)

    def test_creates_parent_dirs(self, tmp_path):
        ds = _make_dataset()
        out = to_netcdf(ds, tmp_path / "sub" / "dir" / "test.nc")
        assert out.exists()

    def test_split_attrs_independent(self, tmp_path):
        """Real and imag components should have independent attrs dicts."""
        da = _make_complex_da()
        da.attrs["custom"] = "value"
        out = to_netcdf(da, tmp_path / "attrs_test.nc")
        raw = read_netcdf(out, merge_complex=False)
        assert raw["slc_real"].attrs["_complex_component"] == "real"
        assert raw["slc_imag"].attrs["_complex_component"] == "imag"
        assert raw["slc_real"].attrs["custom"] == "value"
        assert raw["slc_imag"].attrs["custom"] == "value"

    def test_mixed_dataset_roundtrip(self, tmp_path):
        """Dataset with both complex and real vars round-trips correctly."""
        ds = _make_mixed_dataset()
        out = to_netcdf(ds, tmp_path / "mixed.nc")
        loaded = read_netcdf(out)
        assert "real_var" in loaded
        assert "complex_var" in loaded
        assert "another_real" in loaded
        np.testing.assert_allclose(
            loaded["complex_var"].values, ds["complex_var"].values, atol=1e-6
        )
        np.testing.assert_allclose(loaded["real_var"].values, ds["real_var"].values)

    def test_variable_order_preserved(self, tmp_path):
        """Round-tripped variable order should match original."""
        ds = _make_mixed_dataset()
        out = to_netcdf(ds, tmp_path / "order.nc")
        loaded = read_netcdf(out)
        assert list(loaded.data_vars) == list(ds.data_vars)

    def test_read_netcdf_closes_file(self, tmp_path):
        """read_netcdf should not leave file handles open."""
        ds = _make_dataset()
        out = to_netcdf(ds, tmp_path / "close_test.nc")
        loaded = read_netcdf(out)
        # Should be eagerly loaded — accessing values should work
        # even though the file is closed
        _ = loaded["temp"].values
        assert loaded["temp"].shape == (8, 10)
