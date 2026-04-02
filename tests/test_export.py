"""Tests for nisar_pytools.io.export."""

import numpy as np
import pytest
import xarray as xr

from nisar_pytools.io.export import read_netcdf, to_netcdf, to_zarr


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


try:
    import zarr as _zarr  # noqa: F401
    _has_zarr = True
except ImportError:
    _has_zarr = False


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


class TestToNetcdf:
    def test_dataset(self, tmp_path):
        ds = _make_dataset()
        out = to_netcdf(ds, tmp_path / "test.nc")
        assert out.exists()
        loaded = xr.open_dataset(out)
        np.testing.assert_allclose(loaded["temp"].values, ds["temp"].values)

    def test_dataarray(self, tmp_path):
        da = _make_dataset()["temp"]
        out = to_netcdf(da, tmp_path / "test.nc")
        loaded = xr.open_dataset(out)
        assert "temp" in loaded

    def test_complex_roundtrip(self, tmp_path):
        da = _make_complex_da()
        out = to_netcdf(da, tmp_path / "complex.nc")
        # Should have split into real/imag
        raw = xr.open_dataset(out)
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
