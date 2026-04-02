"""Tests for nisar_pytools.processing.polsar."""

import numpy as np
import xarray as xr

from nisar_pytools.processing.polsar import (
    alpha,
    anisotropy,
    covariance_elements,
    entropy,
    h_a_alpha,
    mean_alpha,
)


def _make_quad_pol(ny=16, nx=20, seed=42):
    """Create synthetic quad-pol SLC DataArrays (HH, HV, VV)."""
    rng = np.random.default_rng(seed)
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
    coords = {"y": y, "x": x}
    dims = ["y", "x"]

    hh = xr.DataArray(
        (rng.normal(0, 1, (ny, nx)) + 1j * rng.normal(0, 1, (ny, nx))).astype(np.complex64),
        dims=dims, coords=coords,
    )
    hv = xr.DataArray(
        (rng.normal(0, 0.3, (ny, nx)) + 1j * rng.normal(0, 0.3, (ny, nx))).astype(np.complex64),
        dims=dims, coords=coords,
    )
    vv = xr.DataArray(
        (rng.normal(0, 1, (ny, nx)) + 1j * rng.normal(0, 1, (ny, nx))).astype(np.complex64),
        dims=dims, coords=coords,
    )
    return hh, hv, vv


def _make_single_mechanism(ny=16, nx=20):
    """Create SLCs with a single dominant scattering mechanism (surface).

    HH and VV are strong and correlated, HV is near zero.
    This should give low entropy and alpha near 0.
    """
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
    coords = {"y": y, "x": x}
    dims = ["y", "x"]

    rng = np.random.default_rng(0)
    phase = rng.uniform(-np.pi, np.pi, (ny, nx))
    hh = xr.DataArray(
        (5.0 * np.exp(1j * phase) + 0.01 * (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx)))).astype(np.complex64),
        dims=dims, coords=coords,
    )
    hv = xr.DataArray(
        (0.01 * (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx)))).astype(np.complex64),
        dims=dims, coords=coords,
    )
    vv = xr.DataArray(
        (5.0 * np.exp(1j * phase) + 0.01 * (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx)))).astype(np.complex64),
        dims=dims, coords=coords,
    )
    return hh, hv, vv


class TestCovarianceElements:
    def test_keys(self):
        hh, hv, vv = _make_quad_pol()
        elems = covariance_elements(hh, hv, vv)
        assert set(elems.keys()) == {"HHHH", "HHHV", "HVHV", "HVVV", "HHVV", "VVVV"}

    def test_auto_correlations_real(self):
        hh, hv, vv = _make_quad_pol()
        elems = covariance_elements(hh, hv, vv)
        assert not np.iscomplexobj(elems["HHHH"].values)
        assert not np.iscomplexobj(elems["HVHV"].values)
        assert not np.iscomplexobj(elems["VVVV"].values)

    def test_auto_correlations_nonnegative(self):
        hh, hv, vv = _make_quad_pol()
        elems = covariance_elements(hh, hv, vv)
        assert np.all(elems["HHHH"].values >= 0)
        assert np.all(elems["HVHV"].values >= 0)
        assert np.all(elems["VVVV"].values >= 0)

    def test_cross_correlations_complex(self):
        hh, hv, vv = _make_quad_pol()
        elems = covariance_elements(hh, hv, vv)
        assert np.iscomplexobj(elems["HHHV"].values)
        assert np.iscomplexobj(elems["HHVV"].values)

    def test_shapes(self):
        hh, hv, vv = _make_quad_pol(ny=8, nx=12)
        elems = covariance_elements(hh, hv, vv)
        for da in elems.values():
            assert da.shape == (8, 12)


class TestEntropy:
    def test_output_range(self):
        hh, hv, vv = _make_quad_pol()
        H = entropy(hh, hv, vv)
        valid = H.values[~np.isnan(H.values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 1)

    def test_single_mechanism_low_entropy(self):
        hh, hv, vv = _make_single_mechanism()
        H = entropy(hh, hv, vv)
        assert np.nanmean(H.values) < 0.5

    def test_shape_and_coords(self):
        hh, hv, vv = _make_quad_pol(ny=8, nx=12)
        H = entropy(hh, hv, vv)
        assert H.shape == (8, 12)
        np.testing.assert_array_equal(H.x.values, hh.x.values)

    def test_name_and_attrs(self):
        hh, hv, vv = _make_quad_pol()
        H = entropy(hh, hv, vv)
        assert H.name == "entropy"
        assert H.dtype == np.float32


class TestAnisotropy:
    def test_output_range(self):
        hh, hv, vv = _make_quad_pol()
        A = anisotropy(hh, hv, vv)
        valid = A.values[~np.isnan(A.values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 1)

    def test_shape_and_coords(self):
        hh, hv, vv = _make_quad_pol(ny=8, nx=12)
        A = anisotropy(hh, hv, vv)
        assert A.shape == (8, 12)

    def test_name(self):
        hh, hv, vv = _make_quad_pol()
        A = anisotropy(hh, hv, vv)
        assert A.name == "anisotropy"


class TestAlpha:
    def test_output_range(self):
        hh, hv, vv = _make_quad_pol()
        a = alpha(hh, hv, vv)
        valid = a.values[~np.isnan(a.values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 90)

    def test_single_mechanism_low_alpha(self):
        hh, hv, vv = _make_single_mechanism()
        a = alpha(hh, hv, vv)
        assert np.nanmean(a.values) < 30

    def test_name_and_attrs(self):
        hh, hv, vv = _make_quad_pol()
        a = alpha(hh, hv, vv)
        assert a.name == "alpha"
        assert a.attrs["units"] == "degrees"


class TestMeanAlpha:
    def test_output_range(self):
        hh, hv, vv = _make_quad_pol()
        ma = mean_alpha(hh, hv, vv)
        valid = ma.values[~np.isnan(ma.values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 90)

    def test_name(self):
        hh, hv, vv = _make_quad_pol()
        ma = mean_alpha(hh, hv, vv)
        assert ma.name == "mean_alpha"


class TestHAAlpha:
    def test_returns_dataset(self):
        hh, hv, vv = _make_quad_pol()
        ds = h_a_alpha(hh, hv, vv)
        assert isinstance(ds, xr.Dataset)
        assert set(ds.data_vars) == {"entropy", "anisotropy", "alpha", "mean_alpha"}

    def test_consistent_with_individual(self):
        hh, hv, vv = _make_quad_pol()
        ds = h_a_alpha(hh, hv, vv)
        H_individual = entropy(hh, hv, vv)
        np.testing.assert_allclose(ds["entropy"].values, H_individual.values, atol=1e-5)

    def test_coords_preserved(self):
        hh, hv, vv = _make_quad_pol(ny=8, nx=12)
        ds = h_a_alpha(hh, hv, vv)
        for var in ds.data_vars:
            np.testing.assert_array_equal(ds[var].x.values, hh.x.values)
            np.testing.assert_array_equal(ds[var].y.values, hh.y.values)

    def test_all_float32(self):
        hh, hv, vv = _make_quad_pol()
        ds = h_a_alpha(hh, hv, vv)
        for var in ds.data_vars:
            assert ds[var].dtype == np.float32
