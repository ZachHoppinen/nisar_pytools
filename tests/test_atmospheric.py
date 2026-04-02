"""Tests for nisar_pytools.processing.atmospheric."""

import numpy as np
import xarray as xr

from nisar_pytools.processing.atmospheric import (
    correct_atmosphere,
    correct_ionosphere,
    correct_troposphere,
)


def _make_unwrapped(ny=32, nx=40):
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
    phase = np.random.default_rng(42).uniform(-10, 10, (ny, nx)).astype(np.float32)
    return xr.DataArray(
        phase, dims=["y", "x"], coords={"y": y, "x": x}, name="unwrapped_phase"
    )


def _make_tropo_screens(n_heights=5, ny_rg=10, nx_rg=12):
    heights = np.linspace(0, 4000, n_heights)
    x_rg = np.linspace(499900, 500900, nx_rg)
    y_rg = np.linspace(4496900, 4500100, ny_rg)
    rng = np.random.default_rng(0)
    hydro = rng.uniform(-3, -1, (n_heights, ny_rg, nx_rg))
    wet = rng.uniform(-1, 1, (n_heights, ny_rg, nx_rg))
    return hydro, wet, heights, x_rg, y_rg


def _make_dem(ny=32, nx=40):
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
    elev = np.full((ny, nx), 1500.0, dtype=np.float32)
    return xr.DataArray(elev, dims=["y", "x"], coords={"y": y, "x": x})


def _make_iono(ny=32, nx=40):
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
    screen = np.random.default_rng(1).uniform(-0.5, 0.5, (ny, nx)).astype(np.float32)
    return xr.DataArray(screen, dims=["y", "x"], coords={"y": y, "x": x})


class TestCorrectTroposphere:
    def test_output_shape(self):
        unw = _make_unwrapped()
        hydro, wet, heights, x_rg, y_rg = _make_tropo_screens()
        dem = _make_dem()
        result = correct_troposphere(unw, hydro, wet, heights, x_rg, y_rg, dem)
        assert result.shape == unw.shape

    def test_modifies_phase(self):
        unw = _make_unwrapped()
        hydro, wet, heights, x_rg, y_rg = _make_tropo_screens()
        dem = _make_dem()
        result = correct_troposphere(unw, hydro, wet, heights, x_rg, y_rg, dem)
        # Should not be identical to input
        assert not np.allclose(result.values, unw.values, equal_nan=True)

    def test_zero_screens_no_change(self):
        unw = _make_unwrapped()
        ny_rg, nx_rg = 10, 12
        heights = np.linspace(0, 4000, 5)
        # Make radarGrid fully cover the unwrapped phase extent
        x_rg = np.linspace(499000, 504500, nx_rg)
        y_rg = np.linspace(4496000, 4501000, ny_rg)
        hydro = np.zeros((5, ny_rg, nx_rg))
        wet = np.zeros((5, ny_rg, nx_rg))
        dem = _make_dem()
        result = correct_troposphere(unw, hydro, wet, heights, x_rg, y_rg, dem)
        np.testing.assert_allclose(result.values, unw.values, atol=1e-5)

    def test_attrs_marked(self):
        unw = _make_unwrapped()
        hydro, wet, heights, x_rg, y_rg = _make_tropo_screens()
        dem = _make_dem()
        result = correct_troposphere(unw, hydro, wet, heights, x_rg, y_rg, dem)
        assert result.attrs["tropospheric_correction"] == "applied"


class TestCorrectIonosphere:
    def test_output_shape(self):
        unw = _make_unwrapped()
        iono = _make_iono()
        result = correct_ionosphere(unw, iono)
        assert result.shape == unw.shape

    def test_subtracts_screen(self):
        unw = _make_unwrapped()
        iono = _make_iono()
        result = correct_ionosphere(unw, iono)
        np.testing.assert_allclose(result.values, (unw - iono).values, atol=1e-5)

    def test_zero_screen_no_change(self):
        unw = _make_unwrapped()
        iono = xr.zeros_like(unw)
        result = correct_ionosphere(unw, iono)
        np.testing.assert_allclose(result.values, unw.values, atol=1e-5)

    def test_attrs_marked(self):
        unw = _make_unwrapped()
        iono = _make_iono()
        result = correct_ionosphere(unw, iono)
        assert result.attrs["ionospheric_correction"] == "applied"


class TestCorrectAtmosphere:
    def test_both_corrections(self):
        unw = _make_unwrapped()
        hydro, wet, heights, x_rg, y_rg = _make_tropo_screens()
        dem = _make_dem()
        iono = _make_iono()
        result = correct_atmosphere(
            unw, hydro, wet, heights, x_rg, y_rg, dem, ionosphere_screen=iono
        )
        assert result.attrs["tropospheric_correction"] == "applied"
        assert result.attrs["ionospheric_correction"] == "applied"

    def test_tropo_only(self):
        unw = _make_unwrapped()
        hydro, wet, heights, x_rg, y_rg = _make_tropo_screens()
        dem = _make_dem()
        result = correct_atmosphere(unw, hydro, wet, heights, x_rg, y_rg, dem)
        assert result.attrs["tropospheric_correction"] == "applied"
        assert "ionospheric_correction" not in result.attrs
