"""Tests for nisar_pytools.utils.conversion."""

import numpy as np
import xarray as xr

from nisar_pytools.utils.conversion import from_db, to_db


def _make_da(values):
    return xr.DataArray(
        np.array(values, dtype="f4"), dims=["y", "x"],
        coords={"y": np.arange(len(values), dtype="f8"), "x": [0.0]},
    )


class TestToDb:
    def test_power_conversion(self):
        da = _make_da([[1.0], [10.0], [100.0]])
        result = to_db(da, power=True)
        np.testing.assert_allclose(result.values.flatten(), [0.0, 10.0, 20.0], atol=1e-5)

    def test_amplitude_conversion(self):
        da = _make_da([[1.0], [10.0], [100.0]])
        result = to_db(da, power=False)
        np.testing.assert_allclose(result.values.flatten(), [0.0, 20.0, 40.0], atol=1e-5)

    def test_zero_gives_neginf(self):
        da = _make_da([[0.0]])
        result = to_db(da)
        assert np.isneginf(result.values[0, 0])

    def test_attrs_set(self):
        da = _make_da([[1.0]])
        result = to_db(da)
        assert result.attrs["units"] == "dB"


class TestFromDb:
    def test_power_conversion(self):
        da = _make_da([[0.0], [10.0], [20.0]])
        result = from_db(da, power=True)
        np.testing.assert_allclose(result.values.flatten(), [1.0, 10.0, 100.0], atol=1e-5)

    def test_amplitude_conversion(self):
        da = _make_da([[0.0], [20.0], [40.0]])
        result = from_db(da, power=False)
        np.testing.assert_allclose(result.values.flatten(), [1.0, 10.0, 100.0], atol=1e-5)

    def test_roundtrip(self):
        original = _make_da([[0.5], [1.0], [2.0]])
        result = from_db(to_db(original, power=True), power=True)
        np.testing.assert_allclose(result.values, original.values, atol=1e-5)

    def test_attrs_set(self):
        da = _make_da([[0.0]])
        result = from_db(da)
        assert result.attrs["units"] == "1"
