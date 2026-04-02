"""Tests for nisar_pytools.utils.overlap."""

import numpy as np
import pytest
import xarray as xr

from nisar_pytools.utils.overlap import crop_to_overlap, overlap_fraction


def _make_da(x_start, x_end, y_start, y_end, nx=20, ny=15):
    x = np.linspace(x_start, x_end, nx)
    y = np.linspace(y_start, y_end, ny)
    data = np.random.rand(ny, nx).astype(np.float32)
    return xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})


class TestCropToOverlap:
    def test_partial_overlap(self):
        da1 = _make_da(0, 100, 0, 100)
        da2 = _make_da(50, 150, 50, 150)
        c1, c2 = crop_to_overlap(da1, da2)
        assert c1.x.values.min() >= 50
        assert c1.x.values.max() <= 100
        assert c2.x.values.min() >= 50
        assert c2.x.values.max() <= 100

    def test_full_overlap(self):
        da1 = _make_da(0, 100, 0, 100)
        da2 = _make_da(0, 100, 0, 100)
        c1, c2 = crop_to_overlap(da1, da2)
        assert c1.shape == da1.shape
        assert c2.shape == da2.shape

    def test_no_overlap_raises(self):
        da1 = _make_da(0, 100, 0, 100)
        da2 = _make_da(200, 300, 200, 300)
        with pytest.raises(ValueError, match="No spatial overlap"):
            crop_to_overlap(da1, da2)

    def test_descending_y(self):
        da1 = _make_da(0, 100, 100, 0, ny=15)  # descending y
        da2 = _make_da(50, 150, 100, 0, ny=15)
        c1, c2 = crop_to_overlap(da1, da2)
        assert c1.sizes["x"] > 0
        assert c1.sizes["y"] > 0

    def test_preserves_values(self):
        da1 = _make_da(0, 100, 0, 100, nx=10, ny=10)
        da2 = _make_da(0, 100, 0, 100, nx=10, ny=10)
        c1, c2 = crop_to_overlap(da1, da2)
        np.testing.assert_array_equal(c1.values, da1.values)


class TestOverlapFraction:
    def test_full_overlap(self):
        da1 = _make_da(0, 100, 0, 100)
        da2 = _make_da(0, 100, 0, 100)
        assert overlap_fraction(da1, da2) == pytest.approx(1.0)

    def test_no_overlap(self):
        da1 = _make_da(0, 100, 0, 100)
        da2 = _make_da(200, 300, 200, 300)
        assert overlap_fraction(da1, da2) == 0.0

    def test_half_overlap(self):
        da1 = _make_da(0, 100, 0, 100)
        da2 = _make_da(50, 150, 0, 100)
        frac = overlap_fraction(da1, da2)
        assert 0.4 < frac < 0.6

    def test_contained(self):
        da1 = _make_da(0, 200, 0, 200)
        da2 = _make_da(50, 150, 50, 150)
        frac = overlap_fraction(da1, da2)
        assert frac == pytest.approx(1.0)
