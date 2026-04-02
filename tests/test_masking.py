"""Tests for nisar_pytools.utils.masking."""

import numpy as np
import xarray as xr

from nisar_pytools.utils.masking import apply_mask


class TestApplyMask:
    def test_masks_invalid_pixels(self):
        data = xr.DataArray(
            np.ones((4, 4), dtype="f4"), dims=["y", "x"],
            coords={"y": np.arange(4.0), "x": np.arange(4.0)},
        )
        mask = xr.DataArray(
            np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype="u1"),
            dims=["y", "x"], coords={"y": np.arange(4.0), "x": np.arange(4.0)},
        )
        result = apply_mask(data, mask)
        # Valid pixels (mask=0) should keep value 1
        assert result.values[0, 0] == 1.0
        # Invalid pixels (mask!=0) should be NaN
        assert np.isnan(result.values[0, 2])

    def test_custom_valid_value(self):
        data = xr.DataArray(np.ones((3, 3), dtype="f4"), dims=["y", "x"])
        mask = xr.DataArray(np.full((3, 3), 255, dtype="u1"), dims=["y", "x"])
        result = apply_mask(data, mask, valid_value=255)
        assert np.all(result.values == 1.0)

    def test_custom_fill(self):
        data = xr.DataArray(np.ones((3, 3), dtype="f4"), dims=["y", "x"])
        mask = xr.DataArray(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype="u1"), dims=["y", "x"])
        result = apply_mask(data, mask, fill=-999.0)
        assert result.values[0, 1] == -999.0

    def test_preserves_coords(self):
        x = np.arange(5.0) * 100
        y = np.arange(3.0) * 100
        data = xr.DataArray(np.ones((3, 5), dtype="f4"), dims=["y", "x"], coords={"y": y, "x": x})
        mask = xr.DataArray(np.zeros((3, 5), dtype="u1"), dims=["y", "x"], coords={"y": y, "x": x})
        result = apply_mask(data, mask)
        np.testing.assert_array_equal(result.x.values, x)

    def test_all_valid(self):
        data = xr.DataArray(np.arange(9.0).reshape(3, 3), dims=["y", "x"])
        mask = xr.DataArray(np.zeros((3, 3), dtype="u1"), dims=["y", "x"])
        result = apply_mask(data, mask)
        np.testing.assert_array_equal(result.values, data.values)
