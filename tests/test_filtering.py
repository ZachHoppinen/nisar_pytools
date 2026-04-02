"""Tests for nisar_pytools.processing.filtering."""

import numpy as np
import pytest
import xarray as xr

from nisar_pytools.processing.filtering import goldstein_filter


def _make_noisy_ifg(ny=64, nx=64, seed=42):
    rng = np.random.default_rng(seed)
    x = np.arange(nx, dtype="f8") * 100.0
    y = np.arange(ny, dtype="f8") * -100.0
    # Phase ramp + noise
    phase = np.linspace(0, 4 * np.pi, nx)[None, :] * np.ones((ny, 1))
    noise = rng.normal(0, 0.5, (ny, nx))
    data = np.exp(1j * (phase + noise)).astype(np.complex64)
    return xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})


def _make_clean_ifg(ny=64, nx=64):
    x = np.arange(nx, dtype="f8") * 100.0
    y = np.arange(ny, dtype="f8") * -100.0
    phase = np.linspace(0, 2 * np.pi, nx)[None, :] * np.ones((ny, 1))
    data = np.exp(1j * phase).astype(np.complex64)
    return xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})


class TestGoldsteinFilter:
    def test_output_shape(self):
        ifg = _make_noisy_ifg()
        result = goldstein_filter(ifg)
        assert result.shape == ifg.shape

    def test_output_is_complex(self):
        ifg = _make_noisy_ifg()
        result = goldstein_filter(ifg)
        assert np.iscomplexobj(result.values)

    def test_coords_preserved(self):
        ifg = _make_noisy_ifg()
        result = goldstein_filter(ifg)
        np.testing.assert_array_equal(result.x.values, ifg.x.values)
        np.testing.assert_array_equal(result.y.values, ifg.y.values)

    def test_reduces_noise(self):
        ifg = _make_noisy_ifg()
        result = goldstein_filter(ifg, alpha=0.8)
        # Phase variance should decrease after filtering
        orig_phase = np.angle(ifg.values)
        filt_phase = np.angle(result.values)
        # Compare phase differences (gradient) variance — more robust
        orig_grad_var = np.var(np.diff(orig_phase, axis=1))
        filt_grad_var = np.var(np.diff(filt_phase, axis=1))
        assert filt_grad_var < orig_grad_var

    def test_alpha_zero_minimal_change(self):
        ifg = _make_clean_ifg()
        result = goldstein_filter(ifg, alpha=0.0)
        # alpha=0 should produce minimal filtering in the interior
        # (edges may differ due to patch windowing)
        margin = 8
        orig = np.angle(ifg.values[margin:-margin, margin:-margin])
        filt = np.angle(result.values[margin:-margin, margin:-margin])
        np.testing.assert_allclose(filt, orig, atol=0.15)

    def test_alpha_one_strong_filtering(self):
        ifg = _make_noisy_ifg()
        weak = goldstein_filter(ifg, alpha=0.2)
        strong = goldstein_filter(ifg, alpha=0.9)
        # Stronger alpha should produce smoother result
        weak_var = np.var(np.diff(np.angle(weak.values), axis=1))
        strong_var = np.var(np.diff(np.angle(strong.values), axis=1))
        assert strong_var < weak_var

    def test_invalid_alpha_raises(self):
        ifg = _make_noisy_ifg()
        with pytest.raises(ValueError, match="alpha must be"):
            goldstein_filter(ifg, alpha=1.5)

    def test_invalid_overlap_raises(self):
        ifg = _make_noisy_ifg()
        with pytest.raises(ValueError, match="overlap"):
            goldstein_filter(ifg, patch_size=32, overlap=16)

    def test_attrs_preserved(self):
        ifg = _make_noisy_ifg()
        ifg.attrs["units"] = "radians"
        result = goldstein_filter(ifg)
        assert result.attrs["units"] == "radians"
        assert result.attrs["goldstein_alpha"] == 0.5

    def test_non_power_of_two_size(self):
        # Should work with non-standard image sizes
        ifg = _make_noisy_ifg(ny=50, nx=70)
        result = goldstein_filter(ifg, patch_size=16, overlap=4)
        assert result.shape == (50, 70)

    def test_real_input_raises(self):
        da = xr.DataArray(
            np.ones((32, 32), dtype=np.float32),
            dims=["y", "x"],
        )
        with pytest.raises(ValueError, match="complex"):
            goldstein_filter(da)

    def test_3d_input_raises(self):
        da = xr.DataArray(
            np.ones((3, 32, 32), dtype=np.complex64),
            dims=["t", "y", "x"],
        )
        with pytest.raises(ValueError, match="2D"):
            goldstein_filter(da)
