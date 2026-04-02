"""Tests for nisar_pytools.processing.sar."""

import numpy as np
import pytest
import xarray as xr

from nisar_pytools.processing.sar import (
    calculate_phase,
    coherence,
    interferogram,
    multilook,
    multilook_interferogram,
    unwrap,
)


def _make_slc(ny=32, nx=40, phase=0.0, amplitude=1.0, seed=42):
    """Create a synthetic complex SLC DataArray."""
    rng = np.random.default_rng(seed)
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
    noise = rng.normal(0, 0.1, (ny, nx)) + 1j * rng.normal(0, 0.1, (ny, nx))
    signal = amplitude * np.exp(1j * phase) + noise
    return xr.DataArray(
        signal.astype(np.complex64),
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )


def _make_slc_pair(ny=32, nx=40, phase_diff=1.5, seed=42):
    """Create a pair of SLCs with a known phase difference."""
    rng = np.random.default_rng(seed)
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
    base_phase = rng.uniform(-np.pi, np.pi, (ny, nx))
    amp = rng.uniform(0.5, 1.5, (ny, nx))

    s1 = amp * np.exp(1j * base_phase)
    s2 = amp * np.exp(1j * (base_phase + phase_diff))
    noise = rng.normal(0, 0.05, (ny, nx)) + 1j * rng.normal(0, 0.05, (ny, nx))
    s2 = s2 + noise

    slc1 = xr.DataArray(s1.astype(np.complex64), dims=["y", "x"], coords={"y": y, "x": x})
    slc2 = xr.DataArray(s2.astype(np.complex64), dims=["y", "x"], coords={"y": y, "x": x})
    return slc1, slc2


class TestCalculatePhase:
    def test_known_phase(self):
        phase_val = 1.2
        arr = np.exp(1j * phase_val) * np.ones((4, 4), dtype=np.complex64)
        da = xr.DataArray(arr, dims=["y", "x"], coords={"y": np.arange(4.0), "x": np.arange(4.0)})
        result = calculate_phase(da)
        np.testing.assert_allclose(result.values, phase_val, atol=1e-6)

    def test_output_is_float32(self):
        da = xr.DataArray(
            np.ones((4, 4), dtype=np.complex64),
            dims=["y", "x"],
            coords={"y": np.arange(4.0), "x": np.arange(4.0)},
        )
        result = calculate_phase(da)
        assert result.dtype == np.float32

    def test_coords_preserved(self):
        x = np.arange(5.0) * 100
        y = np.arange(3.0) * 100
        da = xr.DataArray(
            np.ones((3, 5), dtype=np.complex64),
            dims=["y", "x"],
            coords={"y": y, "x": x},
        )
        result = calculate_phase(da)
        np.testing.assert_array_equal(result.x.values, x)
        np.testing.assert_array_equal(result.y.values, y)

    def test_range_minus_pi_to_pi(self):
        rng = np.random.default_rng(42)
        arr = (rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))).astype(np.complex64)
        da = xr.DataArray(arr, dims=["y", "x"], coords={"y": np.arange(10.0), "x": np.arange(10.0)})
        result = calculate_phase(da)
        assert np.all(result.values >= -np.pi)
        assert np.all(result.values <= np.pi)

    def test_name_and_attrs(self):
        da = xr.DataArray(
            np.ones((2, 2), dtype=np.complex64),
            dims=["y", "x"],
            coords={"y": np.arange(2.0), "x": np.arange(2.0)},
        )
        result = calculate_phase(da)
        assert result.name == "phase"
        assert result.attrs["units"] == "radians"


class TestGridCheck:
    def test_mismatched_x_raises(self):
        slc1 = _make_slc(ny=8, nx=10, seed=1)
        x2 = slc1.x.values + 50.0  # shifted x
        slc2 = slc1.copy()
        slc2 = slc2.assign_coords(x=x2)
        with pytest.raises(ValueError, match="x coordinates do not match"):
            interferogram(slc1, slc2)

    def test_mismatched_y_raises(self):
        slc1 = _make_slc(ny=8, nx=10, seed=1)
        slc2 = slc1.isel(y=slice(0, 6))  # fewer y pixels
        with pytest.raises(ValueError, match="y coordinates do not match"):
            interferogram(slc1, slc2)

    def test_coherence_also_checks(self):
        slc1 = _make_slc(ny=8, nx=10, seed=1)
        slc2 = slc1.assign_coords(x=slc1.x.values + 1.0)
        with pytest.raises(ValueError, match="x coordinates do not match"):
            coherence(slc1, slc2)

    def test_matching_grids_pass(self):
        slc1, slc2 = _make_slc_pair()
        # Should not raise
        interferogram(slc1, slc2)
        coherence(slc1, slc2)


class TestInterferogram:
    def test_output_is_complex(self):
        slc1, slc2 = _make_slc_pair()
        ifg = interferogram(slc1, slc2)
        assert np.iscomplexobj(ifg.values)

    def test_shape_preserved(self):
        slc1, slc2 = _make_slc_pair(ny=16, nx=20)
        ifg = interferogram(slc1, slc2)
        assert ifg.shape == (16, 20)

    def test_coords_preserved(self):
        slc1, slc2 = _make_slc_pair()
        ifg = interferogram(slc1, slc2)
        np.testing.assert_array_equal(ifg.x.values, slc1.x.values)
        np.testing.assert_array_equal(ifg.y.values, slc1.y.values)

    def test_phase_difference_recovered(self):
        phase_diff = 2.0
        slc1, slc2 = _make_slc_pair(phase_diff=phase_diff)
        ifg = interferogram(slc1, slc2)
        # Mean phase of interferogram should approximate -phase_diff
        # (since ifg = s1 * conj(s2), phase = phase1 - phase2 = -phase_diff)
        mean_phase = np.angle(np.mean(ifg.values))
        np.testing.assert_allclose(mean_phase, -phase_diff, atol=0.3)

    def test_name_and_attrs(self):
        slc1, slc2 = _make_slc_pair()
        ifg = interferogram(slc1, slc2)
        assert ifg.name == "interferogram"
        assert "units" in ifg.attrs

    def test_identical_slcs_give_real(self):
        slc = _make_slc()
        ifg = interferogram(slc, slc)
        # s * conj(s) = |s|², which is real (imaginary part ≈ 0)
        np.testing.assert_allclose(ifg.values.imag, 0, atol=1e-5)


class TestCoherence:
    def test_identical_slcs_give_one(self):
        slc = _make_slc()
        coh = coherence(slc, slc, window_size=5)
        # Identical signals → coherence = 1
        # Edge effects reduce values at borders, check interior
        interior = coh.values[5:-5, 5:-5]
        np.testing.assert_allclose(interior, 1.0, atol=1e-5)

    def test_uncorrelated_slcs_give_low(self):
        # Pure noise SLCs with no shared signal
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(999)
        ny, nx = 64, 64
        x = np.arange(nx, dtype="f8")
        y = np.arange(ny, dtype="f8")
        s1 = (rng1.normal(0, 1, (ny, nx)) + 1j * rng1.normal(0, 1, (ny, nx))).astype(np.complex64)
        s2 = (rng2.normal(0, 1, (ny, nx)) + 1j * rng2.normal(0, 1, (ny, nx))).astype(np.complex64)
        slc1 = xr.DataArray(s1, dims=["y", "x"], coords={"y": y, "x": x})
        slc2 = xr.DataArray(s2, dims=["y", "x"], coords={"y": y, "x": x})
        coh = coherence(slc1, slc2, window_size=11)
        assert np.mean(coh.values) < 0.5

    def test_correlated_pair_gives_high(self):
        slc1, slc2 = _make_slc_pair()
        coh = coherence(slc1, slc2, window_size=7)
        interior = coh.values[5:-5, 5:-5]
        assert np.mean(interior) > 0.8

    def test_output_range(self):
        slc1, slc2 = _make_slc_pair()
        coh = coherence(slc1, slc2, window_size=5)
        assert np.all(coh.values >= 0)
        assert np.all(coh.values <= 1)

    def test_shape_preserved(self):
        slc1, slc2 = _make_slc_pair(ny=16, nx=20)
        coh = coherence(slc1, slc2, window_size=3)
        assert coh.shape == (16, 20)

    def test_coords_preserved(self):
        slc1, slc2 = _make_slc_pair()
        coh = coherence(slc1, slc2, window_size=5)
        np.testing.assert_array_equal(coh.x.values, slc1.x.values)

    def test_name_and_attrs(self):
        slc1, slc2 = _make_slc_pair()
        coh = coherence(slc1, slc2, window_size=5)
        assert coh.name == "coherence"
        assert coh.dtype == np.float32

    def test_even_window_raises(self):
        slc1, slc2 = _make_slc_pair()
        with pytest.raises(ValueError, match="odd"):
            coherence(slc1, slc2, window_size=4)

    def test_window_size_one(self):
        slc1, slc2 = _make_slc_pair()
        coh = coherence(slc1, slc2, window_size=1)
        # Window of 1 → no averaging → coherence should be 1 everywhere
        np.testing.assert_allclose(coh.values, 1.0, atol=1e-5)

    def test_larger_window_smoother(self):
        slc1, slc2 = _make_slc_pair()
        coh3 = coherence(slc1, slc2, window_size=3)
        coh11 = coherence(slc1, slc2, window_size=11)
        # Larger window should produce smoother (lower variance) output
        assert np.std(coh11.values) < np.std(coh3.values)

    def test_gaussian_identical_gives_one(self):
        slc = _make_slc()
        coh = coherence(slc, slc, window_size=3, method="gaussian")
        interior = coh.values[5:-5, 5:-5]
        np.testing.assert_allclose(interior, 1.0, atol=1e-4)

    def test_gaussian_output_range(self):
        slc1, slc2 = _make_slc_pair()
        coh = coherence(slc1, slc2, window_size=3, method="gaussian")
        assert np.all(coh.values >= 0)
        assert np.all(coh.values <= 1)

    def test_gaussian_correlated_high(self):
        slc1, slc2 = _make_slc_pair()
        coh = coherence(slc1, slc2, window_size=3, method="gaussian")
        interior = coh.values[5:-5, 5:-5]
        assert np.mean(interior) > 0.7

    def test_gaussian_smoother_than_boxcar(self):
        slc1, slc2 = _make_slc_pair()
        coh_box = coherence(slc1, slc2, window_size=5, method="boxcar")
        coh_gau = coherence(slc1, slc2, window_size=3, method="gaussian")
        # Gaussian with sigma=3 covers a wider effective window than boxcar 5
        # so should be smoother
        assert np.std(coh_gau.values) < np.std(coh_box.values)

    def test_invalid_method_raises(self):
        slc1, slc2 = _make_slc_pair()
        with pytest.raises(ValueError, match="method must be"):
            coherence(slc1, slc2, window_size=5, method="hamming")


class TestUnwrap:
    def test_basic_unwrap(self):
        """Unwrap a simple constant-phase interferogram."""
        ny, nx = 64, 64
        x = np.arange(nx, dtype="f8")
        y = np.arange(ny, dtype="f8")
        phase = 1.5
        igram_arr = np.full((ny, nx), np.exp(1j * phase), dtype=np.complex64)
        corr_arr = np.ones((ny, nx), dtype=np.float32)

        igram = xr.DataArray(igram_arr, dims=["y", "x"], coords={"y": y, "x": x})
        corr = xr.DataArray(corr_arr, dims=["y", "x"], coords={"y": y, "x": x})

        unw, conncomp = unwrap(igram, corr, nlooks=10.0)

        assert isinstance(unw, xr.DataArray)
        assert isinstance(conncomp, xr.DataArray)
        assert unw.shape == (ny, nx)
        assert unw.name == "unwrapped_phase"
        assert conncomp.name == "connected_components"

        # Unwrapped phase should be approximately constant
        valid = conncomp.values > 0
        if np.any(valid):
            phase_vals = unw.values[valid]
            np.testing.assert_allclose(
                phase_vals - phase_vals[0], 0, atol=0.1,
            )

    def test_coords_preserved(self):
        ny, nx = 32, 32
        x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
        y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
        igram = xr.DataArray(
            np.ones((ny, nx), dtype=np.complex64),
            dims=["y", "x"],
            coords={"y": y, "x": x},
        )
        corr = xr.DataArray(np.ones((ny, nx), dtype=np.float32), dims=["y", "x"], coords={"y": y, "x": x})

        unw, conncomp = unwrap(igram, corr, nlooks=10.0)
        np.testing.assert_array_equal(unw.x.values, x)
        np.testing.assert_array_equal(unw.y.values, y)

    def test_ramp_unwrap(self):
        """Unwrap a phase ramp that wraps multiple times."""
        ny, nx = 64, 64
        x = np.arange(nx, dtype="f8")
        y = np.arange(ny, dtype="f8")
        # Create a phase ramp across x: 0 to 6*pi (wraps ~3 times)
        true_phase = np.broadcast_to(np.linspace(0, 6 * np.pi, nx), (ny, nx)).copy()
        igram_arr = np.exp(1j * true_phase).astype(np.complex64)
        corr_arr = np.ones((ny, nx), dtype=np.float32)

        igram = xr.DataArray(igram_arr, dims=["y", "x"], coords={"y": y, "x": x})
        corr = xr.DataArray(corr_arr, dims=["y", "x"], coords={"y": y, "x": x})

        unw, conncomp = unwrap(igram, corr, nlooks=20.0)

        valid = conncomp.values > 0
        if np.any(valid):
            # The unwrapped phase gradient should match the true gradient
            unw_row = unw.values[ny // 2, :]
            true_row = true_phase[ny // 2, :]
            # Remove offset (unwrapping is relative)
            unw_shifted = unw_row - unw_row[0] + true_row[0]
            np.testing.assert_allclose(unw_shifted, true_row, atol=0.5)


class TestMultilook:
    def test_shape_downsampled(self):
        slc = _make_slc(ny=32, nx=40)
        ml = multilook(slc, looks_y=4, looks_x=5)
        assert ml.shape == (8, 8)

    def test_coords_are_block_centers(self):
        ny, nx = 16, 20
        x = np.arange(nx, dtype="f8") * 10.0
        y = np.arange(ny, dtype="f8") * 10.0
        data = xr.DataArray(np.ones((ny, nx)), dims=["y", "x"], coords={"y": y, "x": x})
        ml = multilook(data, looks_y=4, looks_x=4)
        # First block center: mean of [0, 10, 20, 30] = 15
        np.testing.assert_allclose(ml.y.values[0], 15.0)
        np.testing.assert_allclose(ml.x.values[0], 15.0)

    def test_constant_value_preserved(self):
        data = xr.DataArray(
            np.full((20, 20), 7.0),
            dims=["y", "x"],
            coords={"y": np.arange(20.0), "x": np.arange(20.0)},
        )
        ml = multilook(data, looks_y=5, looks_x=5)
        np.testing.assert_allclose(ml.values, 7.0)

    def test_averaging_correct(self):
        arr = np.zeros((4, 4))
        arr[0:2, 0:2] = 4.0  # top-left block = 4
        arr[0:2, 2:4] = 8.0  # top-right block = 8
        arr[2:4, 0:2] = 2.0  # bottom-left block = 2
        arr[2:4, 2:4] = 6.0  # bottom-right block = 6
        data = xr.DataArray(arr, dims=["y", "x"], coords={"y": np.arange(4.0), "x": np.arange(4.0)})
        ml = multilook(data, looks_y=2, looks_x=2)
        expected = np.array([[4.0, 8.0], [2.0, 6.0]])
        np.testing.assert_allclose(ml.values, expected)

    def test_trims_remainder(self):
        # 10x10 with 3x3 looks → 3x3 output (last row/col trimmed)
        data = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(10.0), "x": np.arange(10.0)},
        )
        ml = multilook(data, looks_y=3, looks_x=3)
        assert ml.shape == (3, 3)

    def test_identity_with_ones(self):
        slc = _make_slc(ny=16, nx=20)
        ml = multilook(slc, looks_y=1, looks_x=1)
        np.testing.assert_array_equal(ml.values, slc.values)

    def test_invalid_looks_raises(self):
        slc = _make_slc()
        with pytest.raises(ValueError, match="looks must be >= 1"):
            multilook(slc, looks_y=0, looks_x=2)

    def test_name_preserved(self):
        data = xr.DataArray(
            np.ones((8, 8)),
            dims=["y", "x"],
            coords={"y": np.arange(8.0), "x": np.arange(8.0)},
            name="my_data",
        )
        ml = multilook(data, looks_y=2, looks_x=2)
        assert ml.name == "my_data"


class TestMultilookInterferogram:
    def test_output_is_complex(self):
        slc1, slc2 = _make_slc_pair(ny=32, nx=40)
        ml_ifg = multilook_interferogram(slc1, slc2, looks_y=4, looks_x=4)
        assert np.iscomplexobj(ml_ifg.values)

    def test_shape_downsampled(self):
        slc1, slc2 = _make_slc_pair(ny=32, nx=40)
        ml_ifg = multilook_interferogram(slc1, slc2, looks_y=4, looks_x=5)
        assert ml_ifg.shape == (8, 8)

    def test_name_and_attrs(self):
        slc1, slc2 = _make_slc_pair()
        ml_ifg = multilook_interferogram(slc1, slc2, looks_y=2, looks_x=2)
        assert ml_ifg.name == "interferogram"
        assert "Multilooked" in ml_ifg.attrs["long_name"]

    def test_phase_preserved(self):
        # Constant phase difference — multilooking should preserve it
        phase_diff = 1.0
        slc1, slc2 = _make_slc_pair(ny=32, nx=32, phase_diff=phase_diff)
        ml_ifg = multilook_interferogram(slc1, slc2, looks_y=4, looks_x=4)
        mean_phase = np.angle(np.mean(ml_ifg.values))
        np.testing.assert_allclose(mean_phase, -phase_diff, atol=0.3)

    def test_no_looks_matches_interferogram(self):
        slc1, slc2 = _make_slc_pair(ny=16, nx=20)
        ifg = interferogram(slc1, slc2)
        ml_ifg = multilook_interferogram(slc1, slc2, looks_y=1, looks_x=1)
        np.testing.assert_allclose(ml_ifg.values, ifg.values, atol=1e-6)
