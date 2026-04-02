"""Tests for nisar_pytools.processing.phase_linking."""

import numpy as np
import pandas as pd
import xarray as xr

from nisar_pytools.processing.phase_linking import (
    emi,
    estimate_coherence_matrix,
    identify_shp,
    phase_link,
)


def _make_slc_stack(n_images=5, ny=16, nx=16, seed=42):
    """Create a synthetic 3D SLC stack (time, y, x) with a known phase history."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n_images, freq="12D")
    x = np.arange(nx, dtype="f8") * 10.0
    y = np.arange(ny, dtype="f8") * -10.0 + 1000.0

    # Constant phase ramp across time: 0, 0.5, 1.0, 1.5, ...
    true_phases = np.arange(n_images, dtype="f4") * 0.5
    amp = 1.0
    data = np.zeros((n_images, ny, nx), dtype=np.complex64)
    for t in range(n_images):
        signal = amp * np.exp(1j * true_phases[t])
        noise = rng.normal(0, 0.05, (ny, nx)) + 1j * rng.normal(0, 0.05, (ny, nx))
        data[t] = signal + noise

    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": times, "y": y, "x": x},
    ), true_phases


class TestEstimateCoherenceMatrix:
    def test_shape(self):
        rng = np.random.default_rng(0)
        pixels = (rng.normal(size=(4, 20)) + 1j * rng.normal(size=(4, 20))).astype(np.complex64)
        C = estimate_coherence_matrix(pixels)
        assert C.shape == (4, 4)

    def test_unit_diagonal(self):
        rng = np.random.default_rng(0)
        pixels = (rng.normal(size=(5, 50)) + 1j * rng.normal(size=(5, 50))).astype(np.complex64)
        C = estimate_coherence_matrix(pixels)
        np.testing.assert_allclose(np.abs(np.diag(C)), 1.0, atol=1e-6)

    def test_hermitian(self):
        rng = np.random.default_rng(0)
        pixels = (rng.normal(size=(4, 30)) + 1j * rng.normal(size=(4, 30))).astype(np.complex64)
        C = estimate_coherence_matrix(pixels)
        np.testing.assert_allclose(C, C.T.conj(), atol=1e-6)

    def test_identical_signals_give_ones(self):
        # All pixels identical across images → coherence = 1
        signal = np.exp(1j * np.array([0, 0.5, 1.0, 1.5]))
        pixels = np.tile(signal[:, None], (1, 10))
        C = estimate_coherence_matrix(pixels)
        np.testing.assert_allclose(np.abs(C), 1.0, atol=1e-6)

    def test_values_bounded(self):
        rng = np.random.default_rng(1)
        pixels = (rng.normal(size=(6, 100)) + 1j * rng.normal(size=(6, 100))).astype(np.complex64)
        C = estimate_coherence_matrix(pixels)
        assert np.all(np.abs(C) <= 1.0 + 1e-6)


class TestEMI:
    def test_first_phase_is_zero(self):
        rng = np.random.default_rng(0)
        pixels = (rng.normal(size=(5, 100)) + 1j * rng.normal(size=(5, 100))).astype(np.complex64)
        C = estimate_coherence_matrix(pixels)
        phases = emi(C)
        np.testing.assert_allclose(phases[0], 0.0, atol=1e-5)

    def test_recovers_known_phases(self):
        # Construct a coherence matrix from known phases
        true_phases = np.array([0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        # Perfect coherence matrix
        phasors = np.exp(1j * true_phases)
        C = np.outer(phasors, phasors.conj())
        # Add slight noise to make it realistic
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, C.shape) + 1j * rng.normal(0, 0.01, C.shape)
        C = C + noise
        C = (C + C.T.conj()) / 2  # keep hermitian
        np.fill_diagonal(C, 1.0)

        phases = emi(C)
        # Check phase differences (EMI recovers relative phases)
        phase_diffs = np.diff(phases)
        expected_diffs = np.diff(true_phases)
        np.testing.assert_allclose(phase_diffs, expected_diffs, atol=0.15)

    def test_output_shape(self):
        C = np.eye(7, dtype=np.complex128)
        phases = emi(C)
        assert phases.shape == (7,)
        assert phases.dtype == np.float32


class TestIdentifySHP:
    def test_identical_variance_all_shp(self):
        var = xr.DataArray(
            np.full((5, 5), 1.0),
            dims=["y", "x"],
            coords={"y": np.arange(5.0), "x": np.arange(5.0)},
        )
        ref = xr.DataArray(1.0)
        mask = identify_shp(var, ref, threshold=0.5)
        assert mask.all()

    def test_different_variance_excluded(self):
        var_vals = np.full((5, 5), 1.0)
        var_vals[0, 0] = 100.0  # very different
        var = xr.DataArray(var_vals, dims=["y", "x"], coords={"y": np.arange(5.0), "x": np.arange(5.0)})
        ref = xr.DataArray(1.0)
        mask = identify_shp(var, ref, threshold=0.5)
        assert not mask.values[0, 0]
        assert mask.values[2, 2]

    def test_output_is_boolean(self):
        var = xr.DataArray(np.ones((3, 3)), dims=["y", "x"], coords={"y": np.arange(3.0), "x": np.arange(3.0)})
        ref = xr.DataArray(1.0)
        mask = identify_shp(var, ref, threshold=1.0)
        assert mask.dtype == bool


class TestPhaseLink:
    def test_output_shapes(self):
        stack, _ = _make_slc_stack(n_images=4, ny=6, nx=6)
        linked, coh = phase_link(stack, window_size=3)
        assert linked.shape == stack.shape
        assert coh.shape == stack.shape
        assert linked.dims == ("time", "y", "x")

    def test_output_is_complex(self):
        stack, _ = _make_slc_stack(n_images=4, ny=6, nx=6)
        linked, _ = phase_link(stack, window_size=3)
        assert np.iscomplexobj(linked.values)

    def test_coords_preserved(self):
        stack, _ = _make_slc_stack(n_images=4, ny=6, nx=6)
        linked, coh = phase_link(stack, window_size=3)
        np.testing.assert_array_equal(linked.x.values, stack.x.values)
        np.testing.assert_array_equal(linked.y.values, stack.y.values)
        np.testing.assert_array_equal(linked.time.values, stack.time.values)

    def test_recovers_phase_trend(self):
        """Phase-linked output should preserve the linear phase ramp."""
        stack, true_phases = _make_slc_stack(n_images=5, ny=8, nx=8, seed=0)
        linked, _ = phase_link(stack, window_size=4)

        # Extract phase at center pixel
        center_y = stack.y.values[4]
        center_x = stack.x.values[4]
        linked_phases = np.angle(linked.sel(y=center_y, x=center_x).values)

        # Remove offset (phase linking is relative)
        linked_phases -= linked_phases[0]
        true_shifted = true_phases - true_phases[0]

        np.testing.assert_allclose(linked_phases, true_shifted, atol=0.3)

    def test_names_and_attrs(self):
        stack, _ = _make_slc_stack(n_images=3, ny=6, nx=6)
        linked, coh = phase_link(stack, window_size=3)
        assert linked.name == "phase_linked"
        assert coh.name == "temporal_coherence"
