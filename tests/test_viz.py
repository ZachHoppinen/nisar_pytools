"""Tests for nisar_pytools.viz.plotting."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from nisar_pytools.viz.plotting import (
    plot_amplitude,
    plot_coherence,
    plot_interferogram,
    plot_phase,
)


def _make_complex_da(ny=16, nx=20):
    rng = np.random.default_rng(42)
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
    data = (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))).astype(np.complex64)
    return xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})


def _make_real_da(ny=16, nx=20, vmin=0.0, vmax=1.0):
    rng = np.random.default_rng(42)
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -100.0 + 4500000.0
    data = rng.uniform(vmin, vmax, (ny, nx)).astype(np.float32)
    return xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})


class TestPlotAmplitude:
    def test_returns_figure(self):
        fig = plot_amplitude(_make_complex_da())
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_db_mode(self):
        fig = plot_amplitude(_make_complex_da(), db=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_linear_mode(self):
        fig = plot_amplitude(_make_complex_da(), db=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_ax(self):
        fig, ax = plt.subplots()
        result = plot_amplitude(_make_complex_da(), ax=ax)
        assert result is fig
        plt.close(fig)

    def test_real_input(self):
        fig = plot_amplitude(_make_real_da(vmin=0.1, vmax=2.0))
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotPhase:
    def test_complex_input(self):
        fig = plot_phase(_make_complex_da())
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_real_input(self):
        phase = _make_real_da(vmin=-np.pi, vmax=np.pi)
        fig = plot_phase(phase)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_title(self):
        fig = plot_phase(_make_complex_da(), title="My Phase")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotInterferogram:
    def test_returns_figure(self):
        fig = plot_interferogram(_make_complex_da())
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_cmap(self):
        fig = plot_interferogram(_make_complex_da(), cmap="hsv")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotCoherence:
    def test_returns_figure(self):
        fig = plot_coherence(_make_real_da(vmin=0, vmax=1))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_custom_cmap(self):
        fig = plot_coherence(_make_real_da(), cmap="viridis")
        assert isinstance(fig, Figure)
        plt.close(fig)
