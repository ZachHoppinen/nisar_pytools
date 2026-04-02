"""Phase filtering for interferograms.

Implements the Goldstein adaptive filter for reducing phase noise
in wrapped interferograms.

Reference:
    Goldstein & Werner 1998, DOI: 10.1029/1998GL900033
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def goldstein_filter(
    igram: xr.DataArray,
    alpha: float = 0.5,
    patch_size: int = 32,
    overlap: int = 8,
) -> xr.DataArray:
    """Apply the Goldstein adaptive phase filter to an interferogram.

    Filters the interferogram in overlapping patches in the frequency
    domain. Each patch's spectrum is weighted by its smoothed power
    spectrum raised to ``alpha``, suppressing noise while preserving
    signal.

    Parameters
    ----------
    igram : xr.DataArray
        Complex interferogram (2D).
    alpha : float
        Filter strength in [0, 1]. 0 = no filtering, 1 = maximum
        filtering. Default 0.5.
    patch_size : int
        Size of the square FFT patches. Default 32.
    overlap : int
        Overlap between adjacent patches. Default 8. Must be less than
        ``patch_size``.

    Returns
    -------
    xr.DataArray
        Filtered complex interferogram, same shape and coordinates.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if overlap >= patch_size:
        raise ValueError(f"overlap ({overlap}) must be < patch_size ({patch_size})")

    data = igram.values
    ny, nx = data.shape

    # Normalize interferogram to unit amplitude (phase-only)
    amp = np.abs(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_only = np.where(amp > 0, data / amp, 0.0)

    # Output accumulator and weight
    filtered = np.zeros_like(data)
    weights = np.zeros((ny, nx), dtype=np.float32)

    step = patch_size - overlap

    # Window for tapering patch edges
    window = _hanning_2d(patch_size)

    for y0 in range(0, ny, step):
        for x0 in range(0, nx, step):
            y1 = min(y0 + patch_size, ny)
            x1 = min(x0 + patch_size, nx)
            py = y1 - y0
            px = x1 - x0

            patch = phase_only[y0:y1, x0:x1]

            # FFT (zero-pad if patch is smaller than patch_size)
            spectrum = np.fft.fft2(patch, s=(patch_size, patch_size))

            # Smooth the power spectrum
            power = np.abs(spectrum)
            smooth_power = _smooth_spectrum(power)

            # Apply the Goldstein weighting
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = np.where(smooth_power > 0, smooth_power**alpha, 0.0)
            filtered_spectrum = spectrum * weight

            # Inverse FFT and trim to actual patch size
            filtered_patch = np.fft.ifft2(filtered_spectrum)[:py, :px]

            # Apply window and accumulate
            win = window[:py, :px]
            filtered[y0:y1, x0:x1] += filtered_patch * win
            weights[y0:y1, x0:x1] += win

    # Normalize by accumulated weights
    with np.errstate(divide="ignore", invalid="ignore"):
        filtered = np.where(weights > 0, filtered / weights, 0.0)

    # Restore original amplitude
    filtered = (amp * np.exp(1j * np.angle(filtered))).astype(np.complex64)

    result = xr.DataArray(
        filtered,
        dims=igram.dims,
        coords=igram.coords,
        name=igram.name or "filtered_interferogram",
        attrs=dict(igram.attrs),
    )
    result.attrs["goldstein_alpha"] = alpha
    return result


def _hanning_2d(size: int) -> np.ndarray:
    """Create a 2D Hanning (cosine-squared) window."""
    w1d = np.hanning(size)
    return np.outer(w1d, w1d).astype(np.float32)


def _smooth_spectrum(power: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Smooth a power spectrum with a uniform box filter."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(power, size=kernel_size)
