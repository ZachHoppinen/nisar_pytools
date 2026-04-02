"""Phase filtering for interferograms.

Implements the Goldstein adaptive filter for reducing phase noise
in wrapped interferograms.

Reference:
    Goldstein & Werner 1998, DOI: 10.1029/1998GL900033
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter


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

    The filter operates on the complex interferogram directly (not
    phase-only), so the adaptive weight is sensitive to local SNR.
    The output preserves the filtered amplitude — both phase and
    amplitude are modified by the filter.

    The input must be an in-memory (non-dask) 2D complex array.
    Call ``.compute()`` on dask-backed arrays before passing.

    Parameters
    ----------
    igram : xr.DataArray
        Complex interferogram (2D, in-memory).
    alpha : float
        Filter strength in [0, 1]. 0 = no filtering, 1 = maximum
        filtering. Default 0.5.
    patch_size : int
        Size of the square FFT patches. Default 32.
    overlap : int
        Overlap between adjacent patches. Default 8. Must be less than
        ``patch_size // 2`` to avoid extremely slow processing.

    Returns
    -------
    xr.DataArray
        Filtered complex interferogram, same shape and coordinates.

    Raises
    ------
    ValueError
        If input is not 2D, not complex, or parameters are invalid.
    """
    # Input validation
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if overlap >= patch_size // 2:
        raise ValueError(
            f"overlap ({overlap}) must be < patch_size//2 ({patch_size // 2}) "
            f"to avoid extremely slow processing"
        )

    # Ensure in-memory 2D complex array
    data = np.asarray(igram)
    if data.ndim != 2:
        raise ValueError(f"igram must be 2D, got {data.ndim}D")
    if not np.iscomplexobj(data):
        raise ValueError("igram must be complex-valued")

    ny, nx = data.shape

    # Output accumulator and weight
    filtered = np.zeros_like(data)
    weights = np.zeros((ny, nx), dtype=np.float64)

    step = patch_size - overlap

    # Periodic Hanning window (correct for overlap-add reconstruction)
    window = _hanning_2d_periodic(patch_size)

    for y0 in range(0, ny, step):
        for x0 in range(0, nx, step):
            y1 = min(y0 + patch_size, ny)
            x1 = min(x0 + patch_size, nx)
            py = y1 - y0
            px = x1 - x0

            # Extract patch from the complex interferogram directly
            patch = data[y0:y1, x0:x1]

            # Skip empty/zero patches
            if np.all(patch == 0):
                continue

            # FFT (zero-pad if patch is smaller than patch_size)
            spectrum = np.fft.fft2(patch, s=(patch_size, patch_size))

            # Smooth the power spectrum
            power = np.abs(spectrum)
            smooth_power = _smooth_spectrum(power)

            # Apply the Goldstein weighting
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = np.where(smooth_power > 0, smooth_power**alpha, 0.0)
            filtered_spectrum = spectrum * weight

            # Inverse FFT — only use the valid (non-padded) region
            filtered_patch = np.fft.ifft2(filtered_spectrum)[:py, :px]

            # Apply window and accumulate
            win = window[:py, :px]
            filtered[y0:y1, x0:x1] += filtered_patch * win
            weights[y0:y1, x0:x1] += win

    # Normalize by accumulated weights
    with np.errstate(divide="ignore", invalid="ignore"):
        filtered = np.where(weights > 0, filtered / weights, 0.0)

    result = xr.DataArray(
        filtered.astype(np.complex64),
        dims=igram.dims,
        coords=igram.coords,
        name=igram.name or "filtered_interferogram",
        attrs=dict(igram.attrs),
    )
    result.attrs["goldstein_alpha"] = alpha
    return result


def _hanning_2d_periodic(size: int) -> np.ndarray:
    """Create a 2D periodic Hanning window for overlap-add.

    Uses a periodic window (no zeros at endpoints) so that overlapping
    windows sum to a constant.
    """
    # Periodic Hanning: hann(N+1)[:-1]
    w1d = np.hanning(size + 1)[:-1]
    return np.outer(w1d, w1d).astype(np.float64)


def _smooth_spectrum(power: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Smooth a power spectrum with a uniform box filter."""
    return uniform_filter(power, size=kernel_size)
