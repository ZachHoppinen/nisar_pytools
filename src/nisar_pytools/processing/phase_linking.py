"""Phase linking via the EMI (Eigenvalue-based Maximum-likelihood) estimator.

Estimates consistent phase histories from a stack of co-registered SLC images
by identifying statistically homogeneous pixels (SHP) and solving for the
optimal phase using the minimum eigenvector of the whitened coherence matrix.

.. warning::
    The ``phase_link`` function uses a per-pixel Python loop and is **not
    suitable for full-resolution processing** of large scenes. It is intended
    for small subsets or prototyping. A vectorized implementation using
    batched FFT/window extraction would be needed for production use.

Reference:
    Ansari, H., De Zan, F., & Bamler, R. (2018). Efficient Phase Estimation
    for Interferogram Stacks. IEEE TGRS, 56(7), 4109-4125.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import scipy.ndimage
import scipy.stats
import xarray as xr

log = logging.getLogger(__name__)


def estimate_coherence_matrix(slc_pixels: np.ndarray) -> np.ndarray:
    """Estimate the sample coherence matrix from SLC pixel samples.

    Computes the sample correlation matrix:
    ``C[i,j] = (Σ s_i * s_j*) / sqrt(Σ|s_i|² · Σ|s_j|²)``

    This is a vector-norm normalization (not per-pixel-pair). For SHP-based
    phase linking this is standard practice, as the SHP set is already
    filtered for statistical homogeneity.

    Parameters
    ----------
    slc_pixels : np.ndarray, shape (n_images, n_pixels)
        Complex SLC values for a set of pixels across all images.

    Returns
    -------
    np.ndarray, shape (n_images, n_images)
        Normalized sample coherence matrix with unit diagonal.
    """
    norms = np.linalg.norm(slc_pixels, axis=1)
    denom = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(denom > 0, np.dot(slc_pixels, slc_pixels.T.conj()) / denom, 0.0)
    return C


def emi(coherence_matrix: np.ndarray) -> np.ndarray:
    """Estimate linked phases using the EMI algorithm.

    Computes the whitened matrix ``W = inv(|C|) ⊙ C`` (Hadamard product),
    finds the eigenvector corresponding to the minimum eigenvalue using
    ``eigh`` (Hermitian eigendecomposition), and extracts relative phases
    referenced to the first image.

    Parameters
    ----------
    coherence_matrix : np.ndarray, shape (n_images, n_images)
        Sample coherence matrix.

    Returns
    -------
    np.ndarray, shape (n_images,)
        Estimated phases in radians, referenced to the first image (phase = 0).
    """
    C = coherence_matrix
    abs_C = np.abs(C)

    # Regularize to avoid singular matrix
    abs_C_reg = abs_C + np.eye(C.shape[0]) * 1e-10

    # Hadamard (elementwise) product: inv(|C|) ⊙ C
    W = np.linalg.inv(abs_C_reg) * C

    # Force Hermitian (numerical noise can break symmetry)
    W = (W + W.T.conj()) / 2

    # eigh for Hermitian matrices: numerically stable, real eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(W)

    # EMI selects the eigenvector with eigenvalue closest to zero
    # in magnitude — this corresponds to the signal subspace
    min_idx = np.argmin(np.abs(eigenvalues))
    min_eigvec = eigenvectors[:, min_idx]

    # Extract phases relative to first image
    phases = np.angle(min_eigvec * min_eigvec[0].conj())
    return phases.astype(np.float32)


def identify_shp(
    amplitude_variance: xr.DataArray,
    ref_variance: xr.DataArray,
    threshold: float,
) -> xr.DataArray:
    """Identify statistically homogeneous pixels using a GLRT test.

    Compares the amplitude variance at each pixel against a reference
    pixel's variance using a generalized likelihood ratio test for
    equal means of exponential distributions.

    Parameters
    ----------
    amplitude_variance : xr.DataArray
        Per-pixel amplitude variance (2D), computed as
        ``sum(|s|²) / (2 * n_images)`` over the temporal stack.
    ref_variance : xr.DataArray
        Amplitude variance at the reference pixel (scalar).
    threshold : float
        GLRT threshold (derived from chi2 distribution). Pixels with
        test statistic below this are classified as SHP.

    Returns
    -------
    xr.DataArray
        Boolean mask where ``True`` indicates a statistically homogeneous pixel.
    """
    p = ref_variance
    sigma = amplitude_variance
    with np.errstate(divide="ignore", invalid="ignore"):
        T = 2 * np.log((p + sigma) / 2) - np.log(p) - np.log(sigma)
    return T < threshold


def phase_link(
    slc_stack: xr.DataArray,
    window_size: int = 11,
    confidence: float = 0.95,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Phase link a stack of SLC images using EMI with SHP selection.

    For each pixel, identifies statistically homogeneous pixels (SHP) within
    a spatial window using a GLRT test, estimates the coherence matrix from
    those pixels, and applies the EMI algorithm to recover consistent phases.

    .. warning::
        This function uses a per-pixel Python loop and is **not suitable
        for full-resolution scenes**. For a 100×100 subset it runs in
        seconds; for 1000×1000 it may take hours. Use on small subsets
        or for prototyping only.

    Parameters
    ----------
    slc_stack : xr.DataArray
        3D complex SLC stack with dimensions ``(time, y, x)``.
        Must be in-memory (not dask-backed). Call ``.compute()`` first
        if needed.
    window_size : int
        Half-width of the spatial search window in pixels.
        The full window is ``2 * window_size + 1`` pixels.
    confidence : float
        Confidence level for the GLRT SHP test (default 0.95).

    Returns
    -------
    linked : xr.DataArray
        Complex phase-linked SLC stack, shape ``(time, y, x)``.
        Contains amplitude (mean of SHP) times linked phase.
    temporal_coherence : xr.DataArray
        Coherence of the first image with all others from the coherence
        matrix, shape ``(time, y, x)``.
    """
    # Ensure in-memory numpy data
    stack_data = np.asarray(slc_stack)
    if stack_data.ndim != 3:
        raise ValueError(f"slc_stack must be 3D (time, y, x), got {stack_data.ndim}D")

    n_images, ny, nx = stack_data.shape

    if ny * nx > 10000:
        warnings.warn(
            f"Phase linking {ny}×{nx} = {ny*nx} pixels with a per-pixel loop. "
            f"This will be slow. Consider subsetting first.",
            UserWarning,
            stacklevel=2,
        )

    # Precompute GLRT threshold and amplitude variance
    gamma = scipy.stats.chi2.ppf(confidence, df=1) / (2 * n_images)
    amp_var = np.sum(np.abs(stack_data) ** 2, axis=0) / (2 * n_images)

    # Output arrays
    linked_data = np.zeros_like(stack_data)
    coh_data = np.zeros_like(stack_data, dtype=np.float32)

    # Connected component structure for 8-connectivity
    struct = np.ones((3, 3), dtype=int)

    for iy in range(ny):
        for ix in range(nx):
            # Window bounds (index-based, not coordinate-based)
            y0 = max(0, iy - window_size)
            y1 = min(ny, iy + window_size + 1)
            x0 = max(0, ix - window_size)
            x1 = min(nx, ix + window_size + 1)

            # Extract window
            window = stack_data[:, y0:y1, x0:x1]  # (n_images, wy, wx)
            var_window = amp_var[y0:y1, x0:x1]  # (wy, wx)

            # Reference pixel position within window
            ref_iy = iy - y0
            ref_ix = ix - x0
            ref_var = amp_var[iy, ix]

            # SHP identification via GLRT
            with np.errstate(divide="ignore", invalid="ignore"):
                T = 2 * np.log((ref_var + var_window) / 2) - np.log(ref_var) - np.log(var_window)
            shp_mask = T < gamma

            # 8-connected component containing the reference pixel
            labeled, _ = scipy.ndimage.label(shp_mask, structure=struct)
            ref_label = labeled[ref_iy, ref_ix]

            if ref_label == 0:
                linked_data[:, iy, ix] = stack_data[:, iy, ix]
                continue

            connected_mask = labeled == ref_label

            # Extract SHP pixels: (n_images, n_shp)
            pixels = window[:, connected_mask]

            if pixels.shape[1] < 2:
                linked_data[:, iy, ix] = stack_data[:, iy, ix]
                continue

            # Coherence matrix and EMI
            C = estimate_coherence_matrix(pixels)
            phases = emi(C)
            amp = np.mean(np.abs(pixels), axis=1)

            linked_data[:, iy, ix] = amp * np.exp(1j * phases)
            coh_data[:, iy, ix] = np.abs(C[0])

    linked = xr.DataArray(
        linked_data,
        dims=slc_stack.dims,
        coords=slc_stack.coords,
        name="phase_linked",
        attrs={"long_name": "Phase-linked SLC stack"},
    )
    temporal_coherence = xr.DataArray(
        coh_data,
        dims=slc_stack.dims,
        coords=slc_stack.coords,
        name="temporal_coherence",
        attrs={"long_name": "Temporal coherence from phase linking"},
    )

    return linked, temporal_coherence
