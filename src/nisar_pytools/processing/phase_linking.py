"""Phase linking via the EMI (Eigenvalue-based Maximum-likelihood) estimator.

Estimates consistent phase histories from a stack of co-registered SLC images
by identifying statistically homogeneous pixels (SHP) and solving for the
optimal phase using the minimum eigenvector of the whitened coherence matrix.

Reference:
    Ansari, H., De Zan, F., & Bamler, R. (2018). Efficient Phase Estimation
    for Interferogram Stacks. IEEE TGRS, 56(7), 4109-4125.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage
import scipy.stats
import xarray as xr


def estimate_coherence_matrix(slc_pixels: np.ndarray) -> np.ndarray:
    """Estimate the sample coherence matrix from SLC pixel samples.

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

    Computes the whitened matrix ``W = inv(|C|) * C``, finds
    the eigenvector corresponding to the minimum eigenvalue,
    and extracts relative phases referenced to the first image.

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
    eigenvalues, eigenvectors = np.linalg.eig(W)

    min_idx = np.argmin(np.abs(eigenvalues))
    min_eigvec = eigenvectors[:, min_idx]

    # Extract phases relative to first image
    phases = np.angle(min_eigvec * min_eigvec[0].conj())
    return phases.real.astype(np.float32)


def identify_shp(
    amplitude_variance: xr.DataArray,
    ref_variance: xr.DataArray,
    threshold: float,
) -> xr.DataArray:
    """Identify statistically homogeneous pixels using a GLRT test.

    Compares the amplitude variance at each pixel against a reference
    pixel's variance using a generalized likelihood ratio test.

    Parameters
    ----------
    amplitude_variance : xr.DataArray
        Per-pixel amplitude variance (2D).
    ref_variance : xr.DataArray
        Amplitude variance at the reference pixel (scalar or broadcastable).
    threshold : float
        GLRT threshold. Pixels with test statistic below this are SHP.

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

    Parameters
    ----------
    slc_stack : xr.DataArray
        3D complex SLC stack with dimensions ``(time, y, x)``.
        The ``time`` dimension indexes acquisitions.
    window_size : int
        Half-width of the spatial search window in coordinate units.
        The full window is ``2 * window_size + 1`` pixels if pixel spacing
        is 1, but selection is done via coordinate slicing.
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
    n_images = slc_stack.sizes["time"]
    y_vals = slc_stack.y.values
    x_vals = slc_stack.x.values
    ny = len(y_vals)
    nx = len(x_vals)

    # Precompute GLRT threshold and amplitude variance
    gamma = scipy.stats.chi2.ppf(confidence, df=1) / (2 * n_images)
    amp_variance = (np.abs(slc_stack) ** 2).sum("time") / (2 * n_images)

    # Determine pixel spacing for window indexing
    dx = np.abs(x_vals[1] - x_vals[0]) if nx > 1 else 1.0
    dy = np.abs(y_vals[1] - y_vals[0]) if ny > 1 else 1.0
    half_x = window_size * dx
    half_y = window_size * dy

    # Output arrays
    linked_data = np.zeros((n_images, ny, nx), dtype=np.complex64)
    coh_data = np.zeros((n_images, ny, nx), dtype=np.float32)

    # Connected component structure for 8-connectivity
    struct = np.ones((3, 3), dtype=int)

    for iy in range(ny):
        y = y_vals[iy]
        # Slice the y window once per row
        if y_vals[0] < y_vals[-1]:  # ascending
            y_slice = slice(y - half_y, y + half_y)
        else:  # descending
            y_slice = slice(y + half_y, y - half_y)

        for ix in range(nx):
            x = x_vals[ix]

            # Extract spatial window
            window = slc_stack.sel(
                x=slice(x - half_x, x + half_x),
                y=y_slice,
            )
            var_window = amp_variance.sel(
                x=slice(x - half_x, x + half_x),
                y=y_slice,
            )

            if window.sizes["y"] == 0 or window.sizes["x"] == 0:
                continue

            # SHP identification via GLRT
            ref_var = amp_variance.sel(x=x, y=y, method="nearest")
            shp_mask = identify_shp(var_window, ref_var, gamma)

            # 8-connected component containing the reference pixel
            labeled, _ = scipy.ndimage.label(shp_mask.values, structure=struct)
            labeled_da = xr.DataArray(labeled, coords=shp_mask.coords, dims=shp_mask.dims)
            ref_label = labeled_da.sel(x=x, y=y, method="nearest").item()

            if ref_label == 0:
                # Reference pixel not in any SHP region — use just itself
                pixel_vals = slc_stack.sel(x=x, y=y, method="nearest").values
                linked_data[:, iy, ix] = pixel_vals
                continue

            connected_mask = labeled_da.values == ref_label

            # Extract SHP pixels: (n_images, n_shp_pixels)
            pixels = window.values[:, connected_mask]

            if pixels.shape[1] < 2:
                pixel_vals = slc_stack.sel(x=x, y=y, method="nearest").values
                linked_data[:, iy, ix] = pixel_vals
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
