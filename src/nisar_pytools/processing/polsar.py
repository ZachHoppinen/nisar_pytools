"""Polarimetric SAR decomposition: H-A-alpha (Cloude-Pottier).

Computes entropy (H), anisotropy (A), and alpha angle from quad-pol
SLC data using the covariance → coherency matrix path.

All functions are fully vectorized and operate on xarray DataArrays.

References:
    Cloude & Pottier 1997, DOI: 10.1109/36.551935
    Nielsen 2022, DOI: 10.1109/LGRS.2022.3169994
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def covariance_elements(
    hh: xr.DataArray,
    hv: xr.DataArray,
    vv: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """Compute the six unique covariance matrix elements from SLC channels.

    Parameters
    ----------
    hh, hv, vv : xr.DataArray
        Complex SLC polarization channels (2D, matching grids).

    Returns
    -------
    dict
        Keys: ``HHHH``, ``HHHV``, ``HVHV``, ``HVVV``, ``HHVV``, ``VVVV``.
        Values are DataArrays on the same grid.
    """
    return {
        "HHHH": (hh * np.conj(hh)).real.astype(np.float32),
        "HHHV": hh * np.conj(hv),
        "HVHV": (hv * np.conj(hv)).real.astype(np.float32),
        "HVVV": hv * np.conj(vv),
        "HHVV": hh * np.conj(vv),
        "VVVV": (vv * np.conj(vv)).real.astype(np.float32),
    }


def _build_C3(
    HHHH: np.ndarray,
    HHHV: np.ndarray,
    HVHV: np.ndarray,
    HVVV: np.ndarray,
    HHVV: np.ndarray,
    VVVV: np.ndarray,
) -> np.ndarray:
    """Build the 3x3 covariance matrix C3 for all pixels.

    Returns array of shape ``(3, 3, ny, nx)`` with complex dtype.
    """
    sqrt2 = np.sqrt(2.0)
    c12 = sqrt2 * HHHV
    c23 = sqrt2 * HVVV

    C3 = np.array(
        [
            [HHHH, c12, HHVV],
            [np.conj(c12), 2 * HVHV, c23],
            [np.conj(HHVV), np.conj(c23), VVVV],
        ],
        dtype=np.complex64,
    )
    return C3


def _C3_to_T3(C3: np.ndarray) -> np.ndarray:
    """Convert covariance matrix C3 to coherency matrix T3.

    Input/output shape: ``(3, 3, ny, nx)``.

    Translation of the PolSARPro C3_to_T3 function.
    """
    sqrt2 = np.sqrt(2.0)
    t11 = 0.5 * (C3[0, 0] + 2 * C3[0, 2].real + C3[2, 2])
    t22 = 0.5 * (C3[0, 0] - 2 * C3[0, 2].real + C3[2, 2])
    t33 = C3[1, 1]

    t21 = 0.5 * (C3[0, 0] - C3[2, 2]) + (-C3[0, 2].imag) * 1j
    t31 = (C3[0, 1].real + C3[1, 2].real + (C3[0, 1].imag - C3[1, 2].imag) * 1j) / sqrt2
    t32 = (C3[0, 1].real - C3[1, 2].real + (C3[0, 1].imag + C3[1, 2].imag) * 1j) / sqrt2

    T3 = np.array(
        [
            [t11, np.conj(t21), np.conj(t31)],
            [t21, t22, np.conj(t32)],
            [t31, t32, t33],
        ],
        dtype=np.complex64,
    )
    return T3


def _eigvals_T3(T3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sorted eigenvalues of T3 (ascending → t3, t2, t1)."""
    # Transpose to (ny, nx, 3, 3) for np.linalg.eigvalsh
    T3_t = np.transpose(T3, (2, 3, 0, 1))
    vals = np.linalg.eigvalsh(T3_t)  # (ny, nx, 3), ascending
    t3 = vals[..., 0]
    t2 = vals[..., 1]
    t1 = vals[..., 2]
    return t3, t2, t1


def entropy(
    hh: xr.DataArray,
    hv: xr.DataArray,
    vv: xr.DataArray,
) -> xr.DataArray:
    """Compute polarimetric entropy (H) from quad-pol SLC channels.

    H = 0 indicates a single scattering mechanism; H = 1 indicates
    a random mixture of mechanisms.

    Parameters
    ----------
    hh, hv, vv : xr.DataArray
        Complex SLC polarization channels.

    Returns
    -------
    xr.DataArray
        Entropy in [0, 1], float32.
    """
    T3 = _slcs_to_T3(hh.values, hv.values, vv.values)
    t3, t2, t1 = _eigvals_T3(T3)

    total = t1 + t2 + t3
    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = np.where(total > 0, t1 / total, 0)
        p2 = np.where(total > 0, t2 / total, 0)
        p3 = np.where(total > 0, t3 / total, 0)

    H = np.zeros_like(p1)
    for p in (p1, p2, p3):
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p > 0, np.log(p) / np.log(3), 0)
        H -= p * logp

    H = np.clip(np.nan_to_num(H, nan=np.nan), 0, 1).astype(np.float32)

    return xr.DataArray(
        H, dims=hh.dims, coords=hh.coords,
        name="entropy", attrs={"units": "1", "long_name": "Polarimetric entropy"},
    )


def anisotropy(
    hh: xr.DataArray,
    hv: xr.DataArray,
    vv: xr.DataArray,
) -> xr.DataArray:
    """Compute polarimetric anisotropy (A) from quad-pol SLC channels.

    A measures the relative importance of the second and third
    scattering mechanisms.

    Parameters
    ----------
    hh, hv, vv : xr.DataArray
        Complex SLC polarization channels.

    Returns
    -------
    xr.DataArray
        Anisotropy in [0, 1], float32.
    """
    T3 = _slcs_to_T3(hh.values, hv.values, vv.values)
    t3, t2, t1 = _eigvals_T3(T3)

    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where((t2 + t3) > 0, (t2 - t3) / (t2 + t3), 0)

    A = np.clip(np.nan_to_num(A, nan=np.nan), 0, 1).astype(np.float32)

    return xr.DataArray(
        A, dims=hh.dims, coords=hh.coords,
        name="anisotropy", attrs={"units": "1", "long_name": "Polarimetric anisotropy"},
    )


def alpha(
    hh: xr.DataArray,
    hv: xr.DataArray,
    vv: xr.DataArray,
) -> xr.DataArray:
    """Compute alpha1 scattering angle from quad-pol SLC channels.

    Uses the eigenvector-eigenvalue identity method (Nielsen 2022).

    Parameters
    ----------
    hh, hv, vv : xr.DataArray
        Complex SLC polarization channels.

    Returns
    -------
    xr.DataArray
        Alpha1 angle in degrees, float32.
    """
    T3 = _slcs_to_T3(hh.values, hv.values, vv.values)
    alpha_vals = _alpha1_from_T3(T3)

    return xr.DataArray(
        alpha_vals, dims=hh.dims, coords=hh.coords,
        name="alpha", attrs={"units": "degrees", "long_name": "Alpha1 scattering angle"},
    )


def mean_alpha(
    hh: xr.DataArray,
    hv: xr.DataArray,
    vv: xr.DataArray,
) -> xr.DataArray:
    """Compute mean alpha scattering angle from quad-pol SLC channels.

    Weighted average of the three alpha angles using eigenvalue weights
    (Nielsen 2022).

    Parameters
    ----------
    hh, hv, vv : xr.DataArray
        Complex SLC polarization channels.

    Returns
    -------
    xr.DataArray
        Mean alpha angle in degrees, float32.
    """
    T3 = _slcs_to_T3(hh.values, hv.values, vv.values)
    alpha_vals = _mean_alpha_from_T3(T3)

    return xr.DataArray(
        alpha_vals, dims=hh.dims, coords=hh.coords,
        name="mean_alpha",
        attrs={"units": "degrees", "long_name": "Mean alpha scattering angle"},
    )


def h_a_alpha(
    hh: xr.DataArray,
    hv: xr.DataArray,
    vv: xr.DataArray,
) -> xr.Dataset:
    """Compute full H-A-alpha decomposition from quad-pol SLC channels.

    Parameters
    ----------
    hh, hv, vv : xr.DataArray
        Complex SLC polarization channels (2D, matching grids).

    Returns
    -------
    xr.Dataset
        Contains ``entropy``, ``anisotropy``, ``alpha``, and ``mean_alpha``.
    """
    T3 = _slcs_to_T3(hh.values, hv.values, vv.values)
    t3, t2, t1 = _eigvals_T3(T3)

    # Entropy
    total = t1 + t2 + t3
    with np.errstate(divide="ignore", invalid="ignore"):
        ps = [np.where(total > 0, t / total, 0) for t in (t1, t2, t3)]
    H = np.zeros_like(ps[0])
    for p in ps:
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p > 0, np.log(p) / np.log(3), 0)
        H -= p * logp
    H = np.clip(np.nan_to_num(H, nan=np.nan), 0, 1).astype(np.float32)

    # Anisotropy
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where((t2 + t3) > 0, (t2 - t3) / (t2 + t3), 0)
    A = np.clip(np.nan_to_num(A, nan=np.nan), 0, 1).astype(np.float32)

    # Alpha1 and mean alpha
    a1 = _alpha1_from_T3(T3)
    ma = _mean_alpha_from_T3(T3)

    coords = hh.coords
    dims = hh.dims
    return xr.Dataset(
        {
            "entropy": xr.DataArray(
                H, dims=dims, coords=coords,
                attrs={"units": "1", "long_name": "Polarimetric entropy"},
            ),
            "anisotropy": xr.DataArray(
                A, dims=dims, coords=coords,
                attrs={"units": "1", "long_name": "Polarimetric anisotropy"},
            ),
            "alpha": xr.DataArray(
                a1, dims=dims, coords=coords,
                attrs={"units": "degrees", "long_name": "Alpha1 scattering angle"},
            ),
            "mean_alpha": xr.DataArray(
                ma, dims=dims, coords=coords,
                attrs={"units": "degrees", "long_name": "Mean alpha scattering angle"},
            ),
        }
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _slcs_to_T3(
    hh: np.ndarray, hv: np.ndarray, vv: np.ndarray
) -> np.ndarray:
    """SLC arrays → coherency matrix T3, shape (3, 3, ny, nx)."""
    HHHH = (hh * np.conj(hh)).real
    HHHV = hh * np.conj(hv)
    HVHV = (hv * np.conj(hv)).real
    HVVV = hv * np.conj(vv)
    HHVV = hh * np.conj(vv)
    VVVV = (vv * np.conj(vv)).real

    C3 = _build_C3(HHHH, HHHV, HVHV, HVVV, HHVV, VVVV)
    return _C3_to_T3(C3)


def _alpha1_from_T3(T3: np.ndarray) -> np.ndarray:
    """Compute alpha1 from T3 using the Nielsen 2022 identity method."""
    T3_t = np.transpose(T3, (2, 3, 0, 1))
    M1 = T3_t[..., 1:, 1:]

    t3, t2, t1 = _eigvals_T3(T3)
    m_vals = np.linalg.eigvalsh(M1)  # (ny, nx, 2)
    m2 = m_vals[..., 0]
    m1 = m_vals[..., 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        e11_sq = ((t1 - m1) * (t1 - m2)) / ((t1 - t2) * (t1 - t3))
    e11 = np.sqrt(np.clip(e11_sq.real, 0, 1))
    alpha1 = np.rad2deg(np.arccos(np.clip(e11, 0, 1)))

    return np.nan_to_num(alpha1, nan=np.nan).astype(np.float32)


def _mean_alpha_from_T3(T3: np.ndarray) -> np.ndarray:
    """Compute mean alpha from T3 using the Nielsen 2022 identity method."""
    T3_t = np.transpose(T3, (2, 3, 0, 1))
    M1 = T3_t[..., 1:, 1:]

    t3, t2, t1 = _eigvals_T3(T3)
    m_vals = np.linalg.eigvalsh(M1)
    m2 = m_vals[..., 0]
    m1 = m_vals[..., 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        e11_sq = ((t1 - m1) * (t1 - m2)) / ((t1 - t2) * (t1 - t3))
        e21_sq = ((t2 - m1) * (t2 - m2)) / ((t2 - t1) * (t2 - t3))
        e31_sq = ((t3 - m1) * (t3 - m2)) / ((t3 - t1) * (t3 - t2))

    a1 = np.arccos(np.clip(np.sqrt(np.clip(e11_sq.real, 0, 1)), 0, 1))
    a2 = np.arccos(np.clip(np.sqrt(np.clip(e21_sq.real, 0, 1)), 0, 1))
    a3 = np.arccos(np.clip(np.sqrt(np.clip(e31_sq.real, 0, 1)), 0, 1))

    total = t1 + t2 + t3
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = np.where(total > 0, t1 / total, 0)
        w2 = np.where(total > 0, t2 / total, 0)
        w3 = np.where(total > 0, t3 / total, 0)

    ma = np.rad2deg(w1 * a1 + w2 * a2 + w3 * a3)

    return np.nan_to_num(ma, nan=np.nan).astype(np.float32)
