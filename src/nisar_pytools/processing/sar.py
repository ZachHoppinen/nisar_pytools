"""Core SAR processing functions: interferogram, coherence, and unwrapping.

All functions operate on xarray DataArrays and preserve coordinates/CRS.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, uniform_filter

log = logging.getLogger(__name__)


def calculate_phase(data: xr.DataArray) -> xr.DataArray:
    """Extract the phase angle from a complex-valued DataArray.

    Parameters
    ----------
    data : xr.DataArray
        Complex-valued input.

    Returns
    -------
    xr.DataArray
        Phase in radians (float32), same coordinates as input.
    """
    # Use np.angle directly on DataArray to preserve dask graphs if present
    phase = np.angle(data)
    return xr.DataArray(
        phase.values.astype(np.float32) if hasattr(phase, "values") else phase.astype(np.float32),
        dims=data.dims,
        coords=data.coords,
        name="phase",
        attrs={"units": "radians", "long_name": "Phase angle"},
    )


def _check_matching_grids(
    a: xr.DataArray, b: xr.DataArray, rtol: float = 1e-9
) -> None:
    """Raise ValueError if two DataArrays have mismatched x/y coordinates.

    Uses approximate comparison (``allclose``) to handle floating-point
    rounding differences between processing paths.
    """
    if a.x.shape != b.x.shape or not np.allclose(a.x.values, b.x.values, rtol=rtol, atol=0):
        raise ValueError(
            f"x coordinates do not match: "
            f"shapes {a.x.shape} vs {b.x.shape}, "
            f"range [{a.x.values[0]}, {a.x.values[-1]}] vs "
            f"[{b.x.values[0]}, {b.x.values[-1]}]"
        )
    if a.y.shape != b.y.shape or not np.allclose(a.y.values, b.y.values, rtol=rtol, atol=0):
        raise ValueError(
            f"y coordinates do not match: "
            f"shapes {a.y.shape} vs {b.y.shape}, "
            f"range [{a.y.values[0]}, {a.y.values[-1]}] vs "
            f"[{b.y.values[0]}, {b.y.values[-1]}]"
        )


def _antialiased_crossmul(
    slc1: np.ndarray, slc2: np.ndarray, upsample: int = 2
) -> np.ndarray:
    """Range-direction antialiased crossmul matching production NISAR Crossmul.

    Replicates the algorithm in ``isce3::signal::Crossmul`` (file
    cxx/isce3/signal/Crossmul.cpp on the develop branch — the class the
    NISAR L2 ``crossmul`` workflow actually invokes via ``isce3.signal.Crossmul``).
    Note: this is *not* identical to the simpler ``isce3.signal.CrossMultiply``
    class, which uses ``multilookSummed`` for the antialias downsample and
    therefore produces 2x-amplitude output. The production Crossmul uses
    ``sum / oversample`` (mean), giving 1x-amplitude output that's directly
    comparable with naive crossmul.

    Algorithm (axis=1 is range):
        1. Zero-pad columns to a fast FFT length
        2. Range FFT (axis=1 only)
        3. Upsample columns by zero-padding the spectrum to ``fftsize * upsample``,
           using ISCE3's asymmetric split: the original Nyquist bin lands at
           the negative-frequency end of the upsampled spectrum
        4. Range IFFT × upsample → SLCs upsampled in range only
        5. Element-wise conjugate multiply on the upsampled grid
        6. Mean of adjacent ``upsample`` columns ("reclaim the extra oversample
           looks across", Crossmul.cpp:340).

    Why range-only: in critically-sampled SAR products the conjugate-product
    aliasing is dominated by range. Azimuth is over-sampled by the PRF margin,
    so naive multiply doesn't alias significantly along azimuth.
    """
    from scipy.fft import next_fast_len

    in_dtype = slc1.dtype
    n_rows, n_cols = slc1.shape

    fftsize = next_fast_len(n_cols)
    up_cols = fftsize * upsample

    # Zero-pad columns to fftsize (no-op when n_cols is already fast)
    s1_pad = np.zeros((n_rows, fftsize), dtype=np.complex64)
    s2_pad = np.zeros((n_rows, fftsize), dtype=np.complex64)
    s1_pad[:, :n_cols] = slc1
    s2_pad[:, :n_cols] = slc2

    # Range FFT (axis=1 only)
    s1_spec = np.fft.fft(s1_pad, axis=1)
    s2_spec = np.fft.fft(s2_pad, axis=1)

    # Spectrum zero-pad upsample, matching the ISCE3 split convention exactly:
    #   shifted[0 : (fftsize+1)//2]                 = spec[0 : (fftsize+1)//2]
    #   shifted[up_cols - fftsize//2 : up_cols]     = spec[(fftsize+1)//2 : fftsize]
    # For even fftsize this puts the original Nyquist bin (index fftsize/2)
    # at the *negative-frequency* end of the upsampled spectrum (index
    # up_cols - fftsize/2), rather than splitting it between both ends.
    # This is an asymmetric but consistent ISCE3 convention — preserved
    # so our output bit-matches isce3.signal.CrossMultiply.
    half_lo = (fftsize + 1) // 2  # bin count going to the low end
    half_hi = fftsize // 2        # bin count going to the high end
    s1_up_spec = np.zeros((n_rows, up_cols), dtype=np.complex64)
    s2_up_spec = np.zeros((n_rows, up_cols), dtype=np.complex64)
    s1_up_spec[:, :half_lo] = s1_spec[:, :half_lo]
    s1_up_spec[:, up_cols - half_hi:] = s1_spec[:, half_lo:]
    s2_up_spec[:, :half_lo] = s2_spec[:, :half_lo]
    s2_up_spec[:, up_cols - half_hi:] = s2_spec[:, half_lo:]

    # Range IFFT, with amplitude-preserving scale factor (= upsample)
    s1_up = np.fft.ifft(s1_up_spec, axis=1) * upsample
    s2_up = np.fft.ifft(s2_up_spec, axis=1) * upsample

    # Conjugate multiply on upsampled grid
    ifg_up = s1_up * np.conj(s2_up)

    # Mean of adjacent ``upsample`` columns: matches the Crossmul.cpp
    # "Reclaim the extra oversample looks across" loop (line 340 in
    # the develop branch) where ifgram = sum / ov.  Output amplitude
    # is ~1x naive (not 2x like CrossMultiply.cpp's multilookSummed).
    ifg = (
        ifg_up[:, : n_cols * upsample]
        .reshape(n_rows, n_cols, upsample)
        .mean(axis=2)
    )
    return ifg.astype(in_dtype)


def _antialiased_crossmul_2d(
    slc1: np.ndarray, slc2: np.ndarray, upsample: int = 2
) -> np.ndarray:
    """Symmetric 2D antialiased crossmul — upsamples both axes.

    Same algorithm as :func:`_antialiased_crossmul` but applied along
    both axes simultaneously. This is the geometrically appropriate
    antialiasing for GSLCs (or any SAR product on a projected x-y grid),
    where the SAR spectrum's bandlimit is rotated diagonally in (kx, ky)
    rather than aligned with either axis.

    Output amplitude is ~1x naive (the 2x2 cell mean preserves amplitude
    at the original sample positions for low-frequency content).
    """
    from scipy.fft import next_fast_len

    in_dtype = slc1.dtype
    n_rows, n_cols = slc1.shape

    fy = next_fast_len(n_rows)
    fx = next_fast_len(n_cols)
    up_y = fy * upsample
    up_x = fx * upsample

    # Zero-pad to (fy, fx)
    s1_pad = np.zeros((fy, fx), dtype=np.complex64)
    s2_pad = np.zeros((fy, fx), dtype=np.complex64)
    s1_pad[:n_rows, :n_cols] = slc1
    s2_pad[:n_rows, :n_cols] = slc2

    # 2D FFT
    s1_spec = np.fft.fft2(s1_pad)
    s2_spec = np.fft.fft2(s2_pad)

    # 2D zero-pad upsample using ISCE3's asymmetric split convention along
    # each axis. Four spectrum corners get copied into their corresponding
    # corners of the upsampled spectrum.
    lo_y = (fy + 1) // 2
    hi_y = fy // 2
    lo_x = (fx + 1) // 2
    hi_x = fx // 2

    s1_up_spec = np.zeros((up_y, up_x), dtype=np.complex64)
    s2_up_spec = np.zeros((up_y, up_x), dtype=np.complex64)

    for src_spec, dst_spec in ((s1_spec, s1_up_spec), (s2_spec, s2_up_spec)):
        # Top-left: low pos y / low pos x (and DC)
        dst_spec[:lo_y, :lo_x] = src_spec[:lo_y, :lo_x]
        # Top-right: low pos y / high (negative) x
        dst_spec[:lo_y, up_x - hi_x:] = src_spec[:lo_y, lo_x:]
        # Bottom-left: high (negative) y / low pos x
        dst_spec[up_y - hi_y:, :lo_x] = src_spec[lo_y:, :lo_x]
        # Bottom-right: high y / high x
        dst_spec[up_y - hi_y:, up_x - hi_x:] = src_spec[lo_y:, lo_x:]

    # 2D IFFT, scale by upsample**2 to preserve amplitude at original sample
    # positions (each axis contributes a factor of ``upsample``).
    s1_up = np.fft.ifft2(s1_up_spec) * (upsample ** 2)
    s2_up = np.fft.ifft2(s2_up_spec) * (upsample ** 2)

    # Conjugate multiply on upsampled grid
    ifg_up = s1_up * np.conj(s2_up)

    # Mean of each upsample x upsample cell (matches Crossmul.cpp's mean-based
    # antialias downsample, generalized symmetrically to 2D).
    ifg = (
        ifg_up[: n_rows * upsample, : n_cols * upsample]
        .reshape(n_rows, upsample, n_cols, upsample)
        .mean(axis=(1, 3))
    )
    return ifg.astype(in_dtype)


def interferogram(
    slc1: xr.DataArray,
    slc2: xr.DataArray,
    antialias: bool | str = False,
) -> xr.DataArray:
    """Generate an interferogram from two co-registered SLC images.

    Default behaviour computes the naive ``slc1 * conj(slc2)``. The
    ``antialias`` parameter selects an FFT-based antialiased variant.

    Parameters
    ----------
    slc1, slc2 : xr.DataArray
        Complex-valued SLC images with matching coordinates.
    antialias : bool or str, default False
        Antialiasing mode for the conjugate multiply.

        - ``False`` (or ``'none'``): naive ``slc1 * conj(slc2)``. Fast,
          dask-friendly. The conjugate product can spill outside the
          principal Nyquist band; multilooking absorbs most of the alias
          noise downstream.
        - ``True`` (or ``'range'``): matches ``isce3.signal.Crossmul`` —
          the production NISAR crossmul (file cxx/isce3/signal/Crossmul.cpp).
          Upsamples along axis=1 (range), conjugate-multiplies on the
          upsampled grid, and downsamples by averaging adjacent cells.
          Output amplitude is ~1x naive. (Note: this differs from the
          simpler ``isce3.signal.CrossMultiply`` class, which sums instead
          of averages and therefore gives 2x amplitude.)
        - ``'2d'``: symmetric 2D antialias along both axes. Geometrically
          the right choice for GSLCs in projected (x, y) coordinates,
          where the SAR spectrum's bandlimit is rotated diagonally rather
          than aligned with either axis. Output amplitude is ~1x naive.

        Memory cost for the antialiased modes: peaks at roughly 16x
        (range) or 32x (2d) the input array size during the FFTs. For
        very large inputs, chunk the array yourself before calling.

    Returns
    -------
    xr.DataArray
        Complex interferogram with the same coordinates as the inputs.

    Raises
    ------
    ValueError
        If x or y coordinates do not match, inputs are not complex, or
        ``antialias`` is not one of the recognised options.
    """
    if not np.iscomplexobj(slc1) or not np.iscomplexobj(slc2):
        raise ValueError("SLC inputs must be complex-valued")
    _check_matching_grids(slc1, slc2)

    # Normalise antialias to a canonical string mode
    if antialias is False or antialias == "none":
        mode = "none"
    elif antialias is True or antialias == "range":
        mode = "range"
    elif antialias == "2d":
        mode = "2d"
    else:
        raise ValueError(
            f"antialias must be False, True, 'none', 'range', or '2d'; got {antialias!r}"
        )

    if mode == "none":
        ifg = slc1 * np.conj(slc2)
    else:
        fn = _antialiased_crossmul if mode == "range" else _antialiased_crossmul_2d
        ifg_arr = fn(np.asarray(slc1), np.asarray(slc2))
        ifg = xr.DataArray(
            ifg_arr,
            dims=slc1.dims,
            coords={d: slc1.coords[d] for d in slc1.dims if d in slc1.coords},
        )

    ifg.name = "interferogram"
    ifg.attrs = {"units": "1", "long_name": "Complex interferogram"}
    return ifg


def multilook(
    data: xr.DataArray,
    looks_y: int = 1,
    looks_x: int = 1,
) -> xr.DataArray:
    """Multilook (spatially average and downsample) a 2D array.

    Averages non-overlapping blocks of ``(looks_y, looks_x)`` pixels.
    The output grid is downsampled accordingly, with coordinates
    taken from the block centers.

    If the array dimensions are not exact multiples of the look factors,
    the trailing rows/columns are trimmed (with a debug log message).

    Parameters
    ----------
    data : xr.DataArray
        2D input array with ``y`` and ``x`` dimensions.
    looks_y : int
        Number of looks (pixels to average) in the y direction.
    looks_x : int
        Number of looks (pixels to average) in the x direction.

    Returns
    -------
    xr.DataArray
        Multilooked array with reduced dimensions.
    """
    if looks_y < 1 or looks_x < 1:
        raise ValueError(f"looks must be >= 1, got looks_y={looks_y}, looks_x={looks_x}")
    if looks_y == 1 and looks_x == 1:
        return data.copy()

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"multilook requires 2D input, got {arr.ndim}D")

    ny, nx = arr.shape

    ny_trim = (ny // looks_y) * looks_y
    nx_trim = (nx // looks_x) * looks_x

    trimmed_y = ny - ny_trim
    trimmed_x = nx - nx_trim
    if trimmed_y > 0 or trimmed_x > 0:
        log.debug(
            "Trimming %d rows and %d columns for multilook (%d×%d)",
            trimmed_y, trimmed_x, looks_y, looks_x,
        )

    arr = arr[:ny_trim, :nx_trim]

    # Reshape and average — works for both real and complex dtypes
    out = arr.reshape(ny_trim // looks_y, looks_y, nx_trim // looks_x, looks_x).mean(axis=(1, 3))

    y_out = data.y.values[:ny_trim].reshape(-1, looks_y).mean(axis=1)
    x_out = data.x.values[:nx_trim].reshape(-1, looks_x).mean(axis=1)

    result = xr.DataArray(
        out.astype(arr.dtype),
        dims=["y", "x"],
        coords={"y": y_out, "x": x_out},
        name=data.name,
        attrs=dict(data.attrs),
    )
    return result


def multilook_interferogram(
    slc1: xr.DataArray,
    slc2: xr.DataArray,
    looks_y: int = 1,
    looks_x: int = 1,
) -> xr.DataArray:
    """Generate a multilooked interferogram from two SLC images.

    Computes ``slc1 * conj(slc2)`` then averages over non-overlapping
    blocks of ``(looks_y, looks_x)`` pixels. Averaging the complex
    interferogram (rather than the phase) preserves coherence information.

    This is a convenience wrapper around :func:`interferogram` +
    :func:`multilook` that sets appropriate output metadata.

    Parameters
    ----------
    slc1, slc2 : xr.DataArray
        Complex-valued SLC images with matching coordinates.
    looks_y : int
        Number of looks in the y direction.
    looks_x : int
        Number of looks in the x direction.

    Returns
    -------
    xr.DataArray
        Multilooked complex interferogram on the downsampled grid.
    """
    ifg = interferogram(slc1, slc2)
    ml = multilook(ifg, looks_y=looks_y, looks_x=looks_x)
    ml.name = "interferogram"
    ml.attrs = {"units": "1", "long_name": "Multilooked complex interferogram"}
    return ml


def coherence(
    slc1: xr.DataArray,
    slc2: xr.DataArray,
    window_size: int = 5,
    method: str = "boxcar",
) -> xr.DataArray:
    """Estimate interferometric coherence magnitude over a spatial window.

    Computes::

        |⟨s1 * conj(s2)⟩| / sqrt(⟨|s1|²⟩ * ⟨|s2|²⟩)

    where ⟨·⟩ denotes spatial averaging over the window.

    Parameters
    ----------
    slc1, slc2 : xr.DataArray
        Complex-valued SLC images with matching coordinates.
    window_size : int
        For ``"boxcar"``: side length of the square averaging window.
        Must be odd and >= 1.
        For ``"gaussian"``: sigma (standard deviation) of the Gaussian
        kernel in pixels. Note: the effective kernel radius is ~4×sigma,
        so ``sigma=3`` gives an effective window of ~25 pixels.
    method : str
        Averaging method: ``"boxcar"`` (default) for uniform weighting,
        or ``"gaussian"`` for Gaussian-weighted averaging.

    Returns
    -------
    xr.DataArray
        Coherence magnitude in [0, 1], same coordinates as inputs.
    """
    if not np.iscomplexobj(slc1) or not np.iscomplexobj(slc2):
        raise ValueError("SLC inputs must be complex-valued")
    _check_matching_grids(slc1, slc2)

    if method not in ("boxcar", "gaussian"):
        raise ValueError(f"method must be 'boxcar' or 'gaussian', got '{method}'")

    if method == "boxcar":
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(f"window_size must be odd and >= 1, got {window_size}")
    else:
        if window_size < 1:
            raise ValueError(f"window_size (sigma) must be >= 1, got {window_size}")

    s1 = np.asarray(slc1)
    s2 = np.asarray(slc2)

    ifg = s1 * np.conj(s2)
    pow1 = np.abs(s1) ** 2
    pow2 = np.abs(s2) ** 2

    def _avg(a):
        if method == "boxcar":
            return uniform_filter(a, size=window_size)
        return gaussian_filter(a, sigma=window_size)

    # scipy filters don't support complex input, so we average real and
    # imaginary parts separately. This is mathematically equivalent since
    # averaging is a linear operation.
    avg_ifg = _avg(ifg.real) + 1j * _avg(ifg.imag)
    avg_pow1 = _avg(pow1)
    avg_pow2 = _avg(pow2)

    denom = np.sqrt(avg_pow1 * avg_pow2)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.where(denom > 0, np.abs(avg_ifg) / denom, 0.0)

    coh = np.clip(coh, 0.0, 1.0).astype(np.float32)

    result = xr.DataArray(
        coh,
        dims=slc1.dims,
        coords=slc1.coords,
        name="coherence",
        attrs={"units": "1", "long_name": "Interferometric coherence magnitude"},
    )
    return result


def multilook_coherence(
    slc1: xr.DataArray,
    slc2: xr.DataArray,
    looks_y: int,
    looks_x: int,
    antialias: bool | str = False,
) -> xr.DataArray:
    """Coherence on the multilooked grid, ISCE3 production convention.

    Computes::

        γ = |multilook(s1 · conj(s2))| / sqrt(multilook(|s1|²) · multilook(|s2|²))

    where each ``multilook`` is a non-overlapping mean over
    ``(looks_y, looks_x)`` pixel blocks. The output lives on the multilooked
    grid (downsampled by the look factors), matching what
    ``nisar.workflows.crossmul`` writes into the RIFG ``coherenceMagnitude``
    band.

    This is the ISCE3 production-style estimator. Compare with
    :func:`coherence`, which produces a sliding-window estimate at the
    full SLC resolution.

    Parameters
    ----------
    slc1, slc2 : xr.DataArray
        Complex-valued SLC images with matching coordinates.
    looks_y, looks_x : int
        Multilook factors. Output grid has shape ``(ny // looks_y, nx // looks_x)``.
    antialias : bool or str, default False
        Forwarded to :func:`interferogram`. Set to ``'range'`` (matches
        production NISAR Crossmul) or ``'2d'`` (symmetric, right for GSLCs)
        to suppress full-resolution alias noise before multilooking.
        Numerator and denominator scale consistently regardless of mode,
        so coherence stays in [0, 1].

    Returns
    -------
    xr.DataArray
        Coherence magnitude on the multilooked grid, dtype float32.
    """
    if not np.iscomplexobj(slc1) or not np.iscomplexobj(slc2):
        raise ValueError("SLC inputs must be complex-valued")
    _check_matching_grids(slc1, slc2)

    ifg = interferogram(slc1, slc2, antialias=antialias)
    pow1 = xr.DataArray(
        (np.abs(np.asarray(slc1)) ** 2).astype(np.float32),
        dims=slc1.dims,
        coords={d: slc1.coords[d] for d in slc1.dims if d in slc1.coords},
    )
    pow2 = xr.DataArray(
        (np.abs(np.asarray(slc2)) ** 2).astype(np.float32),
        dims=slc2.dims,
        coords={d: slc2.coords[d] for d in slc2.dims if d in slc2.coords},
    )

    ifg_ml = multilook(ifg, looks_y=looks_y, looks_x=looks_x)
    pow1_ml = multilook(pow1, looks_y=looks_y, looks_x=looks_x)
    pow2_ml = multilook(pow2, looks_y=looks_y, looks_x=looks_x)

    denom = np.sqrt(np.asarray(pow1_ml) * np.asarray(pow2_ml))
    with np.errstate(divide="ignore", invalid="ignore"):
        coh_arr = np.where(denom > 0, np.abs(np.asarray(ifg_ml)) / denom, 0.0)

    return xr.DataArray(
        coh_arr.astype(np.float32),
        dims=ifg_ml.dims,
        coords=ifg_ml.coords,
        name="coherence",
        attrs={"units": "1", "long_name": "Multilooked interferometric coherence"},
    )


def unwrap(
    igram: xr.DataArray,
    corr: xr.DataArray,
    nlooks: float,
    mask: xr.DataArray | None = None,
    cost: str = "smooth",
    init: str = "mcf",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Unwrap an interferogram using SNAPHU.

    Parameters
    ----------
    igram : xr.DataArray
        Complex interferogram (2D). SNAPHU accepts complex input
        directly and extracts the wrapped phase internally.
    corr : xr.DataArray
        Coherence magnitude in [0, 1], same shape as ``igram``.
    nlooks : float
        Equivalent number of independent looks.
    mask : xr.DataArray, optional
        Binary mask of valid pixels (1 = valid, 0 = masked).
    cost : str
        SNAPHU cost mode: ``"smooth"`` or ``"defo"``. Default ``"smooth"``.
    init : str
        Initialization method: ``"mcf"`` or ``"mst"``. Default ``"mcf"``.

    Returns
    -------
    unwrapped_phase : xr.DataArray
        Unwrapped phase in radians.
    connected_components : xr.DataArray
        Connected component labels (0 = not connected).
    """
    import snaphu

    igram_vals = np.asarray(igram)
    corr_vals = np.asarray(corr)
    mask_arr = np.asarray(mask) if mask is not None else None

    unw_arr, conncomp_arr = snaphu.unwrap(
        igram_vals,
        corr_vals,
        nlooks,
        cost=cost,
        init=init,
        mask=mask_arr,
    )

    unw = xr.DataArray(
        unw_arr.astype(np.float32),
        dims=igram.dims,
        coords=igram.coords,
        name="unwrapped_phase",
        attrs={"units": "radians", "long_name": "Unwrapped phase"},
    )

    conncomp = xr.DataArray(
        conncomp_arr,
        dims=igram.dims,
        coords=igram.coords,
        name="connected_components",
        attrs={"long_name": "Connected component labels"},
    )

    return unw, conncomp
