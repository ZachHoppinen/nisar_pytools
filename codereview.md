# Code Review Notes

Running notes on validation, comparisons, and findings during development.

## Interferogram Comparison (GSLC vs GUNW)

- Our `interferogram()` + `multilook(looks_y=4, looks_x=4)` compared to GUNW wrapped interferogram
- **Phase correlation: 0.94**, mean bias: 0.01 rad (0.65 deg)
- GUNW uses 5x6 range/azimuth looks in radar geometry + SINC geocoding to 20m
- We do 4x4 isotropic looks in map geometry
- Residual 6% decorrelation is from different multilooking geometry, not a code error
- No common band filtering or phase filtering applied by GUNW processor (`no_filter`)

## Coherence Comparison (GSLC vs GUNW)

| Method | Window | Our Mean | GUNW Mean | Correlation | RMSE | Bias |
|--------|--------|----------|-----------|-------------|------|------|
| Boxcar | 5 | 0.691 | 0.700 | 0.866 | 0.082 | -0.008 |
| Boxcar | 7 | 0.692 | 0.700 | 0.926 | 0.058 | -0.008 |
| **Boxcar** | **11** | **0.696** | **0.700** | **0.934** | **0.050** | **-0.003** |
| Boxcar | 15 | 0.701 | 0.700 | 0.902 | 0.059 | 0.002 |
| Gaussian | 2 | 0.692 | 0.700 | 0.937 | 0.053 | -0.008 |
| **Gaussian** | **3** | **0.696** | **0.700** | **0.954** | **0.042** | **-0.004** |
| Gaussian | 5 | 0.704 | 0.700 | 0.908 | 0.057 | 0.005 |

- **Best match: Gaussian sigma=3** (0.954 correlation, 0.042 RMSE)
- GUNW computes coherence in radar geometry (5x6 looks) then geocodes with BILINEAR
- Gaussian approximates the geocoded effect of asymmetric radar-geometry averaging

## Baseline Comparison (GSLC vs GUNW)

| Component | Our Result | GUNW | Difference |
|-----------|-----------|------|------------|
| B_perp mean | -98.5 m | -98.9 m | 0.4 m |
| B_par mean | -40.0 m | -40.2 m | 0.2 m |

- Sub-meter agreement using ECEF geometry with orbit interpolation
- Sign convention: `compute_baseline(secondary, reference)` matches GUNW convention
- Small residual from using height=0 for ground points vs GUNW's multi-height computation

## GUNW Processing Parameters

Key parameters from the GUNW `runConfigurationContents`:

### Crossmul (interferogram + coherence)
- `range_looks: 5`, `azimuth_looks: 6` (30 looks total)
- `oversample: 2` (2x oversampling before cross-multiplication)
- `flatten: True` (topographic flattening)
- `common_band_range_filter: False`
- `common_band_azimuth_filter: False`

### Geocoding
- Complex data: SINC interpolation
- Floating point (coherence): BILINEAR interpolation
- Integer data (masks): nearest neighbor
- DEM interpolation: biquintic
- Wrapped IFG posting: 20m x 20m
- Unwrapped posting: 80m x 80m
- Radar grid cubes: 500m x 500m, 20 height layers (-500m to 9000m)

### Corrections applied during geocoding
- Azimuth ionospheric: True
- Range ionospheric: True
- Hydrostatic tropospheric: True
- Wet tropospheric: True

### Phase unwrapping
- Algorithm: SNAPHU
- Cost mode: smooth
- Initialization: MCF
- Unwrapped looks: 13 range x 16 azimuth (208 total)

## Code Review Fixes Applied

### download.py
- Atomic writes via temp file + `os.replace()` (no partial files on failure)
- Thread-local `requests.Session` (thread-safe connection pooling)
- File handle cleanup on all failure paths
- O(1) corrupted file lookup (set-based instead of O(n²) list removal)
- SSAR band support in validation
- Duplicate filename warning

### export.py
- Independent attrs dicts for real/imag components (no shared-dict mutation)
- `source.copy(data=...)` preserves all metadata (CRS, name, coords)
- `read_netcdf` uses context manager + `.load()` (no leaked file handles)
- Variable ordering preserved via OrderedDict
- Orphan `_real`/`_imag` variables logged as warnings

### h5_to_datatree.py
- `threading.Lock()` for thread-safe HDF5 reads via dask
- Square grid dim collision handled (tracks used dims)
- Unchunked datasets get 512-pixel default chunks (not one giant chunk)
- EPSG fallback handles string/bytes values
- `_dims_from_dimension_list` no longer abandons all resolved dims on partial failure

### reader.py
- `weakref.finalize` + module-level `_open_files` set for file handle lifetime
- Warns for `chunks=None` on files > 100 MB
- Documented that derived DataTrees don't carry file handles

### search.py
- `urllib.parse.urlparse` for robust QA filename filtering (handles query strings)
- Top-level imports for first-party validation modules
- Direction validated before search kwargs assignment
- Docstring corrected: returns empty list, doesn't raise on no results

### stack.py
- `weakref.finalize` + module-level set for N file handles
- All file handles closed on any failure (validation or consistency)
- Shared `threading.Lock()` across all files
- `_extract_epsg` reused from h5_to_datatree (no duplication)
- Deduplicates input filepaths, warns on duplicates
- Timestamp parse errors include filename context

### atmospheric.py
- All three interpolator axes (height, y, x) sorted to ascending
- Returns `xr.DataArray` with DEM coords (not bare numpy)
- `source.copy(data=...)` preserves CRS/coords
- DEM shape validated against unwrapped phase
- Warns if >5% pixels NaN from extrapolation
- Ionosphere screen auto-interpolated if coords don't match
- Hydrostatic + wet interpolated separately (halves peak memory)

### baseline.py
- 1D `az_time` broadcast to 2D
- Gram-Schmidt orthogonalization for proper cross-track direction
- `make_interp_spline` with 3 spline fits (not 9 `interp1d` calls)
- Warns on orbit time extrapolation
- Orbit time attrs validated as array (not scalar)
- Optional `dem` parameter for terrain-aware ground points
- EPSG extraction raises on failure (no wrong default)

### filtering.py
- Filter operates on complex interferogram directly (not phase-only)
- Periodic Hanning window for correct overlap-add
- Input validation: 2D, complex, overlap < patch_size//2
- `scipy.ndimage` imported at top level
- Documented as in-memory only

### phase_linking.py
- `eigh` instead of `eig` (Hermitian, numerically stable, real eigenvalues)
- `argmin(abs(eigenvalues))` preserved (correct for EMI)
- All xarray `.sel` replaced with numpy index slicing (orders of magnitude faster)
- `np.asarray` at entry for in-memory guarantee
- Warns when pixel count > 10k (loop will be slow)

## Phase Linking: Vectorization Needed

The `phase_link()` function currently uses a **per-pixel Python loop** that is
fundamentally unsuitable for full-resolution scenes:

- 100×100 pixels: ~seconds
- 500×500 pixels: ~minutes
- 5000×5000 pixels: ~hours (estimated)

**Root cause**: For each pixel, the loop does:
1. Extract a spatial window (`O(1)` with numpy slicing — fast)
2. Compute GLRT test (`O(window²)` — fast)
3. `scipy.ndimage.label` for connected components (`O(window²)` — moderate)
4. Extract SHP pixels and compute coherence matrix (`O(n_images² × n_shp)` — fast)
5. `np.linalg.eigh` for EMI (`O(n_images³)` — fast)

The bottleneck is the **Python loop overhead** (function call dispatch, xarray
coordinate lookups) multiplied by ny×nx iterations, not the per-pixel math.

**Path to vectorization**:
1. Pre-compute the SHP mask for the entire scene using vectorized GLRT
2. Use `numpy.lib.stride_tricks.sliding_window_view` to extract all windows at once
3. Batch coherence matrix estimation across all pixels using einsum
4. Batch eigendecomposition (or use iterative methods like power iteration)
5. The connected-component step is the hardest to vectorize — may need
   approximation (e.g., fixed window without connectivity filtering)

**Alternative**: Use MiaplPy or DOLPHIN (both implement vectorized phase linking
for InSAR time series) and call from nisar_pytools as an optional dependency.

## Dolphin Comparison (Phase Linking)

Compared our phase linking against ISCE-framework/dolphin (the reference
implementation for NISAR InSAR time series).

**Coherence matrix**: Mathematically equivalent. Both compute
`C[i,j] = (Σ s_i * s_j*) / sqrt(Σ|s_i|² · Σ|s_j|²)`. Dolphin uses
power-sum form, we use L2-norm — same result.

**EMI eigenvalue selection**: Dolphin uses shift-inverse iteration with
`mu=0.99` (targets eigenvalue closest to 1). We use `argmin(abs(eigenvalues))`
which selects eigenvalue closest to 0. In practice both converge to the
same eigenvector for well-conditioned matrices (our test case: eigenvalue =
1.0009). For ill-conditioned cases, dolphin's approach is more robust.

**GLRT threshold**: Mathematically equivalent but expressed differently.
- Ours: `T < chi2.ppf(conf, df=1) / (2*N)`
- Dolphin: `N * T < chi2.ppf(1-alpha, df=1)`
Same comparison rearranged. Both match Parizzi et al. 2011.

**Regularization**: Improved from 1e-10 jitter to Cholesky factorization
with 1e-6 jitter (matching dolphin's approach). Falls back to direct
inversion if Cholesky fails.

**Performance**: Dolphin uses JAX vmap for vectorized processing. Our
per-pixel loop is correct but orders of magnitude slower. Production
use should call dolphin as an optional dependency.

## RSLC Notes

- RSLC available at `local/rslc/` (same acquisition as GSLC: track 77, frame 24, 2025-11-03)
- SLC shape: (54720, 55701) complex64 — radar geometry (azimuth x range)
- Has geolocationGrid mapping radar coords to map coords at 20 height layers
- Geocoding deferred — requires ISCE3 dependency
- Range-azimuth processing advantageous for: ionospheric streaks, RFI, subswath boundaries, azimuth ambiguities
