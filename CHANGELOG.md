# Changelog

## v0.1.0

### IO
- `open_nisar()` — read NISAR HDF5 files into lazy xarray DataTree with CRS
- `stack_gslcs()` — stack multiple same-track GSLCs into a `(time, y, x)` DataArray
- `find_nisar()` — search ASF for NISAR product URLs by AOI, dates, track, direction
- `download_urls()` — parallel download with retry and post-download HDF5 validation
- `to_zarr()` / `to_netcdf()` / `read_netcdf()` — export and import with complex data support
- Generic h5py-to-DataTree walker with dask-backed lazy arrays
- Automatic coordinate assignment and CRS from `projection` dataset
- HDF5 dimension scale resolution via `DIMENSION_LIST` attributes

### Processing
- `interferogram()` — complex interferogram with grid matching validation
- `coherence()` — sliding-window coherence estimation
- `multilook()` / `multilook_interferogram()` — spatial averaging and downsampling
- `unwrap()` — phase unwrapping via SNAPHU
- `calculate_phase()` — extract phase from complex data
- `phase_link()` — EMI phase linking with SHP selection on SLC stacks
- `h_a_alpha()` — Cloude-Pottier polarimetric decomposition (entropy, anisotropy, alpha)
- Individual `entropy()`, `anisotropy()`, `alpha()`, `mean_alpha()` functions

### Utilities
- `fetch_dem()` — auto-download Copernicus GLO-30 DEM matching a NISAR file's extent
- `local_incidence_angle()` — compute LIA from LOS vectors and a DEM
- `get_acquisition_time()`, `get_orbit_info()`, `get_bounding_polygon()` — metadata extraction
- `apply_mask()` / `get_mask()` — apply NISAR mask datasets
- `to_db()` / `from_db()` — dB conversion
- Date, AOI, URL, and path validation utilities

### Visualization
- `plot_amplitude()`, `plot_phase()`, `plot_interferogram()`, `plot_coherence()`

### Infrastructure
- PyPI-ready packaging with `pyproject.toml`
- CI/CD via GitHub Actions (lint, test, build, publish)
- Pre-commit hooks (nbstripout, ruff)
- Conda environment file
- 243 tests with synthetic HDF5 fixtures
