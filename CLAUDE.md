# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python tools for reading NISAR (NASA-ISRO SAR) HDF5 data products into lazy xarray DataTree objects. Currently supports GSLC and GUNW product types.

## Commands

```bash
# Use the nisar_pytools conda env for all commands
# Install: mamba env create -f environment.yml
# Activate or prefix with: /Users/zmhoppinen/miniforge3/envs/nisar_pytools/bin/python

# Run all unit tests
/Users/zmhoppinen/miniforge3/envs/nisar_pytools/bin/python -m pytest tests/ -v -m "not integration"

# Run a single test file or test
/Users/zmhoppinen/miniforge3/envs/nisar_pytools/bin/python -m pytest tests/test_reader.py -v
/Users/zmhoppinen/miniforge3/envs/nisar_pytools/bin/python -m pytest tests/test_reader.py::TestOpenNisar::test_lazy_by_default -v

# Run integration tests (requires real NISAR files in local/)
NISAR_TEST_GSLC=local/gslcs/<file>.h5 NISAR_TEST_GUNW=local/gunws/<file>.h5 \
  /Users/zmhoppinen/miniforge3/envs/nisar_pytools/bin/python -m pytest tests/ -v

# Lint
/Users/zmhoppinen/miniforge3/envs/nisar_pytools/bin/ruff check src/ tests/

# Install dependencies with mamba, not pip or conda
mamba install -n nisar_pytools -c conda-forge <package>
```

## Architecture

**Entry point**: `from nisar_pytools import open_nisar` — takes an HDF5 file path, validates it, detects product type, returns `xr.DataTree` with dask-backed lazy arrays.

**Flow**: `open_nisar()` → `validate_nisar_hdf5()` → `detect_product_type()` → `h5_to_datatree()`

**Key design decisions**:
- Uses custom h5py walker (not `xr.open_datatree` with h5netcdf) because NISAR files have named types at root and 100+ scalar datasets that belong in attrs, not as DataArrays.
- All 2D+ arrays are dask-backed via `dask.array.from_array()`. Coordinates (1D `xCoordinates`/`yCoordinates`) are loaded eagerly since they're small.
- Scalar datasets become `Dataset.attrs`. 1D non-coordinate arrays (e.g., `listOfPolarizations`) also go to attrs.
- Unnamed dimensions are prefixed with the variable name (e.g., `eulerAngles_dim_1`) to avoid conflicts when sibling datasets have different shapes.
- The `h5py.File` handle is kept alive on `tree.__dict__["_h5file"]` to prevent GC while dask arrays reference it.

**`src/nisar_pytools/` layout**:
- `io/_reader.py` — `open_nisar()` entry point
- `io/_h5_to_datatree.py` — generic HDF5 group walker that builds DataTree
- `utils/_validation.py` — file validation and product type detection

**Tests** use synthetic HDF5 fixtures (tiny 8x10 arrays) defined in `tests/conftest.py`. Integration tests against real files are gated behind `@pytest.mark.integration` and env vars.
