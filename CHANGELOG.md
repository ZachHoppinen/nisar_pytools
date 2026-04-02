# Changelog

## v0.1.0

### Added
- `open_nisar()` entry point for reading NISAR HDF5 files into lazy xarray DataTree
- Support for GSLC and GUNW product types
- Generic h5py-to-DataTree walker with dask-backed lazy arrays
- Automatic coordinate assignment (`xCoordinates`/`yCoordinates` to `x`/`y` dims)
- HDF5 dimension scale resolution via `DIMENSION_LIST` attributes
- File validation and product type detection (`validate_nisar_hdf5`, `detect_product_type`)
- Scalar and string dataset decoding to DataTree attrs
- Test suite with synthetic HDF5 fixtures and integration test markers
- CI workflow via GitHub Actions
- Conda environment file (`environment.yml`)
