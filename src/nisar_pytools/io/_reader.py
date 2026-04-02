"""Main entry point for reading NISAR HDF5 files."""

from __future__ import annotations

import os

import xarray as xr

from nisar_pytools.io._h5_to_datatree import h5_to_datatree
from nisar_pytools.utils._validation import detect_product_type, validate_nisar_hdf5


def open_nisar(
    filepath: str | os.PathLike,
    chunks: dict[str, int] | str | None = "auto",
) -> xr.DataTree:
    """Open a NISAR HDF5 product file as a lazy xarray DataTree.

    Parameters
    ----------
    filepath : str or path-like
        Path to the NISAR HDF5 file (``.h5``).
    chunks : dict, "auto", or None
        Chunk specification for dask arrays.
        - ``"auto"`` (default): Use the HDF5 file's native chunk sizes.
        - dict: e.g. ``{"y": 512, "x": 512}``
        - ``None``: Load eagerly (not recommended for large files).

    Returns
    -------
    xr.DataTree
        Tree mirroring the HDF5 group structure. Data variables are
        dask-backed. Coordinates (``xCoordinates``, ``yCoordinates``) are
        assigned as proper dimension coordinates named ``x`` and ``y``.

    Notes
    -----
    The returned DataTree holds a reference to the open HDF5 file.
    The lazy arrays will become invalid if the file is closed or
    garbage-collected. Keep a reference to the DataTree alive as long
    as you need the data.

    Examples
    --------
    >>> from nisar_pytools import open_nisar
    >>> dt = open_nisar("NISAR_L2_PR_GSLC_...h5")
    >>> dt["science/LSAR/GSLC/grids/frequencyA"].dataset
    """
    h5file = validate_nisar_hdf5(filepath)

    try:
        product_type = detect_product_type(h5file)
        dt = h5_to_datatree(h5file, chunks=chunks)
    except Exception:
        h5file.close()
        raise

    # Store product type and keep the file handle alive
    dt.attrs["product_type"] = product_type
    dt.__dict__["_h5file"] = h5file

    return dt
