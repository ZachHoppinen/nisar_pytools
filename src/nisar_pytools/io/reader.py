"""Main entry point for reading NISAR HDF5 files."""

from __future__ import annotations

import os
import warnings
import weakref

import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

from nisar_pytools.io.h5_to_datatree import h5_to_datatree
from nisar_pytools.utils.validation import detect_product_type, validate_nisar_hdf5

# Module-level set to prevent file handles from being garbage-collected
# while any DataTree references them. Cleaned up by the weak-ref finalizer.
_open_files: set = set()


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
        - ``None``: Load eagerly. Warns for files > 100 MB.

    Returns
    -------
    xr.DataTree
        Tree mirroring the HDF5 group structure. Data variables are
        dask-backed. Coordinates (``xCoordinates``, ``yCoordinates``) are
        assigned as proper dimension coordinates named ``x`` and ``y``.
        CRS is set via rioxarray on nodes that have a ``projection`` attribute.

    Notes
    -----
    The HDF5 file is kept open as long as the returned DataTree exists.
    When the DataTree is garbage-collected, the file is closed automatically
    via a weak-reference finalizer.

    **Important**: xarray operations that return a *new* DataTree (e.g.
    ``dt.sel()``, ``dt.map_over_datasets()``) do not carry the file
    handle — compute any lazy arrays before discarding the original tree.

    The ``product_type`` attribute (e.g. ``"GSLC"``) is set on the root
    node but may be dropped by operations that produce derived trees.

    Examples
    --------
    >>> from nisar_pytools import open_nisar
    >>> dt = open_nisar("NISAR_L2_PR_GSLC_...h5")
    >>> dt["science/LSAR/GSLC/grids/frequencyA"].dataset
    """
    filepath = os.fspath(filepath)

    # Warn if loading eagerly on a large file
    if chunks is None:
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > 100:
            warnings.warn(
                f"Loading {file_size_mb:.0f} MB file eagerly (chunks=None). "
                f"This will use significant memory. Consider chunks='auto'.",
                UserWarning,
                stacklevel=2,
            )

    # Open and validate (named clearly to show ownership transfer)
    h5file = validate_nisar_hdf5(filepath)

    try:
        product_type = detect_product_type(h5file)
        dt = h5_to_datatree(h5file, chunks=chunks)
    except Exception:
        h5file.close()
        raise

    dt.attrs["product_type"] = product_type

    # Keep the file handle alive via a module-level set.
    # When the DataTree is garbage-collected, the finalizer removes
    # the handle from the set, allowing it to be closed.
    _open_files.add(h5file)
    weakref.finalize(dt, _release_h5file, h5file)

    return dt


def _release_h5file(h5file) -> None:
    """Weak-reference finalizer: close the HDF5 file and remove from the set."""
    _open_files.discard(h5file)
    try:
        if h5file.id.valid:
            h5file.close()
    except Exception:
        pass
