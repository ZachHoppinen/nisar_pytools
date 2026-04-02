"""Validation utilities for NISAR HDF5 files."""

from __future__ import annotations

from pathlib import Path

import h5py

REQUIRED_GROUP = "science/LSAR"
PRODUCT_TYPE_PATH = "science/LSAR/identification/productType"
SUPPORTED_PRODUCTS = {"GSLC", "GUNW"}


def validate_nisar_hdf5(filepath: str | Path) -> h5py.File:
    """Validate that a file is a readable NISAR HDF5 file and return an open handle.

    Checks:
    1. Path exists and is a file
    2. File is valid HDF5
    3. Contains ``science/LSAR/`` group
    4. Contains ``science/LSAR/identification/productType``

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.

    Returns
    -------
    h5py.File
        Open file handle (caller is responsible for lifetime management).

    Raises
    ------
    FileNotFoundError
        Path does not exist or is not a file.
    ValueError
        Not a valid HDF5 file or missing required NISAR structure.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    if not h5py.is_hdf5(filepath):
        raise ValueError(f"Not a valid HDF5 file: {filepath}")

    h5file = h5py.File(filepath, "r")
    try:
        if REQUIRED_GROUP not in h5file:
            raise ValueError(
                f"Missing required group '{REQUIRED_GROUP}' — not a NISAR file: {filepath}"
            )
        if PRODUCT_TYPE_PATH not in h5file:
            raise ValueError(
                f"Missing '{PRODUCT_TYPE_PATH}' dataset — cannot determine product type: "
                f"{filepath}"
            )
    except Exception:
        h5file.close()
        raise

    return h5file


def detect_product_type(h5file: h5py.File) -> str:
    """Read the product type from a validated NISAR HDF5 file.

    Parameters
    ----------
    h5file : h5py.File
        Open, validated NISAR HDF5 file.

    Returns
    -------
    str
        Product type string, e.g. ``"GSLC"`` or ``"GUNW"``.

    Raises
    ------
    ValueError
        Product type dataset missing or value not recognized.
    """
    if PRODUCT_TYPE_PATH not in h5file:
        raise ValueError(f"Missing '{PRODUCT_TYPE_PATH}' dataset")

    raw = h5file[PRODUCT_TYPE_PATH][()]
    product_type = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    product_type = product_type.strip()

    if product_type not in SUPPORTED_PRODUCTS:
        raise ValueError(
            f"Unsupported product type '{product_type}'. "
            f"Supported: {sorted(SUPPORTED_PRODUCTS)}"
        )

    return product_type
