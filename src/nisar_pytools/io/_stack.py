"""Stack multiple GSLC files into a time-series DataArray."""

from __future__ import annotations

import os
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import xarray as xr

from nisar_pytools.utils._validation import validate_nisar_hdf5

# HDF5 paths
_IDENT_PATH = "science/LSAR/identification"
_TIME_DATASET = "zeroDopplerStartTime"
_TRACK_DATASET = "trackNumber"
_FRAME_DATASET = "frameNumber"
_GRIDS_TEMPLATE = "science/LSAR/GSLC/grids/{frequency}"


def stack_gslcs(
    filepaths: list[str | os.PathLike],
    frequency: str = "frequencyA",
    polarization: str = "HH",
    chunks: dict[str, int] | str | None = "auto",
    check_track_frame: bool = True,
) -> xr.DataArray:
    """Stack multiple GSLC files into a 3D ``(time, y, x)`` DataArray.

    Opens each file lazily, extracts the requested frequency/polarization,
    validates that all files share the same spatial grid, and concatenates
    along a ``time`` dimension parsed from each file's metadata.

    Parameters
    ----------
    filepaths : list of str or path-like
        Paths to GSLC HDF5 files. Order does not matter — the output is
        sorted by acquisition time.
    frequency : str
        Frequency group to extract (default ``"frequencyA"``).
    polarization : str
        Polarization to extract (default ``"HH"``).
    chunks : dict, "auto", or None
        Chunk specification for dask arrays. ``"auto"`` uses HDF5 native chunks.
    check_track_frame : bool
        If ``True`` (default), verify all files share the same track and frame
        number. Set to ``False`` to skip this check.

    Returns
    -------
    xr.DataArray
        Complex 3D array with dimensions ``(time, y, x)`` and a ``time``
        coordinate of ``datetime64`` values. Data is dask-backed.

    Raises
    ------
    ValueError
        If files have mismatched grids, track/frame numbers, or the requested
        frequency/polarization is missing.
    FileNotFoundError
        If any file does not exist.
    """
    if not filepaths:
        raise ValueError("filepaths must be a non-empty list")

    grids_path = _GRIDS_TEMPLATE.format(frequency=frequency)

    # First pass: open all files, extract metadata and validate
    entries: list[dict] = []
    for fp in filepaths:
        fp = Path(fp)
        h5file = validate_nisar_hdf5(fp)
        try:
            _validate_gslc(h5file, fp, grids_path, polarization)
            entry = _extract_entry(h5file, fp, grids_path, polarization, chunks)
        except Exception:
            h5file.close()
            raise
        entries.append(entry)

    # Sort by time
    entries.sort(key=lambda e: e["time"])

    # Validate consistent grids and track/frame
    _validate_consistency(entries, check_track_frame)

    # Build the stacked DataArray
    ref = entries[0]
    times = pd.DatetimeIndex([e["time"] for e in entries])

    dask_arrays = [e["dask_array"] for e in entries]
    stacked = da.stack(dask_arrays, axis=0)

    result = xr.DataArray(
        stacked,
        dims=["time", "y", "x"],
        coords={
            "time": times,
            "y": ref["y"],
            "x": ref["x"],
        },
        name=polarization,
        attrs={
            "frequency": frequency,
            "polarization": polarization,
            "long_name": f"GSLC {frequency} {polarization} stack",
        },
    )

    # Keep file handles alive by attaching to the underlying array object
    result.attrs["_h5files"] = [e["h5file"] for e in entries]

    return result


def _validate_gslc(
    h5file: h5py.File, filepath: Path, grids_path: str, polarization: str
) -> None:
    """Check that the file is a GSLC with the requested frequency/pol."""
    ident = h5file[_IDENT_PATH]
    product_type = ident[_TIME_DATASET.replace(_TIME_DATASET, "productType")][()]
    if isinstance(product_type, bytes):
        product_type = product_type.decode()
    if product_type.strip() != "GSLC":
        raise ValueError(f"Expected GSLC product, got '{product_type}': {filepath}")

    if grids_path not in h5file:
        raise ValueError(f"Frequency group '{grids_path}' not found: {filepath}")

    if polarization not in h5file[grids_path]:
        available = [
            k for k in h5file[grids_path].keys()
            if isinstance(h5file[grids_path][k], h5py.Dataset) and h5file[grids_path][k].ndim == 2
        ]
        raise ValueError(
            f"Polarization '{polarization}' not found in {grids_path}. "
            f"Available: {available}. File: {filepath}"
        )


def _extract_entry(
    h5file: h5py.File,
    filepath: Path,
    grids_path: str,
    polarization: str,
    chunks: dict[str, int] | str | None,
) -> dict:
    """Extract metadata and a lazy dask array from a single GSLC file."""
    ident = h5file[_IDENT_PATH]

    # Parse time
    time_raw = ident[_TIME_DATASET][()]
    if isinstance(time_raw, bytes):
        time_raw = time_raw.decode()
    time = pd.Timestamp(time_raw)

    # Track/frame
    track = int(ident[_TRACK_DATASET][()])
    frame = int(ident[_FRAME_DATASET][()])

    # Coordinates
    grp = h5file[grids_path]
    x = grp["xCoordinates"][()]
    y = grp["yCoordinates"][()]

    # Lazy data
    ds = grp[polarization]
    chunk_spec = _resolve_chunks(ds, chunks)
    arr = da.from_array(ds, chunks=chunk_spec, lock=False)

    return {
        "time": time,
        "track": track,
        "frame": frame,
        "x": x,
        "y": y,
        "dask_array": arr,
        "h5file": h5file,
        "filepath": filepath,
    }


def _resolve_chunks(
    ds: h5py.Dataset, chunks: dict[str, int] | str | None
) -> tuple[int, ...] | None:
    """Resolve chunk spec for a single h5py dataset."""
    if chunks is None:
        return ds.shape
    if isinstance(chunks, str) and chunks == "auto":
        return ds.chunks if ds.chunks else ds.shape
    if isinstance(chunks, dict):
        dims = ("y", "x")
        return tuple(chunks.get(d, s) for d, s in zip(dims, ds.shape))
    return ds.shape


def _validate_consistency(entries: list[dict], check_track_frame: bool) -> None:
    """Validate that all entries share the same grid and optionally track/frame."""
    ref = entries[0]

    for entry in entries[1:]:
        if not np.array_equal(ref["x"], entry["x"]):
            raise ValueError(
                f"x coordinates do not match between "
                f"{ref['filepath'].name} and {entry['filepath'].name}"
            )
        if not np.array_equal(ref["y"], entry["y"]):
            raise ValueError(
                f"y coordinates do not match between "
                f"{ref['filepath'].name} and {entry['filepath'].name}"
            )

        if check_track_frame:
            if ref["track"] != entry["track"]:
                raise ValueError(
                    f"Track numbers do not match: {ref['track']} "
                    f"({ref['filepath'].name}) vs {entry['track']} "
                    f"({entry['filepath'].name})"
                )
            if ref["frame"] != entry["frame"]:
                raise ValueError(
                    f"Frame numbers do not match: {ref['frame']} "
                    f"({ref['filepath'].name}) vs {entry['frame']} "
                    f"({entry['filepath'].name})"
                )
