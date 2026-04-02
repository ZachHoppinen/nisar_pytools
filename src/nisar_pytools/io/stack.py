"""Stack multiple GSLC files into a time-series DataArray."""

from __future__ import annotations

import logging
import os
import threading
import warnings
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

from nisar_pytools.io.h5_to_datatree import _extract_epsg
from nisar_pytools.utils.validation import validate_nisar_hdf5

log = logging.getLogger(__name__)

# HDF5 paths
_IDENT_PATH = "science/LSAR/identification"
_TIME_DATASET = "zeroDopplerStartTime"
_TRACK_DATASET = "trackNumber"
_FRAME_DATASET = "frameNumber"
_GRIDS_TEMPLATE = "science/LSAR/GSLC/grids/{frequency}"
_KNOWN_POLARIZATIONS = {"HH", "HV", "VV", "VH"}

# Default chunk size for unchunked datasets
_DEFAULT_CHUNK_SIZE = 512

# Module-level set to keep file handles alive.
# Protected by a lock for thread safety (finalizers run on arbitrary threads).
_open_files: set = set()
_open_files_lock = threading.Lock()


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

    File handles are kept alive via a module-level set. They are **not**
    automatically closed — call :func:`close_stack_files` when done, or
    use the module-level ``_open_files`` set directly.

    Parameters
    ----------
    filepaths : list of str or path-like
        Paths to GSLC HDF5 files. Order does not matter — the output is
        sorted by acquisition time. Duplicate paths are removed.
    frequency : str
        Frequency group to extract (default ``"frequencyA"``).
    polarization : str
        Polarization to extract (default ``"HH"``).
    chunks : dict, "auto", or None
        Chunk specification for dask arrays.
        - ``"auto"``: Use HDF5 native chunks (default).
        - dict: e.g. ``{"y": 512, "x": 512}``
        - ``None``: Load eagerly. Warns for files > 100 MB.
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
        If files have mismatched grids, track/frame numbers, duplicate
        timestamps, or the requested frequency/polarization is missing.
    FileNotFoundError
        If any file does not exist.
    """
    if not filepaths:
        raise ValueError("filepaths must be a non-empty list")

    # Deduplicate paths
    unique_paths = list(dict.fromkeys(str(Path(fp).resolve()) for fp in filepaths))
    if len(unique_paths) < len(filepaths):
        log.warning(
            "Removed %d duplicate filepath(s)", len(filepaths) - len(unique_paths)
        )

    grids_path = _GRIDS_TEMPLATE.format(frequency=frequency)

    # Shared lock for thread-safe HDF5 reads across all files
    lock = threading.Lock()

    # Open all files, closing everything on any failure
    opened_files: list[h5py.File] = []
    entries: list[dict] = []
    try:
        for fp_str in unique_paths:
            fp = Path(fp_str)

            if chunks is None:
                file_size_mb = fp.stat().st_size / (1024 * 1024)
                if file_size_mb > 100:
                    warnings.warn(
                        f"Loading {file_size_mb:.0f} MB file eagerly: {fp.name}",
                        UserWarning,
                        stacklevel=2,
                    )

            h5file = validate_nisar_hdf5(fp)
            opened_files.append(h5file)

            _validate_gslc(h5file, fp, grids_path, polarization)
            entry = _extract_entry(h5file, fp, grids_path, polarization, chunks, lock)
            entries.append(entry)

        # Sort by time
        entries.sort(key=lambda e: e["time"])

        # Check for duplicate timestamps
        times = [e["time"] for e in entries]
        if len(set(times)) < len(times):
            raise ValueError(
                "Duplicate timestamps detected in GSLC stack. "
                "Each file must have a unique acquisition time."
            )

        # Validate consistent grids and track/frame
        _validate_consistency(entries, check_track_frame)

    except Exception:
        _close_files(opened_files)
        raise

    # Build the stacked DataArray
    ref = entries[0]
    time_index = pd.DatetimeIndex(times)

    dask_arrays = [e["dask_array"] for e in entries]
    stacked = da.stack(dask_arrays, axis=0)

    result = xr.DataArray(
        stacked,
        dims=["time", "y", "x"],
        coords={
            "time": time_index,
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

    # Assign CRS — wrap in try/except to avoid leaking handles on CRS failure
    try:
        epsg = ref.get("epsg")
        if epsg is not None:
            result = result.rio.write_crs(epsg)
            result = result.rio.set_spatial_dims(x_dim="x", y_dim="y")
    except Exception:
        _close_files(opened_files)
        raise

    # Keep file handles alive via module-level set (thread-safe)
    with _open_files_lock:
        for f in opened_files:
            _open_files.add(f)

    return result


def close_stack_files() -> None:
    """Close all file handles opened by :func:`stack_gslcs`.

    Call this when you are done computing from all stacked DataArrays.
    """
    with _open_files_lock:
        for f in list(_open_files):
            _open_files.discard(f)
            try:
                if f.id.valid:
                    f.close()
            except Exception:
                pass


def _close_files(files: list[h5py.File]) -> None:
    """Close a list of file handles (used for error cleanup)."""
    for f in files:
        try:
            if f.id.valid:
                f.close()
        except Exception:
            pass


def _validate_gslc(
    h5file: h5py.File, filepath: Path, grids_path: str, polarization: str
) -> None:
    """Check that the file is a GSLC with the requested frequency/pol."""
    ident = h5file[_IDENT_PATH]
    product_type = ident["productType"][()]
    if isinstance(product_type, bytes):
        product_type = product_type.decode()
    if product_type.strip() != "GSLC":
        raise ValueError(f"Expected GSLC product, got '{product_type}': {filepath}")

    if grids_path not in h5file:
        raise ValueError(f"Frequency group '{grids_path}' not found: {filepath}")

    if polarization not in h5file[grids_path]:
        grp = h5file[grids_path]
        available = [k for k in grp.keys() if k in _KNOWN_POLARIZATIONS]
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
    lock: threading.Lock,
) -> dict:
    """Extract metadata and a lazy dask array from a single GSLC file."""
    ident = h5file[_IDENT_PATH]

    # Parse time with error context
    time_raw = ident[_TIME_DATASET][()]
    if isinstance(time_raw, bytes):
        time_raw = time_raw.decode()
    try:
        time = pd.Timestamp(time_raw)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Could not parse timestamp '{time_raw}' from {filepath.name}: {e}"
        ) from e

    # Track/frame
    track = int(ident[_TRACK_DATASET][()])
    frame = int(ident[_FRAME_DATASET][()])

    # Coordinates
    grp = h5file[grids_path]
    x = grp["xCoordinates"][()]
    y = grp["yCoordinates"][()]

    # EPSG — use shared utility from h5_to_datatree
    epsg = _extract_epsg(grp)

    # Data loading
    ds = grp[polarization]
    chunk_spec = _resolve_chunks(ds, chunks)
    if chunk_spec is not None:
        arr = da.from_array(ds, chunks=chunk_spec, lock=lock)
    else:
        # Eager load — keep as numpy, wrap in dask for da.stack compatibility
        arr = da.from_array(np.asarray(ds[()]), chunks=-1)

    return {
        "time": time,
        "track": track,
        "frame": frame,
        "x": x,
        "y": y,
        "epsg": epsg,
        "dask_array": arr,
        "h5file": h5file,
        "filepath": filepath,
    }


def _resolve_chunks(
    ds: h5py.Dataset, chunks: dict[str, int] | str | None
) -> tuple[int, ...] | None:
    """Resolve chunk spec for a single h5py dataset.

    Returns None when chunks=None (eager loading).
    """
    if chunks is None:
        return None
    if isinstance(chunks, str) and chunks == "auto":
        if ds.chunks is not None:
            return ds.chunks
        return tuple(min(s, _DEFAULT_CHUNK_SIZE) for s in ds.shape)
    if isinstance(chunks, dict):
        dims = ("y", "x")
        return tuple(chunks.get(d, s) for d, s in zip(dims, ds.shape))
    return tuple(min(s, _DEFAULT_CHUNK_SIZE) for s in ds.shape)


def _validate_consistency(entries: list[dict], check_track_frame: bool) -> None:
    """Validate that all entries share the same grid and optionally track/frame."""
    ref = entries[0]

    for entry in entries[1:]:
        if not np.allclose(ref["x"], entry["x"], rtol=1e-9, atol=0):
            raise ValueError(
                f"x coordinates do not match between "
                f"{ref['filepath'].name} and {entry['filepath'].name}"
            )
        if not np.allclose(ref["y"], entry["y"], rtol=1e-9, atol=0):
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
