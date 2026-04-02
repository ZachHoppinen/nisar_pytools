"""Generic HDF5 to xarray DataTree converter with lazy (dask-backed) arrays."""

from __future__ import annotations

import logging
import threading
from typing import Any

import dask.array as da
import h5py
import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr

log = logging.getLogger(__name__)

# 1D datasets with these names are treated as dimension coordinates
COORD_NAMES = {"xCoordinates", "yCoordinates", "heightAboveEllipsoid"}
# Mapping from coordinate dataset names to short dimension names
COORD_TO_DIM = {
    "xCoordinates": "x",
    "yCoordinates": "y",
    "heightAboveEllipsoid": "z",
}

# Default chunk size for unchunked datasets (per spatial dim)
_DEFAULT_CHUNK_SIZE = 512


def h5_to_datatree(
    h5file: h5py.File,
    root: str = "/",
    chunks: dict[str, int] | str | None = "auto",
) -> xr.DataTree:
    """Convert an HDF5 file to an xarray DataTree with lazy dask arrays.

    .. warning::
        The ``h5file`` handle **must remain open** for the lifetime of the
        returned DataTree. Closing the file (or letting it be garbage-collected)
        will invalidate all lazy dask arrays. The recommended pattern is::

            h5file = h5py.File(path, "r")
            dt = h5_to_datatree(h5file)
            # ... use dt ...
            # h5file stays open

        Do **not** use ``with h5py.File(...) as f: dt = h5_to_datatree(f)``
        — the arrays will be broken after the ``with`` block exits.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file handle.
    root : str
        Root group path to start from. Default ``"/"``.
    chunks : dict, "auto", or None
        Chunk specification for dask arrays.
        - ``"auto"``: Use the HDF5 file's native chunk sizes. For unchunked
          datasets, falls back to 512 per spatial dimension.
        - dict: e.g. ``{"y": 512, "x": 512}``
        - ``None``: Load eagerly (not recommended for large files).

    Returns
    -------
    xr.DataTree
    """
    # Shared lock for thread-safe HDF5 reads
    lock = threading.Lock()

    group = h5file[root] if root != "/" else h5file
    datasets = _walk_group(group, "/", chunks, lock)
    return xr.DataTree.from_dict(datasets)


def _walk_group(
    group: h5py.Group,
    path: str,
    chunks: dict[str, int] | str | None,
    lock: threading.Lock,
) -> dict[str, xr.Dataset]:
    """Recursively walk HDF5 groups, building a flat dict of path → Dataset."""
    datasets: dict[str, xr.Dataset] = {}

    # Build a dataset for this group if it contains any datasets
    ds = _build_dataset(group, chunks, lock)
    if ds is not None:
        datasets[path] = ds
    elif group.attrs:
        # Group has HDF5 attributes but no child datasets — store attrs only
        datasets[path] = xr.Dataset(attrs=_extract_attrs(group))

    # Recurse into child groups
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            child_path = f"/{key}" if path == "/" else f"{path}/{key}"
            datasets.update(_walk_group(item, child_path, chunks, lock))

    return datasets


def _build_dataset(
    group: h5py.Group,
    chunks: dict[str, int] | str | None,
    lock: threading.Lock,
) -> xr.Dataset | None:
    """Build an xr.Dataset from the datasets within a single HDF5 group.

    Returns None if the group contains no datasets (only subgroups).
    """
    coords_data: dict[str, np.ndarray] = {}
    coord_dim_map: dict[str, str] = {}
    data_vars: dict[str, xr.Variable] = {}
    attrs: dict[str, Any] = {}

    # Single pass over group keys
    for key in group.keys():
        item = group[key]
        if not isinstance(item, h5py.Dataset):
            continue

        if item.shape == ():
            # Scalar dataset → attribute
            attrs[key] = _decode_scalar(item[()])
        elif item.ndim == 1 and key in COORD_NAMES:
            # Coordinate dataset → load eagerly (small)
            dim_name = COORD_TO_DIM[key]
            coords_data[dim_name] = item[()]
            coord_dim_map[key] = dim_name
        elif item.ndim == 1 and key not in COORD_NAMES:
            # 1D non-coordinate dataset → attribute
            attrs[key] = _decode_array_attr(item[()])
        else:
            # 2D+ dataset → defer to second pass (needs coords_data populated)
            pass

    # Second pass: build data variables (2D+ arrays, now that coords are known)
    for key in group.keys():
        item = group[key]
        if not isinstance(item, h5py.Dataset):
            continue
        if item.ndim < 2:
            continue

        dims = _resolve_dims(item, coords_data, key)
        dask_chunks = _get_chunks(item, dims, chunks)
        if dask_chunks is not None:
            arr = da.from_array(item, chunks=dask_chunks, lock=lock)
        else:
            arr = item[()]

        var_attrs = _extract_attrs(item)
        data_vars[key] = xr.Variable(dims, arr, attrs=var_attrs)

    if not data_vars and not coords_data:
        if attrs:
            group_attrs = _extract_attrs(group)
            group_attrs.update(attrs)
            return xr.Dataset(attrs=group_attrs)
        return None

    # Merge HDF5 group attributes with scalar attrs
    group_attrs = _extract_attrs(group)
    group_attrs.update(attrs)

    coords = dict(coords_data)
    ds = xr.Dataset(data_vars, coords=coords, attrs=group_attrs)

    # Assign CRS if projection with epsg_code is present
    epsg = _extract_epsg(group)
    if epsg is not None and "x" in coords and "y" in coords:
        ds = ds.rio.write_crs(epsg)
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

    return ds


def _resolve_dims(
    dataset: h5py.Dataset,
    coords_data: dict[str, np.ndarray],
    var_name: str = "",
) -> tuple[str, ...]:
    """Determine dimension names for a dataset.

    Tries DIMENSION_LIST attribute first (HDF5 dimension scales),
    then falls back to shape matching against known coordinates.
    Unnamed dimensions are prefixed with the variable name to avoid
    conflicts between variables with different shapes in the same group.
    """
    ndim = dataset.ndim

    # Try DIMENSION_LIST references
    if "DIMENSION_LIST" in dataset.attrs:
        dim_list = dataset.attrs["DIMENSION_LIST"]
        if len(dim_list) == ndim:
            dims = _dims_from_dimension_list(dataset, dim_list, coords_data, var_name)
            if dims is not None:
                return dims

    # Fallback: match by shape
    return _dims_from_shape(dataset.shape, coords_data, var_name)


def _dims_from_dimension_list(
    dataset: h5py.Dataset,
    dim_list: np.ndarray,
    coords_data: dict[str, np.ndarray],
    var_name: str = "",
) -> tuple[str, ...] | None:
    """Resolve dims from HDF5 DIMENSION_LIST object references.

    If a specific dim has no reference, uses a generated name for that dim
    while preserving successfully resolved dims.
    """
    h5file = dataset.file
    prefix = f"{var_name}_" if var_name else ""
    dims: list[str] = []
    unnamed_counter = 0
    for i, refs in enumerate(dim_list):
        if len(refs) == 0:
            dims.append(f"{prefix}dim_{unnamed_counter}")
            unnamed_counter += 1
            continue
        ref = refs[0]
        try:
            ref_ds = h5file[ref]
            ref_name = ref_ds.name.split("/")[-1]
            if ref_name in COORD_TO_DIM:
                dims.append(COORD_TO_DIM[ref_name])
            else:
                dims.append(f"{prefix}dim_{unnamed_counter}")
                unnamed_counter += 1
        except Exception:
            dims.append(f"{prefix}dim_{unnamed_counter}")
            unnamed_counter += 1
    return tuple(dims)


def _dims_from_shape(
    shape: tuple[int, ...],
    coords_data: dict[str, np.ndarray],
    var_name: str = "",
) -> tuple[str, ...]:
    """Fallback: match dimensions by size against known coordinates.

    Handles the case where x and y have the same length (square grid)
    by tracking which coordinate dims have already been assigned.
    """
    # Build size → list of candidate dim names
    size_to_dims: dict[int, list[str]] = {}
    for dim_name, arr in coords_data.items():
        size_to_dims.setdefault(len(arr), []).append(dim_name)

    prefix = f"{var_name}_" if var_name else ""
    dims: list[str] = []
    unnamed_counter = 0
    used: set[str] = set()

    for size in shape:
        matched = False
        if size in size_to_dims:
            for candidate in size_to_dims[size]:
                if candidate not in used:
                    dims.append(candidate)
                    used.add(candidate)
                    matched = True
                    break
        if not matched:
            dims.append(f"{prefix}dim_{unnamed_counter}")
            unnamed_counter += 1

    return tuple(dims)


def _get_chunks(
    h5ds: h5py.Dataset,
    dims: tuple[str, ...],
    chunks_spec: dict[str, int] | str | None,
) -> tuple[int, ...] | None:
    """Resolve chunk specification to a concrete chunk tuple.

    Returns None if chunks_spec is None (eager loading).
    """
    if chunks_spec is None:
        return None

    if isinstance(chunks_spec, str) and chunks_spec == "auto":
        if h5ds.chunks is not None:
            return h5ds.chunks
        # Unchunked dataset: use sensible default rather than one giant chunk
        return tuple(min(s, _DEFAULT_CHUNK_SIZE) for s in h5ds.shape)

    if isinstance(chunks_spec, dict):
        return tuple(chunks_spec.get(dim, size) for dim, size in zip(dims, h5ds.shape))

    return tuple(min(s, _DEFAULT_CHUNK_SIZE) for s in h5ds.shape)


def _extract_epsg(group: h5py.Group) -> int | None:
    """Extract EPSG code from a projection dataset in the group, if present."""
    if "projection" not in group:
        return None
    proj = group["projection"]
    if not isinstance(proj, h5py.Dataset):
        return None
    if "epsg_code" in proj.attrs:
        return int(proj.attrs["epsg_code"])
    # Fall back to the dataset value itself
    val = proj[()]
    if isinstance(val, (int, np.integer)):
        return int(val)
    # Handle string/bytes EPSG values
    if isinstance(val, bytes):
        try:
            return int(val.decode().strip())
        except ValueError:
            return None
    if isinstance(val, str):
        try:
            return int(val.strip())
        except ValueError:
            return None
    return None


def _decode_scalar(value: Any) -> Any:
    """Decode an HDF5 scalar value to a Python type."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_array_attr(value: np.ndarray) -> Any:
    """Decode a small HDF5 array to a Python list (for use as an attribute)."""
    if value.dtype.kind == "S":  # byte strings
        return [v.decode("utf-8") for v in value]
    return value.tolist()


def _extract_attrs(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    """Extract HDF5 attributes as a Python dict, decoding bytes to str."""
    result: dict[str, Any] = {}
    for key, value in obj.attrs.items():
        if key in ("DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME"):
            continue  # Skip HDF5 internal attributes
        if isinstance(value, bytes):
            result[key] = value.decode("utf-8")
        elif isinstance(value, np.ndarray):
            if value.dtype.kind == "S":
                result[key] = [v.decode("utf-8") for v in value.flat]
            elif value.size == 1:
                result[key] = value.item()
            else:
                result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result
