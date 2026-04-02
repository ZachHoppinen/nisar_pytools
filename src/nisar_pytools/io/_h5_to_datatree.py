"""Generic HDF5 to xarray DataTree converter with lazy (dask-backed) arrays."""

from __future__ import annotations

from typing import Any

import dask.array as da
import h5py
import numpy as np
import xarray as xr

# 1D datasets with these names are treated as dimension coordinates
COORD_NAMES = {"xCoordinates", "yCoordinates", "heightAboveEllipsoid"}
# Mapping from coordinate dataset names to short dimension names
COORD_TO_DIM = {
    "xCoordinates": "x",
    "yCoordinates": "y",
    "heightAboveEllipsoid": "z",
}
# Dimension ordering: first dim is y (rows), second is x (cols), third is z if present
DIM_ORDER_2D = ("y", "x")
DIM_ORDER_3D = ("z", "y", "x")


def h5_to_datatree(
    h5file: h5py.File,
    root: str = "/",
    chunks: dict[str, int] | str | None = "auto",
) -> xr.DataTree:
    """Convert an HDF5 file to an xarray DataTree with lazy dask arrays.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file handle. Must remain open for the lifetime of the DataTree.
    root : str
        Root group path to start from. Default ``"/"``.
    chunks : dict, "auto", or None
        Chunk specification for dask arrays.
        - ``"auto"``: Use the HDF5 file's native chunk sizes.
        - dict: e.g. ``{"y": 512, "x": 512}``
        - ``None``: Load eagerly (not recommended for large files).

    Returns
    -------
    xr.DataTree
    """
    group = h5file[root] if root != "/" else h5file
    datasets = _walk_group(group, "/", chunks)
    return xr.DataTree.from_dict(datasets)


def _walk_group(
    group: h5py.Group,
    path: str,
    chunks: dict[str, int] | str | None,
) -> dict[str, xr.Dataset]:
    """Recursively walk HDF5 groups, building a flat dict of path → Dataset."""
    datasets: dict[str, xr.Dataset] = {}

    # Build a dataset for this group if it contains any datasets
    ds = _build_dataset(group, chunks)
    if ds is not None:
        datasets[path] = ds
    elif group.attrs:
        # Group has HDF5 attributes but no child datasets — store attrs only
        datasets[path] = xr.Dataset(attrs=_extract_attrs(group))

    # Recurse into child groups
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            child_path = f"{path.rstrip('/')}/{key}" if path != "/" else f"/{key}"
            datasets.update(_walk_group(item, child_path, chunks))

    return datasets


def _build_dataset(
    group: h5py.Group,
    chunks: dict[str, int] | str | None,
) -> xr.Dataset | None:
    """Build an xr.Dataset from the datasets within a single HDF5 group.

    Returns None if the group contains no datasets (only subgroups).
    """
    coords_data: dict[str, np.ndarray] = {}
    coord_dim_map: dict[str, str] = {}
    data_vars: dict[str, xr.Variable] = {}
    attrs: dict[str, Any] = {}

    # First pass: identify coordinate datasets and scalars
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

    # Second pass: build data variables (2D+ arrays)
    for key in group.keys():
        item = group[key]
        if not isinstance(item, h5py.Dataset):
            continue
        if item.shape == () or (item.ndim == 1 and key in COORD_NAMES):
            continue
        if item.ndim == 1 and key not in COORD_NAMES:
            # 1D non-coordinate dataset (e.g., listOfPolarizations) → attribute
            attrs[key] = _decode_array_attr(item[()])
            continue

        dims = _resolve_dims(item, coords_data, key)
        dask_chunks = _get_chunks(item, dims, chunks)
        if dask_chunks is not None:
            arr = da.from_array(item, chunks=dask_chunks, lock=False)
        else:
            arr = item[()]

        var_attrs = _extract_attrs(item)
        data_vars[key] = xr.Variable(dims, arr, attrs=var_attrs)

    if not data_vars and not coords_data:
        # Group has only scalars or nothing — still store attrs if present
        if attrs:
            group_attrs = _extract_attrs(group)
            group_attrs.update(attrs)
            return xr.Dataset(attrs=group_attrs)
        return None

    # Merge HDF5 group attributes
    group_attrs = _extract_attrs(group)
    group_attrs.update(attrs)

    coords = {name: (name, arr) for name, arr in coords_data.items()}
    return xr.Dataset(data_vars, coords=coords, attrs=group_attrs)


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
    """Resolve dims from HDF5 DIMENSION_LIST object references."""
    h5file = dataset.file
    prefix = f"{var_name}_" if var_name else ""
    dims: list[str] = []
    for i, refs in enumerate(dim_list):
        if len(refs) == 0:
            # No reference for this dim — fall back
            return None
        ref = refs[0]
        try:
            ref_ds = h5file[ref]
            ref_name = ref_ds.name.split("/")[-1]
            if ref_name in COORD_TO_DIM:
                dims.append(COORD_TO_DIM[ref_name])
            else:
                dims.append(f"{prefix}dim_{i}")
        except Exception:
            return None
    return tuple(dims)


def _dims_from_shape(
    shape: tuple[int, ...],
    coords_data: dict[str, np.ndarray],
    var_name: str = "",
) -> tuple[str, ...]:
    """Fallback: match dimensions by size against known coordinates."""
    size_to_dim: dict[int, str] = {}
    for dim_name, arr in coords_data.items():
        size_to_dim[len(arr)] = dim_name

    prefix = f"{var_name}_" if var_name else ""
    dims: list[str] = []
    unnamed_counter = 0
    for i, size in enumerate(shape):
        if size in size_to_dim:
            dims.append(size_to_dim[size])
        else:
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
        return h5ds.shape

    if isinstance(chunks_spec, dict):
        return tuple(chunks_spec.get(dim, size) for dim, size in zip(dims, h5ds.shape))

    return h5ds.shape


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
