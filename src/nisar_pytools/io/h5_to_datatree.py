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
    ds, offgrid = _build_dataset(group, chunks, lock)
    if ds is not None:
        datasets[path] = ds
        # Variables that cannot sit on this group's grid become their own child
        # nodes. HDF5 names are unique within a group, so these cannot collide
        # with a real child group.
        for name, var_ds in offgrid.items():
            datasets[f"/{name}" if path == "/" else f"{path}/{name}"] = var_ds
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
) -> tuple[xr.Dataset | None, dict[str, xr.Dataset]]:
    """Build an xr.Dataset from the datasets within a single HDF5 group.

    Returns ``(dataset, offgrid)``. ``dataset`` is None if the group contains no
    datasets (only subgroups). ``offgrid`` maps variable name to a single-variable
    Dataset for any layer that does not sit on the group's x/y grid; the caller
    hangs these off the tree as their own nodes. See :func:`_split_offgrid_vars`.
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

    # Second pass: build data variables (2D+ arrays, now that coords are known).
    # dim_sizes accumulates the length claimed by each dim name so that a layer
    # on a different grid cannot redefine a name another layer already owns.
    dim_sizes = {dim: len(arr) for dim, arr in coords_data.items()}

    for key in group.keys():
        item = group[key]
        if not isinstance(item, h5py.Dataset):
            continue
        if item.ndim < 2:
            continue

        dims = _resolve_dims(item, coords_data, key)
        dims = _deconflict_dims(dims, item.shape, dim_sizes, key)
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
            return xr.Dataset(attrs=group_attrs), {}
        return None, {}

    # Merge HDF5 group attributes with scalar attrs
    group_attrs = _extract_attrs(group)
    group_attrs.update(attrs)

    offgrid = _split_offgrid_vars(data_vars, coords_data)

    coords = dict(coords_data)
    ds = xr.Dataset(data_vars, coords=coords, attrs=group_attrs)

    # Assign CRS if projection with epsg_code is present
    epsg = _extract_epsg(group)
    if epsg is not None and "x" in coords and "y" in coords:
        ds = ds.rio.write_crs(epsg)
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

    return ds, {k: xr.Dataset({k: v}) for k, v in offgrid.items()}


def _split_offgrid_vars(
    data_vars: dict[str, xr.Variable],
    coords_data: dict[str, np.ndarray],
) -> dict[str, xr.Variable]:
    """Remove and return variables that do not sit on the group's x/y grid.

    Such a variable makes the whole Dataset unusable: rioxarray's Dataset-level
    operations (``clip_box``, ``reproject``) require x and y on *every* data
    variable and raise on the first one that lacks them.

    This is not hypothetical. Provisional GSLC products size frequencyB's
    ``inputDataExceptionMask`` from the frequencyA grid (4x too wide) while
    attaching frequencyB's own coordinate scales to it, so the array carries no
    valid georeferencing for the group it is stored in. Rather than drop it, the
    caller re-homes it as its own node so the data stays reachable.
    """
    grid_dims = {d for d in ("x", "y") if d in coords_data}
    if not grid_dims:
        return {}

    offgrid = {}
    for name in list(data_vars):
        if not grid_dims <= set(data_vars[name].dims):
            offgrid[name] = data_vars.pop(name)
            log.warning(
                "%s is not on this group's x/y grid (dims %s); moving it to its "
                "own node so the rest of the group stays usable",
                name,
                offgrid[name].dims,
            )
    return offgrid


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


def _deconflict_dims(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    dim_sizes: dict[str, int],
    var_name: str = "",
) -> tuple[str, ...]:
    """Rename any axis that would redefine the length of an existing dim.

    ``dim_sizes`` maps dim name → length already claimed within the group; it is
    updated in place. NISAR products routinely attach the xCoordinates /
    yCoordinates dimension scales to layers on a different grid (e.g. the
    provisional GSLC ``inputDataExceptionMask``, posted 4-8x finer than the
    frequencyB grid it sits in). Trusting those references would make the
    group's Dataset unbuildable, so the odd axis gets a generated name instead.
    """
    prefix = f"{var_name}_" if var_name else ""
    out: list[str] = []

    for axis, dim in enumerate(dims):
        size = shape[axis]
        if dim_sizes.get(dim, size) != size:
            new_dim = f"{prefix}dim_{axis}"
            while dim_sizes.get(new_dim, size) != size:
                new_dim += "_"
            log.warning(
                "%s axis %d has length %d but dimension %r is already %d in this "
                "group; naming it %r instead",
                var_name or "dataset",
                axis,
                size,
                dim,
                dim_sizes[dim],
                new_dim,
            )
            dim = new_dim
        dim_sizes[dim] = size
        out.append(dim)

    return tuple(out)


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
