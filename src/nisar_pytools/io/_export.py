"""Export NISAR data to Zarr and NetCDF formats."""

from __future__ import annotations

import os
from pathlib import Path

import xarray as xr


def to_zarr(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
    path: str | os.PathLike,
    mode: str = "w",
    **kwargs,
) -> Path:
    """Save data to a Zarr store.

    Parameters
    ----------
    data : xr.DataArray, xr.Dataset, or xr.DataTree
        Data to save. DataArrays are converted to Datasets first.
    path : str or Path
        Output Zarr store path.
    mode : str
        Write mode: ``"w"`` (overwrite) or ``"a"`` (append). Default ``"w"``.
    **kwargs
        Additional keyword arguments passed to ``.to_zarr()``.

    Returns
    -------
    Path
        Path to the written Zarr store.
    """
    path = Path(path)

    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name=data.name or "data")

    data.to_zarr(path, mode=mode, **kwargs)
    return path


def to_netcdf(
    data: xr.DataArray | xr.Dataset,
    path: str | os.PathLike,
    split_complex: bool = True,
    **kwargs,
) -> Path:
    """Save data to a NetCDF file.

    Complex-valued variables are split into ``_real`` and ``_imag``
    components by default, since standard NetCDF does not support
    complex dtypes natively.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data to save. DataArrays are converted to Datasets first.
    path : str or Path
        Output NetCDF file path.
    split_complex : bool
        If ``True`` (default), split complex variables into real/imag
        pairs. If ``False``, pass through as-is (may require
        ``engine="h5netcdf"`` and compatible h5netcdf version).
    **kwargs
        Additional keyword arguments passed to ``.to_netcdf()``.

    Returns
    -------
    Path
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name=data.name or "data")

    if split_complex:
        data = _split_complex_vars(data)

    data.to_netcdf(path, **kwargs)
    return path


def read_netcdf(
    path: str | os.PathLike,
    merge_complex: bool = True,
    **kwargs,
) -> xr.Dataset:
    """Read a NetCDF file, recombining split complex variables.

    Parameters
    ----------
    path : str or Path
        Path to the NetCDF file.
    merge_complex : bool
        If ``True`` (default), recombine ``_real``/``_imag`` pairs
        back into complex variables.
    **kwargs
        Additional keyword arguments passed to ``xr.open_dataset()``.

    Returns
    -------
    xr.Dataset
    """
    ds = xr.open_dataset(path, **kwargs)
    if merge_complex:
        ds = _merge_complex_vars(ds)
    return ds


def _split_complex_vars(ds: xr.Dataset) -> xr.Dataset:
    """Split complex variables into real/imag pairs."""
    import numpy as np

    new_vars = {}
    for name, var in ds.data_vars.items():
        if np.iscomplexobj(var):
            new_vars[f"{name}_real"] = var.real.astype(np.float32)
            new_vars[f"{name}_imag"] = var.imag.astype(np.float32)
            new_vars[f"{name}_real"].attrs = var.attrs
            new_vars[f"{name}_real"].attrs["_complex_component"] = "real"
            new_vars[f"{name}_imag"].attrs["_complex_component"] = "imag"
        else:
            new_vars[name] = var
    return xr.Dataset(new_vars, coords=ds.coords, attrs=ds.attrs)


def _merge_complex_vars(ds: xr.Dataset) -> xr.Dataset:
    """Recombine real/imag pairs into complex variables."""
    import numpy as np

    merged = {}
    skip = set()
    var_names = list(ds.data_vars)

    for name in var_names:
        if name.endswith("_real"):
            base = name[:-5]
            imag_name = f"{base}_imag"
            if imag_name in var_names:
                merged[base] = (ds[name] + 1j * ds[imag_name]).astype(np.complex64)
                # Restore attrs from the real component (minus the marker)
                attrs = {k: v for k, v in ds[name].attrs.items() if k != "_complex_component"}
                merged[base].attrs = attrs
                skip.add(name)
                skip.add(imag_name)

    new_vars = {n: ds[n] for n in var_names if n not in skip}
    new_vars.update(merged)
    return xr.Dataset(new_vars, coords=ds.coords, attrs=ds.attrs)
