"""Export NISAR data to Zarr and NetCDF formats."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


def to_zarr(
    data: xr.DataArray | xr.Dataset,
    path: str | os.PathLike,
    mode: str = "w",
    **kwargs,
) -> Path:
    """Save data to a Zarr store.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
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
    path.parent.mkdir(parents=True, exist_ok=True)

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

    The returned Dataset is eagerly loaded into memory so the file
    handle is closed immediately. For lazy loading, use
    ``xr.open_dataset()`` directly.

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
    with xr.open_dataset(path, **kwargs) as ds:
        ds = ds.load()
    if merge_complex:
        ds = _merge_complex_vars(ds)
    return ds


def _split_complex_vars(ds: xr.Dataset) -> xr.Dataset:
    """Split complex variables into real/imag pairs."""
    new_vars: dict[str, xr.DataArray] = {}
    for name, var in ds.data_vars.items():
        if np.iscomplexobj(var):
            real_da = var.real.astype(np.float32)
            imag_da = var.imag.astype(np.float32)
            # Copy attrs independently to avoid shared-dict mutation
            real_da.attrs = {**var.attrs, "_complex_component": "real"}
            imag_da.attrs = {**var.attrs, "_complex_component": "imag"}
            new_vars[f"{name}_real"] = real_da
            new_vars[f"{name}_imag"] = imag_da
        else:
            new_vars[name] = var
    return xr.Dataset(new_vars, coords=ds.coords, attrs=ds.attrs)


def _merge_complex_vars(ds: xr.Dataset) -> xr.Dataset:
    """Recombine real/imag pairs into complex variables.

    Preserves variable ordering — merged variables are placed at the
    position of their ``_real`` counterpart. Orphan ``_real`` or ``_imag``
    variables without a matching pair are passed through with a warning.
    """
    merged: dict[str, xr.DataArray] = {}
    skip: set[str] = set()
    var_names = list(ds.data_vars)
    var_set = set(var_names)

    # First pass: find matching pairs
    for name in var_names:
        if name.endswith("_real"):
            base = name[:-5]
            imag_name = f"{base}_imag"
            if imag_name in var_set:
                # Efficient complex construction — stay in float32
                real = ds[name].values.astype(np.float32)
                imag = ds[imag_name].values.astype(np.float32)
                complex_da = xr.DataArray(
                    (real + 1j * imag).astype(np.complex64),
                    dims=ds[name].dims,
                    coords=ds[name].coords,
                )
                # Restore attrs (minus the marker)
                complex_da.attrs = {
                    k: v for k, v in ds[name].attrs.items() if k != "_complex_component"
                }
                merged[base] = complex_da
                skip.add(name)
                skip.add(imag_name)
            else:
                log.warning("Orphan _real variable without matching _imag: %s", name)
        elif name.endswith("_imag") and name not in skip:
            base = name[:-5]
            real_name = f"{base}_real"
            if real_name not in var_set:
                log.warning("Orphan _imag variable without matching _real: %s", name)

    # Build output preserving order — merged vars go at the _real position
    new_vars: dict[str, xr.DataArray] = OrderedDict()
    for name in var_names:
        if name in skip:
            if name.endswith("_real"):
                base = name[:-5]
                new_vars[base] = merged[base]
            # Skip _imag (already handled via _real)
        else:
            new_vars[name] = ds[name]

    return xr.Dataset(new_vars, coords=ds.coords, attrs=ds.attrs)
