"""Extract common metadata from NISAR DataTrees."""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from shapely import wkt

from nisar_pytools.utils.validation import VALID_POLARIZATIONS


def get_product_type(dt: xr.DataTree) -> str:
    """Get the product type (e.g. "GSLC", "GUNW")."""
    return dt.attrs.get("product_type", "")


class AcquisitionTime(NamedTuple):
    """Start times of the acquisitions a product was built from.

    ``secondary`` is ``None`` for single-acquisition products; a GUNW is an
    interferogram between two passes and fills both.
    """

    reference: pd.Timestamp
    secondary: pd.Timestamp | None = None


def get_acquisition_time(dt: xr.DataTree) -> AcquisitionTime:
    """Get the acquisition start time(s) from identification metadata.

    Returns
    -------
    AcquisitionTime
        ``reference`` always; ``secondary`` only for products built from a
        pair, so ``t.reference`` is safe to reach for on any product.

    Examples
    --------
    >>> t = get_acquisition_time(gunw)          # doctest: +SKIP
    >>> (t.secondary - t.reference).days        # doctest: +SKIP
    12
    """
    ident = dt["science/LSAR/identification"].dataset.attrs
    # GUNW names its two passes and carries no bare zeroDopplerStartTime, so
    # reading only that name returned NaT for every interferogram.
    if "zeroDopplerStartTime" in ident:
        return AcquisitionTime(pd.Timestamp(ident["zeroDopplerStartTime"]))
    return AcquisitionTime(
        pd.Timestamp(ident.get("referenceZeroDopplerStartTime", "")),
        pd.Timestamp(ident.get("secondaryZeroDopplerStartTime", "")),
    )


def get_orbit_info(dt: xr.DataTree) -> dict:
    """Get orbit information from identification metadata.

    Returns
    -------
    dict
        Keys: ``track_number``, ``frame_number``, ``orbit_direction``,
        ``absolute_orbit_number``.
    """
    ident = dt["science/LSAR/identification"].dataset.attrs
    return {
        "track_number": ident.get("trackNumber"),
        "frame_number": ident.get("frameNumber"),
        "orbit_direction": ident.get("orbitPassDirection"),
        "absolute_orbit_number": ident.get("absoluteOrbitNumber"),
    }


def get_slc(
    dt: xr.DataTree,
    polarization: str = "HH",
    frequency: str = "frequencyA",
    valid_mask: bool = True,
) -> xr.DataArray:
    """Extract a polarization channel from a GSLC DataTree.

    Parameters
    ----------
    dt : xr.DataTree
        Opened GSLC DataTree from :func:`open_nisar`.
    polarization : str
        Polarization to extract (e.g. ``"HH"``, ``"HV"``). Default ``"HH"``.
    frequency : str
        Frequency group (default ``"frequencyA"``).
    valid_mask : bool
        If ``True`` (default), apply the product's ``mask`` dataset so invalid
        pixels become ``NaN``. The GSLC mask encodes the subswath number for
        valid samples (1..N); ``0`` means at least one RSLC pixel in the
        interpolation window was partially-focused or invalid, and ``255``
        means the pixel sits outside the acquisition extent. Pixels are kept
        only where ``mask != 0 and mask != 255``.

    Returns
    -------
    xr.DataArray
        Complex SLC DataArray with ``y``/``x`` coordinates. If
        ``valid_mask=True``, masked pixels are ``complex(nan, nan)``.

    Raises
    ------
    ValueError
        If the product type is not GSLC, or the requested frequency/polarization
        is not available, or ``valid_mask=True`` but no mask exists.
    """
    product_type = get_product_type(dt)
    if product_type and product_type != "GSLC":
        raise ValueError(f"get_slc expects a GSLC DataTree, got '{product_type}'")

    grids_path = f"science/LSAR/GSLC/grids/{frequency}"
    try:
        ds = dt[grids_path].dataset
    except KeyError:
        available = [
            k for k in dt["science/LSAR/GSLC/grids"].children
        ]
        raise ValueError(
            f"Frequency '{frequency}' not found. Available: {available}"
        )

    if polarization not in ds:
        available = [k for k in ds.data_vars if k in VALID_POLARIZATIONS]
        raise ValueError(
            f"Polarization '{polarization}' not found in {frequency}. "
            f"Available: {available}"
        )

    slc = ds[polarization]
    if valid_mask:
        if "mask" not in ds:
            raise ValueError(
                f"valid_mask=True but no 'mask' variable found in {grids_path}"
            )
        mask = ds["mask"]
        # Keep pixels with a real subswath number; drop invalid (0) and fill (255).
        # Preserve encoding (e.g. grid_mapping) so rio.crs survives .where().
        encoding = slc.encoding
        slc = slc.where((mask != 0) & (mask != 255))
        slc.encoding = encoding
    return slc


def _keep_encoding(obj, keep):
    """``obj.where(keep)``, preserving encoding so ``grid_mapping`` survives.

    ``.where()`` drops encoding, and rioxarray reads the CRS out of it.
    """
    if isinstance(obj, xr.Dataset):
        encodings = {name: var.encoding for name, var in obj.data_vars.items()}
        out = obj.where(keep)
        for name, encoding in encodings.items():
            out[name].encoding = encoding
        return out
    encoding = obj.encoding
    out = obj.where(keep)
    out.encoding = encoding
    return out


def get_gunw(
    dt: xr.DataTree,
    variable: str | None = None,
    polarization: str = "HH",
    layer: str = "unwrappedInterferogram",
    frequency: str = "frequencyA",
    valid_mask: bool = True,
) -> xr.Dataset | xr.DataArray:
    """Read a GUNW layer, or one variable out of it.

    A GUNW layer holds several co-registered variables on one grid, so the
    default is to return all of them together: ``unwrappedInterferogram``
    carries the phase, its coherence, the connected components and the
    ionosphere screen.

    The three layers are *not* on a common grid. In a typical product the
    unwrapped interferogram and pixel offsets are posted at 80 m while the
    wrapped interferogram is at 20 m, so they cannot be combined into one
    Dataset without resampling. Ask for one layer at a time.

    Parameters
    ----------
    dt : xr.DataTree
        Opened GUNW DataTree from :func:`open_nisar`.
    variable : str, optional
        A single data variable inside the ``layer``/``polarization`` group,
        e.g. ``"unwrappedPhase"``, ``"coherenceMagnitude"``,
        ``"connectedComponents"``, ``"wrappedInterferogram"``,
        ``"alongTrackOffset"``. ``None`` (default) returns the whole layer as
        a Dataset.
    polarization : str
        Polarization (e.g. ``"HH"``, ``"VV"``). Default ``"HH"``.
    layer : str
        GUNW layer group: ``"unwrappedInterferogram"`` (default),
        ``"wrappedInterferogram"``, or ``"pixelOffsets"``.
    frequency : str
        Frequency group (default ``"frequencyA"``).
    valid_mask : bool
        If ``True`` (default), apply the layer's ``mask`` so invalid pixels
        become ``NaN``. The GUNW mask is a three-digit ``WRS`` integer:
        ``W`` is the reference water flag (1 water, 0 land), ``R`` is the
        reference RSLC subswath number, ``S`` is the secondary RSLC subswath
        number. Either subswath digit being ``0`` marks an invalid sample;
        ``255`` is fill outside the acquisition extent. Pixels are kept where
        both subswath digits are nonzero (water is *not* dropped — mask water
        separately if needed).

    Returns
    -------
    xr.Dataset or xr.DataArray
        The whole layer as a Dataset, or a single DataArray when ``variable``
        is given. Either way with ``y``/``x`` coordinates. With
        ``valid_mask=True``, integer-typed variables (e.g.
        ``connectedComponents``) are promoted to float to hold ``NaN``.

    Raises
    ------
    ValueError
        If the product type is not GUNW, or the requested frequency/layer/
        polarization/variable is not available, or ``valid_mask=True`` but
        no mask exists for the layer.
    """
    product_type = get_product_type(dt)
    if product_type and product_type != "GUNW":
        raise ValueError(f"get_gunw expects a GUNW DataTree, got '{product_type}'")

    freq_path = f"science/LSAR/GUNW/grids/{frequency}"
    layer_path = f"{freq_path}/{layer}"

    try:
        layer_ds = dt[layer_path].dataset
    except KeyError:
        try:
            available_layers = list(dt[freq_path].children)
        except KeyError:
            available_freqs = list(dt["science/LSAR/GUNW/grids"].children)
            raise ValueError(
                f"Frequency '{frequency}' not found. Available: {available_freqs}"
            )
        raise ValueError(
            f"Layer '{layer}' not found in {frequency}. Available: {available_layers}"
        )

    pol_path = f"{layer_path}/{polarization}"
    try:
        pol_ds = dt[pol_path].dataset
    except KeyError:
        available_pols = [c for c in dt[layer_path].children if c != "mask"]
        raise ValueError(
            f"Polarization '{polarization}' not found in {layer}. "
            f"Available: {available_pols}"
        )

    if variable is None:
        # .dataset hands back a read-only DatasetView; to_dataset() gives a
        # real Dataset so callers can add variables to what they get.
        out = dt[pol_path].to_dataset()
    else:
        if variable not in pol_ds:
            raise ValueError(
                f"Variable '{variable}' not found in {layer}/{polarization}. "
                f"Available: {list(pol_ds.data_vars)}"
            )
        out = pol_ds[variable]

    if valid_mask:
        if "mask" not in layer_ds:
            raise ValueError(
                f"valid_mask=True but no 'mask' variable found in {layer_path}"
            )
        mask = layer_ds["mask"]
        ref_subswath = (mask // 10) % 10
        sec_subswath = mask % 10
        keep = (ref_subswath != 0) & (sec_subswath != 0) & (mask != 255)
        out = _keep_encoding(out, keep)
    return out


def get_bounding_polygon(dt: xr.DataTree) -> Polygon:
    """Get the bounding polygon from identification metadata.

    Returns
    -------
    shapely.geometry.Polygon
    """
    ident = dt["science/LSAR/identification"].dataset.attrs
    poly_wkt = ident.get("boundingPolygon", "")
    if not poly_wkt:
        raise ValueError("No boundingPolygon found in identification metadata")
    return wkt.loads(poly_wkt)
