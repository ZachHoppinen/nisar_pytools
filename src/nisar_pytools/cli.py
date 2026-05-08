"""Command-line interface for nisar_pytools.

Single binary ``nisar_pytools`` with subcommands.

Currently supported:
    nisar_pytools to-geotiff <h5> [--band ...] [--pol ...] [--freq ...]
                                  [--output-dir ...] [--bbox ...]
                                  [--bbox-wgs84 ...]
    nisar_pytools info <h5> [--json]

Subcommand: ``to-geotiff``
    Convert a NISAR HDF5 file (GUNW or GSLC) into one or more GeoTIFFs.

    GSLC bands (default-all = ``amplitude``):
        amplitude  -- 10*log10(|SLC|^2)  in dB

    GUNW bands (default-all = unwrapped_phase, wrapped_phase, coherence,
    ionosphere):
        unwrapped_phase -- unwrappedInterferogram/<pol>/unwrappedPhase
        wrapped_phase   -- np.angle(wrappedInterferogram/<pol>/wrappedInterferogram)
        coherence       -- unwrappedInterferogram/<pol>/coherenceMagnitude
                           (multilooked grid, matches unwrapped_phase)
        ionosphere      -- unwrappedInterferogram/<pol>/ionospherePhaseScreen

If ``--band`` is omitted the default-all set is written. If ``--output-dir``
is omitted, GeoTIFFs are placed next to the input HDF5 file. Output naming:
``<h5_stem>_<band>_<pol>.tif``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

import dask.array as dask_array
import numpy as np
import pandas as pd
import xarray as xr

from nisar_pytools import open_nisar
from nisar_pytools.utils.conversion import to_db
from nisar_pytools.utils.metadata import get_bounding_polygon, get_product_type, get_slc
from nisar_pytools.utils.overlap import reproject_bbox
from nisar_pytools.utils.validation import (
    VALID_POLARIZATIONS,
    validate_frequency,
    validate_polarization,
)

log = logging.getLogger("nisar_pytools")


# Per-product band catalog. The order is the default-all order.
_GSLC_BANDS: tuple[str, ...] = ("amplitude",)
_GUNW_BANDS: tuple[str, ...] = (
    "unwrapped_phase",
    "wrapped_phase",
    "coherence",
    "ionosphere",
)


def _normalize_freq(freq: str) -> str:
    """Accept 'A'/'B' or 'frequencyA'/'frequencyB' and return the full key.

    Delegates final validation to :func:`utils.validation.validate_frequency`
    so error messages match the rest of the package.
    """
    full = freq if freq.startswith("frequency") else f"frequency{freq.upper()}"
    try:
        return validate_frequency(full)
    except ValueError as exc:
        raise SystemExit(str(exc))


def _resolve_pol(
    dt: xr.DataTree, product: str, freq: str, requested: str | None
) -> str:
    """Pick a polarization, defaulting to the first available if not given.

    Discovery rules:
      * GSLC: polarizations are data_vars under ``grids/<freq>``.
      * GUNW: polarizations are subgroups under
        ``grids/<freq>/unwrappedInterferogram``.

    The requested name is checked against the package-wide
    :data:`utils.validation.VALID_POLARIZATIONS` set (via
    :func:`validate_polarization`) before being matched against
    what is actually present in the file.
    """
    if product == "GSLC":
        ds = dt[f"science/LSAR/GSLC/grids/{freq}"].dataset
        available = [p for p in ds.data_vars if p in VALID_POLARIZATIONS]
    else:  # GUNW
        node = dt[f"science/LSAR/GUNW/grids/{freq}/unwrappedInterferogram"]
        available = [k for k in node.children if k in VALID_POLARIZATIONS]

    if not available:
        raise SystemExit(f"No polarizations found in {freq}")

    if requested is None:
        return available[0]
    try:
        validate_polarization(requested)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if requested not in available:
        raise SystemExit(
            f"Polarization '{requested}' not found in {freq}. "
            f"Available: {available}"
        )
    return requested


def _gslc_amplitude(dt: xr.DataTree, freq: str, pol: str) -> xr.DataArray:
    """SLC -> dB amplitude on geocoded grid: 10*log10(|SLC|^2).

    Uses :func:`utils.metadata.get_slc` to pull the complex SLC (CRS
    already attached by the reader) and :func:`utils.conversion.to_db`
    with ``power=False`` so the formula is ``20*log10(|SLC|)``, which
    is identical to ``10*log10(|SLC|^2)`` -- the standard SAR dB
    backscatter convention.
    """
    slc = get_slc(dt, polarization=pol, frequency=freq)
    db = to_db(slc, power=False).astype("float32")
    # to_db copies all input attrs onto the result, but the SLC carries a
    # complex-typed _FillValue / nodata that rioxarray cannot coerce to
    # float when writing the dB raster. Reset to a clean attribute set.
    db.attrs = {"long_name": "Amplitude", "units": "dB"}
    if slc.rio.crs is not None:
        db.rio.write_crs(slc.rio.crs, inplace=True)
    return db


def _gunw_band(dt: xr.DataTree, freq: str, pol: str, band: str) -> xr.DataArray:
    """Return a 2D DataArray for the requested GUNW band, with CRS preserved."""
    base = f"science/LSAR/GUNW/grids/{freq}"
    if band == "unwrapped_phase":
        return dt[f"{base}/unwrappedInterferogram/{pol}"].dataset["unwrappedPhase"]
    if band == "coherence":
        # Multilooked coherence -- matches the unwrapped_phase grid.
        return dt[f"{base}/unwrappedInterferogram/{pol}"].dataset["coherenceMagnitude"]
    if band == "ionosphere":
        return dt[f"{base}/unwrappedInterferogram/{pol}"].dataset[
            "ionospherePhaseScreen"
        ]
    if band == "wrapped_phase":
        # wrappedInterferogram is complex64; phase = np.angle().
        complex_ifg = dt[f"{base}/wrappedInterferogram/{pol}"].dataset[
            "wrappedInterferogram"
        ]
        # Dispatch on backing array type so we (a) keep dask laziness when
        # the source is dask-backed, and (b) avoid the ComplexWarning that
        # xr.apply_ufunc emits when casting its complex-typed dask meta to
        # float during graph construction.
        cdata = complex_ifg.data
        if isinstance(cdata, dask_array.Array):
            pdata = dask_array.angle(cdata).astype(np.float32)
        else:
            pdata = np.angle(cdata).astype(np.float32)
        phase = xr.DataArray(
            pdata,
            dims=complex_ifg.dims,
            coords=complex_ifg.coords,
            attrs={"long_name": "Wrapped phase", "units": "radians"},
        )
        if complex_ifg.rio.crs is not None:
            phase.rio.write_crs(complex_ifg.rio.crs, inplace=True)
        return phase
    raise ValueError(f"Unknown GUNW band '{band}'")


def _write_geotiff(da: xr.DataArray, out_path: Path) -> Path:
    """Stream a (possibly dask-backed) DataArray to a tiled GeoTIFF.

    For dask-backed inputs, ``rio.to_raster`` dispatches to
    ``dask.array.store`` and processes one chunk at a time, so peak
    memory stays bounded by the chunk size rather than the full raster.
    Tiled output (512x512 blocks) is GDAL-friendly for downstream readers.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The lock is critical: rioxarray's to_raster only takes the streaming
    # dask.array.store code path when `lock` is truthy AND the data is a
    # dask collection. With lock=None it silently falls back to calling
    # .values on the underlying xarray variable -- which materializes the
    # entire output array in memory before writing. Passing a real Lock
    # forces chunk-by-chunk streaming and bounds peak memory by chunk size.
    da.rio.to_raster(
        out_path,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        lock=threading.Lock(),
    )
    log.info("Wrote %s  shape=%s dtype=%s", out_path.name, tuple(da.shape), da.dtype)
    return out_path


def _apply_bbox(
    da: xr.DataArray,
    bbox_native: tuple[float, float, float, float],
    band: str,
) -> xr.DataArray:
    """Crop a 2D DataArray to (xmin, ymin, xmax, ymax) in its native CRS.

    Handles either y-axis ordering (ascending or descending). Cropping
    preserves dask-laziness, so the streaming write only touches chunks
    that intersect the bbox.
    """
    xmin, ymin, xmax, ymax = bbox_native
    # Pick the y slice direction to match the coordinate's orientation.
    if float(da.y[0]) < float(da.y[-1]):
        y_slice = slice(ymin, ymax)
    else:
        y_slice = slice(ymax, ymin)
    cropped = da.sel(x=slice(xmin, xmax), y=y_slice)
    if cropped.size == 0:
        raise SystemExit(
            f"Bbox {bbox_native} has no overlap with the {band} grid "
            f"(x:[{float(da.x.min()):.1f}, {float(da.x.max()):.1f}], "
            f"y:[{float(da.y.min()):.1f}, {float(da.y.max()):.1f}])."
        )
    return cropped


def _resolve_bbox(
    args: argparse.Namespace, da: xr.DataArray
) -> tuple[float, float, float, float] | None:
    """Return the bbox to apply, in the data's native CRS, or None.

    ``--bbox`` is taken as-is (assumed already in the file's CRS).
    ``--bbox-wgs84`` is reprojected from EPSG:4326 to the data's CRS via
    :func:`nisar_pytools.utils.overlap.reproject_bbox`.
    """
    # argparse's mutually exclusive group already enforces only-one.
    if args.bbox is not None:
        return tuple(args.bbox)
    if args.bbox_wgs84 is not None:
        target_crs = da.rio.crs
        if target_crs is None:
            raise SystemExit(
                "--bbox-wgs84 requires the file to have a CRS, but none was found."
            )
        # Use to_string() (e.g. "EPSG:32611" or full WKT) instead of
        # to_epsg() so this still works for non-EPSG CRSes -- to_epsg()
        # returns None for those and would propagate as a confusing error.
        return reproject_bbox(
            tuple(args.bbox_wgs84),
            src_crs=4326,
            dst_crs=target_crs.to_string(),
        )
    return None


def cmd_to_geotiff(args: argparse.Namespace) -> None:
    """Implementation of the ``to-geotiff`` subcommand."""
    h5_path = Path(args.h5_path).expanduser().resolve()

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else h5_path.parent
    )
    freq = _normalize_freq(args.freq)

    # Open lazily; CRS is attached per-group by the reader.
    dt = open_nisar(h5_path)
    product = get_product_type(dt)

    if product not in ("GSLC", "GUNW"):
        raise SystemExit(
            f"Unsupported product type '{product}'. "
            f"Only GSLC and GUNW are currently supported."
        )

    pol = _resolve_pol(dt, product, freq, args.pol)

    # Resolve which bands to write.
    all_bands = _GSLC_BANDS if product == "GSLC" else _GUNW_BANDS
    if args.band is None:
        bands = list(all_bands)
    else:
        if args.band not in all_bands:
            raise SystemExit(
                f"Band '{args.band}' is not valid for {product}. "
                f"Valid bands: {list(all_bands)}"
            )
        bands = [args.band]

    log.info(
        "Product=%s  freq=%s  pol=%s  bands=%s  out=%s",
        product, freq, pol, bands, out_dir,
    )

    # Extract + write each band. Bbox is resolved per-band because the
    # source DataArray's CRS is needed for --bbox-wgs84 reprojection.
    for band in bands:
        if product == "GSLC":
            da = _gslc_amplitude(dt, freq, pol)
        else:
            da = _gunw_band(dt, freq, pol, band)

        bbox_native = _resolve_bbox(args, da)
        if bbox_native is not None:
            da = _apply_bbox(da, bbox_native, band)
            log.info("Cropped %s to bbox %s -> shape=%s", band, bbox_native, tuple(da.shape))

        out_path = out_dir / f"{h5_path.stem}_{band}_{pol}.tif"
        _write_geotiff(da, out_path)


# ---------------------------------------------------------------------------
# `info` subcommand
# ---------------------------------------------------------------------------

def _format_bytes(n: float) -> str:
    """Render a byte count as a short human-readable string (B/KB/MB/GB/TB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _coord_resolution(da: xr.DataArray) -> tuple[float, float]:
    """Pixel size in (x, y) from the first two coordinate values."""
    # Falls back to 0.0 if the axis has fewer than 2 samples (1-pixel raster).
    x_res = abs(float(da.x[1] - da.x[0])) if da.x.size > 1 else 0.0
    y_res = abs(float(da.y[1] - da.y[0])) if da.y.size > 1 else 0.0
    return x_res, y_res


def _array_stats(da: xr.DataArray) -> dict | None:
    """Min / max / mean / median / quartiles over the finite pixels of a 2D array.

    Drops non-finite (NaN/inf, off-swath nodata) before computing.
    Returns ``None`` if nothing is left after masking. The full array
    is materialized via ``.values`` -- on multi-looked GUNW grids this
    is ~80 MB so it's cheap; partial-sampling would defeat the point of
    a "stats on this product" summary.
    """
    arr = da.values
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return None
    return {
        "valid_count": int(valid.size),
        "total_count": int(arr.size),
        "valid_fraction": float(valid.size / arr.size),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "q25": float(np.quantile(valid, 0.25)),
        "q75": float(np.quantile(valid, 0.75)),
    }


def _mask_coverage(da_mask: xr.DataArray) -> dict:
    """Fraction of valid pixels in an integer mask band.

    Any nonzero value is treated as valid. NISAR L2 mask layers are
    typically 0=invalid, 1=valid but the convention can vary.
    """
    arr = da_mask.values
    valid = int((arr != 0).sum())
    total = int(arr.size)
    return {
        "valid_count": valid,
        "total_count": total,
        "valid_fraction": (valid / total) if total else 0.0,
    }


def _connected_components_stats(da_cc: xr.DataArray) -> dict | None:
    """Summarize a uint16 connectedComponents band.

    Label 0 is the standard "unconnected / no-component" sentinel.
    Real components start at 1. Reports the count of distinct components,
    pixels assigned to a component, and the largest component's share.
    """
    arr = da_cc.values
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return None
    labels, counts = np.unique(nonzero, return_counts=True)
    largest_idx = int(counts.argmax())
    return {
        "num_components": int(labels.size),
        "connected_pixels": int(nonzero.size),
        "total_pixels": int(arr.size),
        "connected_fraction": float(nonzero.size / arr.size),
        "largest_component_label": int(labels[largest_idx]),
        "largest_component_pixels": int(counts[largest_idx]),
        "largest_component_fraction": float(counts[largest_idx] / nonzero.size),
    }


def _gunw_scene_center_baseline(dt: xr.DataTree, h5_path: Path) -> dict | None:
    """Scene-center perpendicular + parallel baseline (meters), GUNW only.

    NISAR GUNW products carry pre-computed baseline cubes on the radar
    grid: ``metadata/radarGrid/perpendicularBaseline`` and
    ``parallelBaseline`` with shape (height_layers, az, rng). These are
    produced by the JPL processing pipeline using the per-pixel look
    vector and slant range -- properly accounting for the off-nadir
    InSAR geometry that an orbit-only computation gets wrong by
    multiple orders of magnitude.

    We report the median across the middle height layer (a single
    representative value for the scene) and the value at the central
    pixel of that layer (the literal "scene center"). Reading is done
    via h5py because these cubes live under metadata and may not be
    surfaced by the DataTree walker. Returns None on missing fields.
    """
    import h5py

    try:
        with h5py.File(h5_path, "r") as f:
            base = "science/LSAR/GUNW/metadata/radarGrid"
            if f"{base}/perpendicularBaseline" not in f:
                return None
            bperp = np.asarray(f[f"{base}/perpendicularBaseline"][...])
            bpar = np.asarray(f[f"{base}/parallelBaseline"][...])
    except (OSError, KeyError):
        return None

    if bperp.ndim != 3 or bperp.shape != bpar.shape:
        return None

    # Middle height layer is closest to the average terrain; the central
    # pixel of that layer is the literal scene center.
    mid_h = bperp.shape[0] // 2
    layer_perp = bperp[mid_h]
    layer_par = bpar[mid_h]
    cy, cx = layer_perp.shape[0] // 2, layer_perp.shape[1] // 2

    finite_perp = layer_perp[np.isfinite(layer_perp)]
    finite_par = layer_par[np.isfinite(layer_par)]
    if finite_perp.size == 0 or finite_par.size == 0:
        return None

    return {
        "perpendicular_center_m": float(layer_perp[cy, cx])
        if np.isfinite(layer_perp[cy, cx])
        else float(np.median(finite_perp)),
        "parallel_center_m": float(layer_par[cy, cx])
        if np.isfinite(layer_par[cy, cx])
        else float(np.median(finite_par)),
        "perpendicular_median_m": float(np.median(finite_perp)),
        "parallel_median_m": float(np.median(finite_par)),
    }


def _common_geometry(dt: xr.DataTree, sample_da: xr.DataArray) -> dict:
    """Extract CRS, native bbox, and WGS84 bbox shared across product types."""
    crs = sample_da.rio.crs
    crs_str = crs.to_string() if crs is not None else None

    polygon = get_bounding_polygon(dt)
    wgs84_minx, wgs84_miny, wgs84_maxx, wgs84_maxy = polygon.bounds

    return {
        "crs": crs_str,
        "native_bbox": (
            float(sample_da.x.min()),
            float(sample_da.y.min()),
            float(sample_da.x.max()),
            float(sample_da.y.max()),
        ),
        "wgs84_bbox": (wgs84_minx, wgs84_miny, wgs84_maxx, wgs84_maxy),
    }


def _gather_gslc_info(dt: xr.DataTree, h5_path: Path) -> dict:
    """Collect GSLC-relevant metadata + per-frequency grid descriptors."""
    ident = dt["science/LSAR/identification"].dataset.attrs
    info: dict = {
        "file": h5_path.name,
        "size": _format_bytes(h5_path.stat().st_size),
        "size_bytes": h5_path.stat().st_size,
        "product": "GSLC",
        "product_version": ident.get("productVersion", ""),
        "spec_version": ident.get("productSpecificationVersion", ""),
        "processing_datetime": ident.get("processingDateTime", ""),
    }

    # Single acquisition (start -> end, single orbit).
    info["acquisition"] = {
        "start": ident.get("zeroDopplerStartTime", ""),
        "end": ident.get("zeroDopplerEndTime", ""),
        "orbit": int(ident["absoluteOrbitNumber"]) if "absoluteOrbitNumber" in ident else None,
    }

    info["geometry_meta"] = {
        "track": int(ident["trackNumber"]),
        "frame": int(ident["frameNumber"]),
        "direction": ident.get("orbitPassDirection", ""),
        "look": ident.get("lookDirection", ""),
        "radar_band": ident.get("radarBand", ""),
    }

    # Roll up polarizations across frequencies for a top-level summary.
    # For GSLC each frequency has its own pol list, e.g. {A: [HH,HV], B: [HH]}.
    info["polarizations"] = {}

    # Walk grids -> per-frequency shape, resolution, polarizations, mask coverage.
    grids: dict = {}
    grids_node = dt["science/LSAR/GSLC/grids"]
    sample_da: xr.DataArray | None = None
    for freq_name in grids_node.children:
        ds = dt[f"science/LSAR/GSLC/grids/{freq_name}"].dataset
        pols = [p for p in ds.data_vars if p in VALID_POLARIZATIONS]
        if not pols:
            continue
        da = ds[pols[0]]
        if sample_da is None:
            sample_da = da
        x_res, y_res = _coord_resolution(da)
        grid_info = {
            "polarizations": pols,
            "shape": tuple(da.shape),
            "resolution_m": (x_res, y_res),
        }
        # GSLC mask layer is at grids/<freq>/mask -- one mask shared by all pols.
        if "mask" in ds.data_vars:
            grid_info["mask_coverage"] = _mask_coverage(ds["mask"])
        grids[freq_name] = grid_info
        info["polarizations"][freq_name] = pols
    info["grids"] = grids

    # Geometry needs a sample DataArray to read CRS + extent off of.
    if sample_da is not None:
        info["geometry"] = _common_geometry(dt, sample_da)
    return info


def _gather_gunw_info(dt: xr.DataTree, h5_path: Path) -> dict:
    """Collect GUNW-relevant metadata + per-sub-product grids + coherence stats."""
    ident = dt["science/LSAR/identification"].dataset.attrs
    info: dict = {
        "file": h5_path.name,
        "size": _format_bytes(h5_path.stat().st_size),
        "size_bytes": h5_path.stat().st_size,
        "product": "GUNW",
        "product_version": ident.get("productVersion", ""),
        "spec_version": ident.get("productSpecificationVersion", ""),
        "processing_datetime": ident.get("processingDateTime", ""),
    }

    # Reference + secondary acquisitions and their absolute orbits.
    ref_start = pd.Timestamp(ident.get("referenceZeroDopplerStartTime", ""))
    sec_start = pd.Timestamp(ident.get("secondaryZeroDopplerStartTime", ""))
    info["acquisition"] = {
        "reference_start": ident.get("referenceZeroDopplerStartTime", ""),
        "reference_end": ident.get("referenceZeroDopplerEndTime", ""),
        "secondary_start": ident.get("secondaryZeroDopplerStartTime", ""),
        "secondary_end": ident.get("secondaryZeroDopplerEndTime", ""),
        "reference_orbit": int(ident["referenceAbsoluteOrbitNumber"])
        if "referenceAbsoluteOrbitNumber" in ident
        else None,
        "secondary_orbit": int(ident["secondaryAbsoluteOrbitNumber"])
        if "secondaryAbsoluteOrbitNumber" in ident
        else None,
        "temporal_baseline_days": int((sec_start - ref_start).days),
    }

    info["geometry_meta"] = {
        "track": int(ident["trackNumber"]),
        "frame": int(ident["frameNumber"]),
        "direction": ident.get("orbitPassDirection", ""),
        "look": ident.get("lookDirection", ""),
        "radar_band": ident.get("radarBand", ""),
    }
    # Top-level pol summary; populated below as we walk the grids. For GUNW
    # we report the union across sub-products per frequency.
    info["polarizations"] = {}

    # Walk grids/<freq>/<subproduct>/<pol> for shape/resolution/pols.
    # Each sub-product carries its own mask layer at the sub-product level.
    grids: dict = {}
    sample_da: xr.DataArray | None = None
    grids_node = dt["science/LSAR/GUNW/grids"]
    for freq_name in grids_node.children:
        freq_path = f"science/LSAR/GUNW/grids/{freq_name}"
        freq_node = dt[freq_path]
        sub_grids: dict = {}
        for sub_name in freq_node.children:
            sub_path = f"{freq_path}/{sub_name}"
            sub_node = dt[sub_path]
            pols = [k for k in sub_node.children if k in VALID_POLARIZATIONS]
            if not pols:
                continue
            pol_ds = dt[f"{sub_path}/{pols[0]}"].dataset
            data_var_2d = next(
                (n for n, v in pol_ds.data_vars.items() if v.ndim == 2),
                None,
            )
            if data_var_2d is None:
                continue
            da = pol_ds[data_var_2d]
            if sample_da is None:
                sample_da = da
            x_res, y_res = _coord_resolution(da)
            sub_info = {
                "polarizations": pols,
                "shape": tuple(da.shape),
                "resolution_m": (x_res, y_res),
            }
            # Mask sits one level up from the polarization groups.
            sub_ds = sub_node.dataset
            if "mask" in sub_ds.data_vars:
                sub_info["mask_coverage"] = _mask_coverage(sub_ds["mask"])
            sub_grids[sub_name] = sub_info
        if sub_grids:
            grids[freq_name] = sub_grids
            # Union of pols across sub-products of this frequency.
            all_pols: list = []
            for g in sub_grids.values():
                for p in g["polarizations"]:
                    if p not in all_pols:
                        all_pols.append(p)
            info["polarizations"][freq_name] = all_pols
    info["grids"] = grids

    if sample_da is not None:
        info["geometry"] = _common_geometry(dt, sample_da)

    # Per-pol stats on the multi-looked unwrappedInterferogram grid.
    # All three (coherence, unwrapped phase, connected components) live
    # at the same grid so they pair up with the same shape/extent.
    info["coherence"] = {}
    info["unwrapped_phase"] = {}
    info["connected_components"] = {}
    for freq_name, sub_grids in grids.items():
        unw = sub_grids.get("unwrappedInterferogram")
        if not unw:
            continue
        for pol in unw["polarizations"]:
            unw_ds_path = (
                f"science/LSAR/GUNW/grids/{freq_name}/unwrappedInterferogram/{pol}"
            )
            try:
                unw_ds = dt[unw_ds_path].dataset
            except KeyError:
                continue
            key = f"{freq_name}/{pol}"
            if "coherenceMagnitude" in unw_ds.data_vars:
                stats = _array_stats(unw_ds["coherenceMagnitude"])
                if stats is not None:
                    info["coherence"][key] = stats
            if "unwrappedPhase" in unw_ds.data_vars:
                stats = _array_stats(unw_ds["unwrappedPhase"])
                if stats is not None:
                    info["unwrapped_phase"][key] = stats
            if "connectedComponents" in unw_ds.data_vars:
                cc = _connected_components_stats(unw_ds["connectedComponents"])
                if cc is not None:
                    info["connected_components"][key] = cc

    # Spatial baseline (perpendicular + parallel at scene center) read
    # from the file's pre-computed radarGrid cubes.
    baseline = _gunw_scene_center_baseline(dt, h5_path)
    if baseline is not None:
        info["baseline"] = baseline
    return info


def _format_info_text(info: dict) -> str:
    """Render the info dict as a compact human-readable block."""
    lines: list[str] = []
    lines.append(info["file"])
    lines.append(
        f"  product:      {info['product']} (v{info['product_version']}, "
        f"spec {info['spec_version']})"
    )
    lines.append(f"  size:         {info['size']}")
    lines.append(f"  processed:    {info['processing_datetime']}")
    lines.append("")

    # Acquisition block differs between GSLC and GUNW.
    acq = info["acquisition"]
    lines.append("  acquisition:")
    if info["product"] == "GSLC":
        lines.append(f"    time:         {acq['start']}  ->  {acq['end']}")
        lines.append(f"    orbit:        {acq['orbit']}")
    else:
        lines.append(
            f"    reference:    {acq['reference_start']}  ->  "
            f"{acq['reference_end']}  (orbit {acq['reference_orbit']})"
        )
        lines.append(
            f"    secondary:    {acq['secondary_start']}  ->  "
            f"{acq['secondary_end']}  (orbit {acq['secondary_orbit']})"
        )
        lines.append(f"    temporal baseline: {acq['temporal_baseline_days']} days")
    lines.append("")

    geom_meta = info["geometry_meta"]
    geom = info.get("geometry", {})
    lines.append("  geometry:")
    lines.append(f"    track:        {geom_meta['track']}")
    lines.append(f"    frame:        {geom_meta['frame']}")
    lines.append(f"    direction:    {geom_meta['direction']}")
    lines.append(f"    look:         {geom_meta['look']}")
    lines.append(f"    radar band:   {geom_meta['radar_band']}")
    pol_summary = info.get("polarizations") or {}
    if pol_summary:
        # Show as "HH, HV (frequencyA); HH (frequencyB)" -- one line, easy to grep.
        pol_parts = [f"{', '.join(p)} ({fq})" for fq, p in pol_summary.items()]
        lines.append(f"    polarizations: {';  '.join(pol_parts)}")
    if geom:
        lines.append(f"    crs:          {geom['crs']}")
        nb = geom["native_bbox"]
        lines.append(
            f"    extent:       {nb[0]:.1f}, {nb[1]:.1f}  ->  "
            f"{nb[2]:.1f}, {nb[3]:.1f}  (native CRS)"
        )
        wb = geom["wgs84_bbox"]
        lines.append(
            f"    extent (lon/lat): "
            f"{wb[0]:.4f}, {wb[1]:.4f}  ->  {wb[2]:.4f}, {wb[3]:.4f}"
        )
    lines.append("")

    # Grids: GSLC is freq -> {pols, shape, res}; GUNW is freq -> sub -> {pols, shape, res}.
    # Mask coverage is appended to a grid's line if present.
    def _mask_str(grid_dict: dict) -> str:
        mc = grid_dict.get("mask_coverage")
        if mc is None:
            return ""
        return f"  [mask valid {mc['valid_fraction']*100:.1f}%]"

    lines.append("  grids:")
    if info["product"] == "GSLC":
        for freq_name, g in info["grids"].items():
            shape = g["shape"]
            x_res, y_res = g["resolution_m"]
            pols = ", ".join(g["polarizations"])
            lines.append(
                f"    {freq_name}: {shape[0]} x {shape[1]} @ "
                f"{x_res:g} m x {y_res:g} m  (pols: {pols}){_mask_str(g)}"
            )
    else:  # GUNW
        for freq_name, sub_grids in info["grids"].items():
            lines.append(f"    {freq_name}:")
            for sub_name, g in sub_grids.items():
                shape = g["shape"]
                x_res, y_res = g["resolution_m"]
                pols = ", ".join(g["polarizations"])
                lines.append(
                    f"      {sub_name}: {shape[0]} x {shape[1]} @ "
                    f"{x_res:g} m x {y_res:g} m  (pols: {pols}){_mask_str(g)}"
                )

    def _stats_block(title: str, stats_dict: dict) -> None:
        """Append a per-pol stats block (used for coherence + phase)."""
        lines.append("")
        lines.append(f"  {title}:")
        for key, s in stats_dict.items():
            lines.append(
                f"    {key}: valid {s['valid_fraction']*100:.1f}% "
                f"({s['valid_count']:,}/{s['total_count']:,})"
            )
            lines.append(
                f"      min={s['min']:.3f}  q25={s['q25']:.3f}  "
                f"median={s['median']:.3f}  mean={s['mean']:.3f}  "
                f"q75={s['q75']:.3f}  max={s['max']:.3f}"
            )

    if info.get("coherence"):
        _stats_block("coherence (multi-looked grid)", info["coherence"])
    if info.get("unwrapped_phase"):
        _stats_block("unwrapped phase (multi-looked grid, radians)", info["unwrapped_phase"])

    cc = info.get("connected_components")
    if cc:
        lines.append("")
        lines.append("  connected components:")
        for key, c in cc.items():
            lines.append(
                f"    {key}: {c['num_components']} component(s), "
                f"{c['connected_fraction']*100:.1f}% of pixels connected"
            )
            lines.append(
                f"      largest (label {c['largest_component_label']}): "
                f"{c['largest_component_pixels']:,} px "
                f"({c['largest_component_fraction']*100:.1f}% of connected)"
            )

    bl = info.get("baseline")
    if bl is not None:
        lines.append("")
        lines.append("  baseline (from pre-computed radarGrid cube):")
        lines.append(
            f"    scene center -> perpendicular: {bl['perpendicular_center_m']:+.1f} m  "
            f"parallel: {bl['parallel_center_m']:+.1f} m"
        )
        lines.append(
            f"    median       -> perpendicular: {bl['perpendicular_median_m']:+.1f} m  "
            f"parallel: {bl['parallel_median_m']:+.1f} m"
        )

    return "\n".join(lines)


def cmd_info(args: argparse.Namespace) -> None:
    """Implementation of the ``info`` subcommand."""
    h5_path = Path(args.h5_path).expanduser().resolve()
    dt = open_nisar(h5_path)
    product = get_product_type(dt)
    if product == "GSLC":
        info = _gather_gslc_info(dt, h5_path)
    elif product == "GUNW":
        info = _gather_gunw_info(dt, h5_path)
    else:
        raise SystemExit(
            f"Unsupported product type '{product}'. "
            f"Only GSLC and GUNW are currently supported."
        )

    if args.json:
        # default=str so pd.Timestamp / np types serialize cleanly.
        print(json.dumps(info, indent=2, default=str))
    else:
        print(_format_info_text(info))


# ---------------------------------------------------------------------------
# `rslc-to-gunw` subcommand
# ---------------------------------------------------------------------------

def cmd_rslc_to_gunw(args: argparse.Namespace) -> None:
    """Implementation of the ``rslc-to-gunw`` subcommand."""
    from nisar_pytools.processing import rslc_to_gunw

    bbox = tuple(args.bbox) if args.bbox else None  # type: ignore[assignment]
    out = rslc_to_gunw(
        reference_rslc=args.reference_rslc,
        secondary_rslc=args.secondary_rslc,
        output_dir=args.output_dir,
        runconfig=args.runconfig,
        dem_file=args.dem,
        aoi_bbox_utm=bbox,
        output_epsg=args.epsg,
        restart=args.restart,
    )
    print(f"GUNW: {out}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_TOP_DESCRIPTION = """\
Command-line tools for working with NISAR HDF5 products.

Subcommands:
  to-geotiff    Convert GUNW / GSLC HDF5 bands into GeoTIFFs.
  info          Print a summary (product type, track/frame, times, pols,
                grids, coherence stats) for a GUNW or GSLC HDF5 file.
  rslc-to-gunw  Run nisar.workflows.insar on an RSLC pair and produce
                a GUNW (requires the [isce3] optional dep group).

Run `nisar_pytools <subcommand> --help` for subcommand-specific help.
"""

_RSLC_TO_GUNW_DESCRIPTION = """\
Process a NISAR L1 RSLC pair into a NISAR L2 GUNW using
``nisar.workflows.insar`` (the production isce3-based InSAR workflow).

Requires the optional ``[isce3]`` dep group plus conda-installed
``isce3`` and ``libgdal-hdf5``. Recommended install:
  mamba install -c conda-forge isce3 libgdal-hdf5 snaphu-py pygrib pyaps3 raider pysolid netcdf4

If --runconfig is omitted, a bundled production-spec default is used
(JPL X05010 settings: 5x6 crossmul, 13x16 unwrap, full coregistration,
split-spectrum ionosphere correction enabled, troposphere disabled).
Paths, EPSG, and bbox are filled in at runtime.

If --dem is omitted, a Copernicus 30 m DEM covering the reference RSLC's
bounding polygon is fetched via dem_stitcher.

If --epsg / --bbox are omitted, the UTM zone is derived from the scene
centroid and the bbox is the full RSLC extent in that projection.

Multi-hour wall time on CPU.
"""

_INFO_DESCRIPTION = """\
Print a summary of a NISAR HDF5 file: product type, track/frame, orbit
direction, acquisition time(s), available polarizations and frequencies,
per-grid shape + resolution, and (for GUNW) coherence stats on the
multi-looked grid.

Use --json to emit machine-readable output instead of the formatted text.
"""

_TO_GEOTIFF_DESCRIPTION = """\
Convert a NISAR HDF5 file (GUNW or GSLC) into one or more GeoTIFFs.

The product type (GUNW or GSLC) is auto-detected from the file. If --band
is omitted, the full default-all set for that product type is written.
Outputs are named: <h5_stem>_<band>_<pol>.tif
"""

_TO_GEOTIFF_EPILOG = """\
Available bands
---------------
GSLC:
  amplitude        10*log10(|SLC|^2) -- backscatter amplitude in dB.
                   Source: science/LSAR/GSLC/grids/<freq>/<pol>  (complex64)

GUNW:
  unwrapped_phase  Geocoded unwrapped interferometric phase (radians).
                   Source: .../unwrappedInterferogram/<pol>/unwrappedPhase
                   Multi-looked grid (matches coherence and ionosphere).

  wrapped_phase    Wrapped phase, derived as np.angle(complex interferogram).
                   Source: .../wrappedInterferogram/<pol>/wrappedInterferogram
                   Full-resolution grid (NOT the multi-looked grid). Output
                   is float32 radians in [-pi, pi].

  coherence        Interferometric coherence magnitude in [0, 1].
                   Source: .../unwrappedInterferogram/<pol>/coherenceMagnitude
                   Uses the multi-looked grid so it co-registers with
                   unwrapped_phase. (A full-res coherence also exists under
                   wrappedInterferogram but is not exposed here.)

  ionosphere       Ionospheric phase screen (radians).
                   Source: .../unwrappedInterferogram/<pol>/ionospherePhaseScreen

  Note: tropospheric phase screens are 3D radar-grid cubes and require
  DEM-based resampling to geocode -- not currently exposed via the CLI.

Default-all
-----------
GSLC -> amplitude
GUNW -> unwrapped_phase, wrapped_phase, coherence, ionosphere

Polarization & frequency
------------------------
NISAR products may carry multiple polarizations (HH, HV, VH, VV) and two
frequencies (A and B). Defaults: first available pol, frequencyA. Use
--pol / --freq to override.

Subsetting
----------
Use --bbox or --bbox-wgs84 to crop the output to an AOI. The two flags
are mutually exclusive; pick whichever matches the units you have.

  --bbox XMIN YMIN XMAX YMAX
      Crop in the file's *native* CRS (typically UTM meters for NISAR
      L2 products -- check with `gdalinfo` or look at the file's
      `projection` scalar). All four numbers must be in the same CRS
      as the data; no reprojection is performed.

      Example: --bbox 324000 4900000 410000 4960000   (UTM 11N meters)

  --bbox-wgs84 LON_MIN LAT_MIN LON_MAX LAT_MAX
      Crop using WGS84 longitude/latitude (EPSG:4326). The four
      corners are reprojected into the file's native CRS via pyproj,
      and the enclosing axis-aligned bbox in native coordinates is
      used for the crop. This means the cropped raster is slightly
      larger than the requested lat/lon rectangle (since lat/lon
      rectangles do not stay rectangular after reprojection).

      Example: --bbox-wgs84 -118.5 41.0 -118.3 41.2

Cropping is applied *before* the streaming write, so only dask chunks
intersecting the bbox are read from the HDF5 file. This is how a small
AOI can be extracted from a 41 GB GSLC in seconds with peak memory in
the hundreds of MB.

If the bbox does not overlap the data extent at all the command exits
with an error message that includes the file's actual extent.

Streaming
---------
Writes are performed chunk-by-chunk via dask + rioxarray. Peak memory
is bounded by the source HDF5 chunk size (typically a few MB), not the
total raster size, so even multi-GB outputs stay tractable.

Examples
--------
  # Write all four default bands for a GUNW, alongside the .h5
  nisar_pytools to-geotiff /path/to/GUNW.h5

  # Just the unwrapped phase, HV polarization, into a separate folder
  nisar_pytools to-geotiff /path/to/GUNW.h5 \\
      --band unwrapped_phase --pol HV --output-dir /tmp/out

  # GSLC amplitude (dB) for the default pol on frequencyB
  nisar_pytools to-geotiff /path/to/GSLC.h5 --freq B

  # Crop a GUNW to a UTM bbox before writing (file's native CRS)
  nisar_pytools to-geotiff /path/to/GUNW.h5 --band coherence \\
      --bbox 400000 4800000 500000 4900000

  # Crop a full-res GSLC to a lat/lon AOI -- streamed, low memory
  nisar_pytools to-geotiff /path/to/GSLC.h5 \\
      --bbox-wgs84 -119.5 35.0 -118.5 36.0
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="nisar_pytools",
        description=_TOP_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # required=False so that running `nisar_pytools` with no args falls
    # through to print_help() in main() rather than erroring out.
    sub = parser.add_subparsers(dest="command", required=False, metavar="<command>")

    # to-geotiff subcommand
    p = sub.add_parser(
        "to-geotiff",
        help="Convert NISAR HDF5 (GUNW or GSLC) bands to GeoTIFF.",
        description=_TO_GEOTIFF_DESCRIPTION,
        epilog=_TO_GEOTIFF_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "h5_path",
        help="Path to a NISAR HDF5 file (GUNW or GSLC). Product type is auto-detected.",
    )
    p.add_argument(
        "--band",
        default=None,
        choices=sorted(set(_GSLC_BANDS) | set(_GUNW_BANDS)),
        metavar="BAND",
        help=(
            "Single band to extract. "
            "GSLC: amplitude. "
            "GUNW: unwrapped_phase | wrapped_phase | coherence | ionosphere. "
            "Default: write the full default-all set for the product type "
            "(see epilog below)."
        ),
    )
    p.add_argument(
        "--pol",
        default=None,
        choices=sorted(VALID_POLARIZATIONS),
        metavar="POL",
        help=(
            "Polarization to extract: HH, HV, VH, or VV. "
            "Default: the first polarization present in the file."
        ),
    )
    p.add_argument(
        "--freq",
        default="A",
        metavar="FREQ",
        help=(
            "NISAR frequency band: 'A' or 'B' (or the full 'frequencyA' / "
            "'frequencyB'). NISAR has two SAR frequencies; most products have "
            "a primary frequencyA grid. Default: A."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to write GeoTIFFs into (created if it doesn't exist). "
            "Default: the same directory as the input HDF5 file."
        ),
    )
    # Subsetting -- mutually exclusive bbox flags. Both crop the data
    # before the streaming write, so only the chunks intersecting the
    # bbox are loaded.
    bbox_group = p.add_mutually_exclusive_group()
    bbox_group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help=(
            "Crop output to (xmin, ymin, xmax, ymax) in the file's native "
            "CRS -- usually UTM meters for NISAR L2 products. No "
            "reprojection is performed; numbers must already be in the "
            "file's CRS. Streamed: only chunks intersecting the bbox are "
            "loaded. Mutually exclusive with --bbox-wgs84. See epilog."
        ),
    )
    bbox_group.add_argument(
        "--bbox-wgs84",
        nargs=4,
        type=float,
        default=None,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help=(
            "Crop output to (lon_min, lat_min, lon_max, lat_max) in WGS84 "
            "lat/lon. Internally reprojected to the file's native CRS "
            "(via pyproj) before slicing -- the resulting native bbox is "
            "the axis-aligned envelope of the four reprojected corners. "
            "Streamed. Mutually exclusive with --bbox. See epilog."
        ),
    )
    p.set_defaults(func=cmd_to_geotiff)

    # info subcommand -----------------------------------------------------
    p_info = sub.add_parser(
        "info",
        help="Print a summary of a NISAR HDF5 file (track/frame, times, pols, coherence stats).",
        description=_INFO_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_info.add_argument(
        "h5_path",
        help="Path to a NISAR HDF5 file (GUNW or GSLC). Product type is auto-detected.",
    )
    p_info.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of formatted text. Numeric stats stay as floats.",
    )
    p_info.set_defaults(func=cmd_info)

    # rslc-to-gunw subcommand --------------------------------------------
    p_r2g = sub.add_parser(
        "rslc-to-gunw",
        help="Run nisar.workflows.insar on an RSLC pair and produce a GUNW.",
        description=_RSLC_TO_GUNW_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_r2g.add_argument("reference_rslc", help="Path to the reference NISAR L1 RSLC HDF5 file.")
    p_r2g.add_argument("secondary_rslc", help="Path to the secondary NISAR L1 RSLC HDF5 file.")
    p_r2g.add_argument(
        "--output-dir", required=True,
        help="Directory for the output GUNW + scratch files. Created if missing.",
    )
    p_r2g.add_argument(
        "--runconfig",
        help="Optional YAML runconfig. If omitted, the bundled production-spec default "
             "is used and filled in with paths/bbox/EPSG at runtime.",
    )
    p_r2g.add_argument(
        "--dem",
        help="Optional DEM raster (TIF or VRT). If omitted, fetched via dem_stitcher.",
    )
    p_r2g.add_argument(
        "--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Output geocode bbox in projected (UTM) coordinates. "
             "If omitted, defaults to the reference RSLC's bounding polygon.",
    )
    p_r2g.add_argument(
        "--epsg", type=int,
        help="Output EPSG code. If omitted, picks the UTM zone of the scene centroid.",
    )
    p_r2g.add_argument(
        "--restart", action="store_true",
        help="Re-run all workflow steps even if their outputs exist "
             "(passed through to nisar.workflows.persistence).",
    )
    p_r2g.set_defaults(func=cmd_rslc_to_gunw)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the ``nisar_pytools`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()

    # No args at all -> print top-level help and exit cleanly.
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    # Subcommand was given but no handler attached (shouldn't normally
    # happen now that subparsers are not required, but guard anyway).
    if not getattr(args, "func", None):
        parser.print_help()
        return 0

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
