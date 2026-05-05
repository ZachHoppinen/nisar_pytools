"""Command-line interface for nisar_pytools.

Single binary ``nisar_pytools`` with subcommands.

Currently supported:
    nisar_pytools to-geotiff <h5> [--band ...] [--pol ...] [--freq ...]
                                  [--output-dir ...]

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
import logging
import sys
import threading
from pathlib import Path

import dask.array as dask_array
import numpy as np
import xarray as xr

from nisar_pytools import open_nisar
from nisar_pytools.utils.conversion import to_db
from nisar_pytools.utils.metadata import get_product_type, get_slc
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
        # Reuse the package's reproject_bbox helper so corner-handling
        # behavior is consistent across the codebase.
        return reproject_bbox(
            tuple(args.bbox_wgs84),
            src_crs=4326,
            dst_crs=target_crs.to_epsg(),
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


_TOP_DESCRIPTION = """\
Command-line tools for working with NISAR HDF5 products.

Subcommands:
  to-geotiff   Convert GUNW / GSLC HDF5 bands into GeoTIFFs.

Run `nisar_pytools <subcommand> --help` for subcommand-specific help.
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
