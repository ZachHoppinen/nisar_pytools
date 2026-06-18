"""Crop NISAR L1 RSLC products to a small radar-coordinate window.

Motivation
----------
The isce3 ``nisar.workflows.insar`` pipeline runs every step before geocoding
(rdr2geo, geo2rdr, resample, dense_offsets, rubbersheet, crossmul, unwrap,
ionosphere) in *radar* geometry over the FULL RSLC swath. A geocode ``--bbox``
only crops the final geocoded output, so it does not reduce that cost -- a full
NISAR frame is multi-hour on CPU regardless of how small the output AOI is.

To make an all-steps run cheap (minutes) the *inputs* must be small. This module
maps an AOI bounding box into each RSLC's radar grid and writes a cropped RSLC
containing only that (azimuth line, range sample) window, so every downstream
step operates on a tiny patch.

How the radar window is found
-----------------------------
We project the four AOI corners into the RSLC's radar grid with a true isce3
``geo2rdr`` solve: each corner is reprojected to lon/lat, given its DEM height,
and inverted against the product's orbit to a (zero-Doppler azimuth time, slant
range). NISAR RSLCs are zero-Doppler processed, so the solve uses a Doppler of
zero (a non-zero Doppler centroid would yield native-Doppler times offset by
~fdc/FM-rate, i.e. seconds / thousands of lines off). The bounding box of the
four (time, range) solutions is mapped onto the full-resolution swath axes to
get integer line/pixel bounds, then padded to absorb coregistration search,
filter kernels, terrain-height variation across the AOI, and ref/secondary
misregistration. The crop itself is a plain integer slice -- no resampling.

What gets cropped vs copied
---------------------------
Cropped (azimuth- or range-indexed):
  - swaths/zeroDopplerTime                       (azimuth axis, shared by A & B)
  - swaths/frequency{A,B}/{pol}                  (the SLC images)
  - swaths/frequency{A,B}/slantRange             (range axis, per frequency)
  - swaths/frequency{A,B}/validSamplesSubSwath1  (rows subset; values shifted)
Copied verbatim:
  - everything else (orbit, attitude, doppler, geolocationGrid, calibration,
    identification, ...). These are coordinate-indexed lookups that the workflow
    interpolates; keeping the full-scene tables is valid because the cropped
    radar grid is a subset of their domain.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import rasterio
from pyproj import Transformer

log = logging.getLogger(__name__)

# HDF5 paths (relative to the file root). LSAR is the only band in these products.
_SWATHS = "science/LSAR/RSLC/swaths"


def _pad_axis(i0, i1, axis_len, margin, min_size):
    """Expand the half-open index window ``[i0, i1)`` by ``margin`` samples on
    each side, widening further if needed so the window spans at least
    ``min_size`` samples. Clamped to ``[0, axis_len)`` (so a window touching the
    swath edge may fall short of ``min_size`` -- there is no more data)."""
    need = (min_size - (i1 - i0) + 1) // 2  # half-pad to reach min_size (ceil)
    pad = max(margin, need)
    return int(max(0, i0 - pad)), int(min(axis_len, i1 + pad))


def _window_from_radar_coords(swath_t, slant_a, slant_b, a_lo, a_hi, r_lo, r_hi,
                              margin, min_size=0):
    """Convert an (azimuth-time, slant-range) extent into padded integer
    line/pixel windows.

    ``swath_t`` / ``slant_a`` / ``slant_b`` are the full-resolution
    zeroDopplerTime, frequency-A slantRange and frequency-B slantRange axis
    vectors (all monotonically increasing). ``margin`` pads each side;
    ``min_size`` forces each axis to span at least that many samples -- the
    floor that protects small AOIs, where ``margin`` alone would leave the
    patch below the processing filter footprint. Pure index math -- no isce3 --
    so it is unit-testable in isolation. Returns the standard window dict.
    """
    if a_hi < swath_t[0] or a_lo > swath_t[-1] or r_hi < slant_a[0] or r_lo > slant_a[-1]:
        raise ValueError(
            "AOI radar extent (az %.3f-%.3f s, range %.0f-%.0f m) does not "
            "overlap the RSLC swath." % (a_lo, a_hi, r_lo, r_hi)
        )
    a0, a1 = np.searchsorted(swath_t, [a_lo, a_hi])
    p0, p1 = np.searchsorted(slant_a, [r_lo, r_hi])
    # Pad (absorbs coregistration search, filter kernels, terrain height
    # variation across the AOI, ref/secondary misregistration), then floor to
    # min_size so tiny AOIs still get a patch big enough for those filters.
    A0, A1 = _pad_axis(a0, a1, len(swath_t), margin, min_size)
    P0, P1 = _pad_axis(p0, p1, len(slant_a), margin, min_size)
    # frequencyB shares the azimuth window; its range window covers the same
    # physical slant-range span as the padded frequencyA window.
    P0b = int(np.clip(np.searchsorted(slant_b, slant_a[P0]), 0, len(slant_b)))
    P1b = int(np.clip(np.searchsorted(slant_b, slant_a[P1 - 1]) + 1, 0, len(slant_b)))
    return {"az": (A0, A1), "A": (P0, P1), "B": (P0b, P1b)}


def radar_window_for_aoi(h5_path, bbox_utm, epsg_aoi, dem_file, margin=512,
                         min_size=2048):
    """Find the radar-coordinate crop window for ``bbox_utm`` in one RSLC.

    Projects the four AOI corners into the radar grid with isce3 ``geo2rdr``
    (zero-Doppler, since NISAR RSLC is zero-Doppler processed) at their DEM
    heights, then maps the bounding (time, range) extent onto the swath axes.

    Returns a dict::

        {"az": (a0, a1),          # azimuth line slice (shared by both freqs)
         "A":  (p0, p1),          # frequencyA range sample slice
         "B":  (p0, p1)}          # frequencyB range sample slice

    ``margin`` is the padding in *frequencyA* samples / azimuth lines added on
    every side (frequencyB inherits the same physical range extent).
    ``min_size`` is the minimum window span per axis; it only binds for small
    AOIs (for large ones ``margin`` dominates) and keeps the patch above the
    processing filter footprint so tiny AOIs don't degrade.

    Requires isce3 + the ``[isce3]`` dep group (imported lazily here).
    """
    import isce3
    from nisar.products.readers import SLC

    xmin, ymin, xmax, ymax = bbox_utm
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    to_lonlat = Transformer.from_crs(epsg_aoi, 4326, always_xy=True)

    slc = SLC(hdf5file=str(h5_path))
    radar_grid = slc.getRadarGrid("A")
    orbit = slc.getOrbit()
    ellipsoid = isce3.core.Ellipsoid()
    zero_doppler = isce3.core.LUT2d()  # empty LUT == Doppler 0 (zero-Doppler RSLC)

    # geo2rdr each AOI corner at its terrain height -> (aztime, slant range).
    az_times, slant_ranges = [], []
    with rasterio.open(dem_file) as dem:
        for x, y in corners:
            lon, lat = to_lonlat.transform(x, y)
            height = float(next(dem.sample([(lon, lat)]))[0])
            llh = np.array([np.radians(lon), np.radians(lat), height])
            aztime, srange = isce3.geometry.geo2rdr(
                llh, ellipsoid, orbit, zero_doppler,
                radar_grid.wavelength, radar_grid.lookside,
            )
            az_times.append(aztime)
            slant_ranges.append(srange)

    with h5py.File(h5_path, "r") as f:
        swath_t = f[f"{_SWATHS}/zeroDopplerTime"][()]
        slant_a = f[f"{_SWATHS}/frequencyA/slantRange"][()]
        slant_b = f[f"{_SWATHS}/frequencyB/slantRange"][()]

    # geo2rdr aztime is referenced to the orbit epoch; the swath time axis must
    # share it (true for NISAR -- both are seconds of the acquisition day).
    # Guard against a silent epoch mismatch that would offset every line.
    if abs(radar_grid.sensing_start - swath_t[0]) > 1.0:
        raise RuntimeError(
            "RSLC radar-grid epoch does not match the swath zeroDopplerTime "
            "axis; geo2rdr azimuth times would be offset."
        )

    win = _window_from_radar_coords(
        swath_t, slant_a, slant_b,
        min(az_times), max(az_times), min(slant_ranges), max(slant_ranges),
        margin, min_size,
    )
    a0, a1 = win["az"]
    log.info("Radar window %s: az %d lines, A %d px, B %d px",
             Path(h5_path).name, a1 - a0, win["A"][1] - win["A"][0],
             win["B"][1] - win["B"][0])
    return win


def _copy_cropped(src_ds, dst_grp, name, sl):
    """Create ``name`` in ``dst_grp`` from ``src_ds`` sliced by ``sl`` (a tuple
    of slices), reading only the requested hyperslab from disk, then copy the
    source dataset's attributes onto the new dataset."""
    data = src_ds[sl]
    d = dst_grp.create_dataset(name, data=data)
    for k, v in src_ds.attrs.items():
        d.attrs[k] = v
    return d


def crop_rslc(src_h5, dst_h5, window):
    """Write a cropped copy of ``src_h5`` to ``dst_h5`` using ``window`` from
    :func:`radar_window_for_aoi`. Datasets on the azimuth/range axes are sliced;
    everything else is copied verbatim."""
    src_h5, dst_h5 = Path(src_h5), Path(dst_h5)
    a0, a1 = window["az"]
    # Map each frequency to its (p0, p1) range window.
    rng = {"frequencyA": window["A"], "frequencyB": window["B"]}

    # The exact set of swath datasets we crop (others are copied as-is).
    az_only = {f"{_SWATHS}/zeroDopplerTime"}                       # 1D azimuth axis
    img_or_valid = {}   # full path -> (slice, value_offset_for_validsamples)
    range_axis = {}     # full path -> (p0, p1)
    for fr, (p0, p1) in rng.items():
        for pol in ("HH", "HV"):
            img_or_valid[f"{_SWATHS}/{fr}/{pol}"] = ((slice(a0, a1), slice(p0, p1)), None)
        # validSamplesSubSwath1: subset rows, and shift the stored column
        # indices by the range offset (they index into the range axis).
        img_or_valid[f"{_SWATHS}/{fr}/validSamplesSubSwath1"] = ((slice(a0, a1), slice(None)), p0)
        range_axis[f"{_SWATHS}/{fr}/slantRange"] = (p0, p1)

    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        # Copy root attributes.
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        def recurse(src_grp, dst_grp):
            for k, v in src_grp.attrs.items():
                dst_grp.attrs[k] = v
            for key, item in src_grp.items():
                path = item.name.lstrip("/")
                if isinstance(item, h5py.Group):
                    recurse(item, dst_grp.create_group(key))
                elif path in az_only:
                    _copy_cropped(item, dst_grp, key, (slice(a0, a1),))
                elif path in range_axis:
                    p0, p1 = range_axis[path]
                    _copy_cropped(item, dst_grp, key, (slice(p0, p1),))
                elif path in img_or_valid:
                    sl, offset = img_or_valid[path]
                    if offset is None:
                        _copy_cropped(item, dst_grp, key, sl)
                    else:
                        # validSamples: shift indices into the cropped range axis
                        # and clip to the new sample range.
                        p0, p1 = rng["frequencyA" if "frequencyA" in path else "frequencyB"]
                        vals = item[sl].astype(np.int64) - offset
                        vals = np.clip(vals, 0, p1 - p0)
                        d = dst_grp.create_dataset(key, data=vals.astype(item.dtype))
                        for ak, av in item.attrs.items():
                            d.attrs[ak] = av
                else:
                    # Not on a cropped axis -> HDF5-level copy (preserves attrs,
                    # dtype, fill value; never loads big arrays into Python).
                    src_grp.copy(key, dst_grp, name=key)

        recurse(src, dst)
    log.info("Wrote cropped RSLC: %s", dst_h5)
    return dst_h5


def crop_rslc_pair(reference_rslc, secondary_rslc, bbox_utm, epsg_aoi,
                   dem_file, out_dir, margin=512, min_size=2048):
    """Crop a reference/secondary RSLC pair to the AOI radar window.

    Each file's window is computed from *its own* geo2rdr solve (the AOI maps to
    different lines/pixels in each acquisition); ``margin`` absorbs the
    ref/secondary misregistration and ``min_size`` floors the patch size for
    small AOIs. Returns ``(ref_sub_path, sec_sub_path)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for tag, src in (("ref", reference_rslc), ("sec", secondary_rslc)):
        win = radar_window_for_aoi(src, bbox_utm, epsg_aoi, dem_file, margin, min_size)
        dst = out_dir / f"{Path(src).stem}_sub.h5"
        crop_rslc(src, dst, win)
        results.append(dst)
    return tuple(results)
