"""Crop a NISAR L1 RSLC to an AOI without downloading the whole granule.

Motivation
----------
An RSLC granule is 9-26 GB, and the isce3 InSAR workflow only ever consumes a
small radar window of it (see :mod:`nisar_pytools.processing.crop_rslc`). The
pol images are chunked ``(512, 512)`` and gzipped, so HDF5 can read that window
over HTTP byte-range and leave the rest on the server.

Measured on the 8.8 GB provisional granule
``NISAR_L1_PR_RSLC_026_092_A_034_2005_QPDH_A_20260726T135041_...``: opening the
remote file costs 2 s and 34 MB, and a 2048x4096 window of one polarization
costs 9 s and 201 MB, i.e. 2.3% of the granule. Expect roughly 4x that for a
quad-pol crop, still around a tenth of a full download.

Why a skeleton is needed
------------------------
The window solve runs through ``nisar.products.readers.SLC``, which opens the
product *by path* -- it cannot be handed a remote handle. So the flow is:

1. Stream a metadata-only *skeleton*: the whole product structure with the pol
   images recreated empty. Everything the readers need is metadata, which
   already arrives inside the file-open read, so this is nearly free.
2. Solve the radar window against that tiny local file.
3. Reopen the remote product and crop the window straight out of it, which
   transfers only the overlapping image chunks.

The result is identical to cropping a downloaded granule.

Byte-range caveat
-----------------
The remote file is block-cached, and the default block is much larger than one
chunk, so a windowed read pulls roughly 3x the bytes it returns. Passing a
smaller ``block_size`` trades that amplification against more requests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py

from nisar_pytools.io.stream_l0b import _ensure_auth

log = logging.getLogger(__name__)

#: collections holding L1 RSLC products, newest maturity first
RSLC_SHORT_NAMES = (
    "NISAR_L1_RSLC_PROVISIONAL_V1",
    "NISAR_L1_RSLC_BETA_V1",
)

_SWATHS = "science/LSAR/RSLC/swaths"


def open_remote_rslc(
    granule: str,
    short_names: tuple[str, ...] = RSLC_SHORT_NAMES,
    auth_strategy: str = "netrc",
    block_size: int | None = None,
) -> h5py.File:
    """Open a remote RSLC over byte-range as an :class:`h5py.File`.

    No data is transferred beyond what later reads touch. Logs in to Earthdata
    if the caller has not already. The caller is responsible for closing the
    handle.

    Parameters
    ----------
    granule : str
        Granule name, with or without the ``.h5`` suffix.
    short_names : tuple of str
        Collections to search, in order.
    auth_strategy : str
        Passed to ``earthaccess.login`` when no session exists.
    block_size : int, optional
        Byte-range block size. Smaller blocks cut the read amplification on
        scattered chunks at the cost of more requests. ``None`` keeps the
        default.

    Raises
    ------
    ValueError
        If the granule is not found in any collection.
    """
    import earthaccess

    _ensure_auth(auth_strategy)
    name = granule[:-3] if granule.endswith(".h5") else granule
    for short_name in short_names:
        results = earthaccess.search_data(short_name=short_name, readable_granule_name=name)
        if results:
            break
    else:
        raise ValueError(f"No NISAR RSLC granule found for {name} in {short_names}")

    kwargs = {} if block_size is None else {"block_size": block_size}
    fileobj = earthaccess.open(results[:1], **kwargs)[0]
    return h5py.File(fileobj, "r", driver="fileobj")


def write_skeleton(src: h5py.File, skeleton_h5: str | Path) -> Path:
    """Write a metadata-only copy of an RSLC: full structure, empty pol images.

    Stands in for the real product wherever a path-based reader only needs
    metadata. The images are recreated with the right shape and dtype but no
    pixels, so the file stays tiny even for a 26 GB granule.

    Parameters
    ----------
    src : h5py.File
        Open source RSLC, typically a remote handle.
    skeleton_h5 : path-like
        Where to write the skeleton.

    Returns
    -------
    pathlib.Path
    """
    from nisar_pytools.processing.crop_rslc import _polarizations

    skeleton_h5 = Path(skeleton_h5)

    # Only the polarization images are emptied; every other dataset, including
    # any complex-valued calibration table, is copied so the readers still work.
    pol_paths = set()
    for name in src[_SWATHS]:
        if not name.startswith("frequency"):
            continue
        grp = src[f"{_SWATHS}/{name}"]
        pol_paths.update(f"{_SWATHS}/{name}/{pol}" for pol in _polarizations(grp))

    with h5py.File(skeleton_h5, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        def recurse(src_grp, dst_grp):
            for k, v in src_grp.attrs.items():
                dst_grp.attrs[k] = v
            for key, item in src_grp.items():
                if isinstance(item, h5py.Group):
                    recurse(item, dst_grp.create_group(key))
                elif item.name.lstrip("/") in pol_paths:
                    d = dst_grp.create_dataset(key, shape=item.shape, dtype=item.dtype)
                    for ak, av in item.attrs.items():
                        d.attrs[ak] = av
                else:
                    src_grp.copy(key, dst_grp, name=key)

        recurse(src, dst)

    log.info("Wrote RSLC skeleton: %s", skeleton_h5)
    return skeleton_h5


def crop_streamed(
    granule: str,
    bbox_utm,
    epsg_aoi,
    dem_file,
    out_dir: str | Path = ".",
    margin: int = 512,
    min_size: int = 2048,
    block_size: int | None = None,
) -> Path:
    """Crop a remote RSLC to ``bbox_utm`` without downloading the granule.

    Streams a skeleton, solves the radar window against it, then reads only the
    windowed image chunks out of the remote product. The remote handle is opened
    twice rather than held across the window solve, which does DEM work.

    Parameters
    ----------
    granule : str
        Granule name, with or without the ``.h5`` suffix.
    bbox_utm : sequence of float
        AOI as ``(xmin, ymin, xmax, ymax)`` in ``epsg_aoi`` coordinates.
    epsg_aoi : int
        EPSG code the bounding box is given in.
    dem_file : path-like
        DEM raster used for the corner heights in the geo2rdr solve.
    out_dir : path-like
        Directory for the skeleton and the cropped output.
    margin, min_size : int
        Passed through to
        :func:`nisar_pytools.processing.crop_rslc.radar_window_for_aoi`.
    block_size : int, optional
        Byte-range block size, see :func:`open_remote_rslc`.

    Returns
    -------
    pathlib.Path
        Path of the cropped RSLC.
    """
    from nisar_pytools.processing.crop_rslc import (
        crop_rslc_from_handle,
        radar_window_for_aoi,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = granule[:-3] if granule.endswith(".h5") else granule

    skeleton = out_dir / f"{name}_skeleton.h5"
    with open_remote_rslc(granule, block_size=block_size) as remote:
        write_skeleton(remote, skeleton)

    window = radar_window_for_aoi(skeleton, bbox_utm, epsg_aoi, dem_file, margin, min_size)

    dst = out_dir / f"{name}_sub.h5"
    with open_remote_rslc(granule, block_size=block_size) as remote:
        crop_rslc_from_handle(remote, dst, window)
    return dst
