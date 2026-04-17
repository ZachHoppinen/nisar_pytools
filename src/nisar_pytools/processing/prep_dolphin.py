"""Prepare NISAR GSLC HDF5 files for dolphin InSAR time-series processing.

dolphin is an open-source InSAR time-series analysis tool from the
OPERA/ISCE framework that performs phase linking, unwrapping (via SNAPHU
or spurt 3D EMCF), and network inversion on SLC stacks.

    https://github.com/isce-framework/dolphin

This module crops GSLCs to an optional WGS84 AOI, exports them as complex
GeoTIFFs named by acquisition date, and generates a dolphin configuration
YAML with sensible defaults and user-specified overrides.

dolphin is an optional dependency -- only required for config generation,
not for SLC export. Install with: ``pip install dolphin-sar``
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from nisar_pytools.io.reader import open_nisar
from nisar_pytools.utils.filename import parse_filename
from nisar_pytools.utils.metadata import get_slc
from nisar_pytools.utils.overlap import reproject_bbox
from nisar_pytools.utils.search_validation import validate_aoi
from nisar_pytools.utils.validation import (
    validate_frequency,
    validate_nisar_hdf5,
    validate_polarization,
)

# Cache for EPSG lookups so we don't re-open the DataTree for each file.
_epsg_cache: dict[str, int] = {}

log = logging.getLogger(__name__)


def _get_epsg(
    h5_path: Path,
    frequency: str = "frequencyA",
    polarization: str = "HH",
) -> int:
    """Get the EPSG code from a GSLC via open_nisar + get_slc."""
    key = str(h5_path)
    if key not in _epsg_cache:
        dt = open_nisar(h5_path, chunks=None)
        da = get_slc(dt, polarization=polarization, frequency=frequency)
        _epsg_cache[key] = da.rio.crs.to_epsg()
    return _epsg_cache[key]


def _set_nested(d: dict, path: list[str], value: object) -> None:
    """Set a value in a nested dict, creating intermediate dicts as needed."""
    cur = d
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def crop_gslc_to_tif(
    h5_path: str | Path,
    out_tif: str | Path,
    bbox_utm: tuple[float, float, float, float] | None = None,
    frequency: str = "frequencyA",
    polarization: str = "HH",
) -> Path:
    """Read a polarization channel from a GSLC HDF5 and write a complex GeoTIFF.

    Uses :func:`~nisar_pytools.io.reader.open_nisar` and
    :func:`~nisar_pytools.utils.metadata.get_slc` to load the data with
    CRS already set, then optionally crops to a UTM bounding box.

    Parameters
    ----------
    h5_path : str or Path
        Input GSLC HDF5 file.
    out_tif : str or Path
        Output GeoTIFF path.
    bbox_utm : tuple of float or None
        (xmin, ymin, xmax, ymax) in the GSLC's native UTM CRS.
        If None, the full grid is exported.
    frequency : str
        Frequency group (default "frequencyA").
    polarization : str
        Polarization channel (default "HH").

    Returns
    -------
    Path
        The output GeoTIFF path.
    """
    out_tif = Path(out_tif)
    dt = open_nisar(h5_path, chunks=None)
    da = get_slc(dt, polarization=polarization, frequency=frequency)

    if bbox_utm is not None:
        xmin, ymin, xmax, ymax = bbox_utm
        full_shape = da.shape
        # Handle ascending or descending y coordinates.
        if da.y[0] < da.y[-1]:
            y_slice = slice(ymin, ymax)
        else:
            y_slice = slice(ymax, ymin)
        da = da.sel(x=slice(xmin, xmax), y=y_slice)
        if da.size == 0:
            raise ValueError(
                f"AOI bbox {bbox_utm} has no overlap with GSLC grid"
            )
        log.debug("Cropped %s -> %s", full_shape, da.shape)

    da = da.compute().astype(np.complex64)
    da.rio.to_raster(out_tif, dtype="complex64")
    log.info("Wrote %s  shape %s", out_tif.name, da.shape)
    return out_tif


def _generate_dolphin_config(
    slc_tifs: list[Path],
    work_dir: Path,
    config_path: Path,
    overrides: list[tuple[list[str], object]] | None = None,
) -> Path:
    """Generate a dolphin config YAML from SLC file list, then apply overrides.

    Parameters
    ----------
    slc_tifs : list of Path
        Paths to the SLC GeoTIFF files.
    work_dir : Path
        Dolphin working directory.
    config_path : Path
        Where to write the config YAML.
    overrides : list of (path, value) tuples or None
        Each entry is (yaml_key_path, value) where yaml_key_path is a list
        of strings identifying the nested key.

    Returns
    -------
    Path
        Path to the generated config file.

    Raises
    ------
    FileNotFoundError
        If dolphin CLI is not available.
    """
    if config_path.exists():
        config_path.unlink()

    cmd = [
        "dolphin", "config",
        "--slc-files", *[str(p) for p in slc_tifs],
        "--work-directory", str(work_dir),
        "--outfile", str(config_path),
    ]
    log.info("Running: %s", " ".join(cmd[:4]) + " ...")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            "dolphin CLI not found. Install with: pip install dolphin-sar"
        )

    if not overrides:
        return config_path

    # Patch the YAML with overrides. Prefer ruamel.yaml (preserves
    # comments/formatting from dolphin's output), fall back to pyyaml.
    try:
        from ruamel.yaml import YAML
        yaml_eng = YAML()
        yaml_eng.preserve_quotes = True
        cfg = yaml_eng.load(config_path.read_text())
        for path, value in overrides:
            _set_nested(cfg, path, value)
        with config_path.open("w") as f:
            yaml_eng.dump(cfg, f)
    except ImportError:
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        for path, value in overrides:
            _set_nested(cfg, path, value)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    log.info("Wrote dolphin config with %d overrides: %s",
             len(overrides), config_path)
    return config_path


def prep_dolphin(
    gslc_paths: Sequence[str | Path],
    out_dir: str | Path,
    aoi_wgs84: tuple[float, float, float, float] | None = None,
    skip_dates: set[str] | None = None,
    frequency: str = "frequencyA",
    polarization: str = "HH",
    dolphin_overrides: list[tuple[list[str], object]] | None = None,
) -> Path:
    """Prepare NISAR GSLCs for dolphin processing.

    Exports each GSLC as a complex GeoTIFF (optionally cropped to an AOI),
    then generates a dolphin configuration YAML.

    Parameters
    ----------
    gslc_paths : sequence of str or Path
        Paths to NISAR GSLC HDF5 files.
    out_dir : str or Path
        Output directory. SLC TIFs are written to ``out_dir/slcs/`` and
        the config to ``out_dir/dolphin_config.yaml``.
    aoi_wgs84 : tuple of float or None
        (lon_min, lat_min, lon_max, lat_max) in WGS84. If None, the full
        GSLC extent is used.
    skip_dates : set of str or None
        Dates as "YYYYMMDD" strings to exclude from processing.
    frequency : str
        GSLC frequency group (default "frequencyA").
    polarization : str
        Polarization channel (default "HH").
    dolphin_overrides : list of (path, value) tuples or None
        Overrides to patch into the dolphin config YAML. Each entry is
        ``([key1, key2, ...], value)``.

    Returns
    -------
    Path
        Path to the generated ``dolphin_config.yaml``.

    Raises
    ------
    FileNotFoundError
        If dolphin CLI is not installed.
    ValueError
        If no valid GSLC files remain after filtering.
    """
    # --- Validate inputs ---
    gslc_paths = sorted(Path(p) for p in gslc_paths)
    if not gslc_paths:
        raise ValueError("No GSLC paths provided")

    if aoi_wgs84 is not None:
        validate_aoi(aoi_wgs84)

    skip_dates = skip_dates or set()
    for d in skip_dates:
        if not re.fullmatch(r"\d{8}", d):
            raise ValueError(
                f"skip_dates entries must be 'YYYYMMDD' strings, got '{d}'"
            )

    validate_frequency(frequency)
    validate_polarization(polarization)

    # --- Setup ---
    out_dir = Path(out_dir)
    slc_dir = out_dir / "slcs"
    slc_dir.mkdir(parents=True, exist_ok=True)

    bbox_utm = None
    if aoi_wgs84 is not None:
        epsg = _get_epsg(gslc_paths[0], frequency, polarization)
        bbox_utm = reproject_bbox(aoi_wgs84, src_crs=4326, dst_crs=epsg)
        log.info("AOI WGS84 %s -> UTM EPSG:%d bbox %s", aoi_wgs84, epsg, bbox_utm)

    # Export each GSLC as a dated complex GeoTIFF.
    slc_tifs: list[Path] = []
    for fp in gslc_paths:
        # Validate each GSLC file.
        try:
            h5f = validate_nisar_hdf5(fp)
            h5f.close()
        except (FileNotFoundError, ValueError) as e:
            log.warning("Skipping %s: %s", fp.name, e)
            continue

        info = parse_filename(fp.name)
        date_str = info.start_time.strftime("%Y%m%d")

        if date_str in skip_dates:
            log.info("Skipping date %s (in skip_dates)", date_str)
            continue

        out_tif = slc_dir / f"{date_str}.tif"
        if out_tif.exists() and out_tif.stat().st_size > 0:
            log.info("Exists, skipping: %s", out_tif.name)
            slc_tifs.append(out_tif)
            continue

        crop_gslc_to_tif(fp, out_tif, bbox_utm, frequency, polarization)
        slc_tifs.append(out_tif)

    if not slc_tifs:
        raise ValueError("No valid GSLC files after filtering")

    log.info("Exported %d SLC GeoTIFFs to %s", len(slc_tifs), slc_dir)

    # Generate dolphin config.
    config_path = out_dir / "dolphin_config.yaml"
    _generate_dolphin_config(slc_tifs, out_dir, config_path, dolphin_overrides)

    log.info("Ready. Review the config, then run:\n"
             "  dolphin run %s", config_path)

    return config_path
