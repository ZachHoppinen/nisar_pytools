"""ISCE3-backed RSLC -> GUNW processing wrappers.

The heavy lifting is done by :mod:`nisar.workflows.insar` from the
isce-framework/isce3 package. Because that package and its runtime
dependencies are heavy (fortran for pygrib, the GDAL HDF5 plugin,
snaphu, pyaps3, RAiDER, pysolid), they are an *optional* dep group of
``nisar_pytools``.

Recommended install (everything via conda-forge)::

    mamba install -c conda-forge \\
        isce3 libgdal-hdf5 snaphu-py pygrib pyaps3 raider pysolid netcdf4

PyPI-only path (works for the bits that are on PyPI; ``isce3`` and
``libgdal-hdf5`` still require conda)::

    mamba install -c conda-forge isce3 libgdal-hdf5
    pip install 'nisar-pytools[isce3]'

Public entry point:

* :func:`rslc_to_gunw` -- run the full NISAR L2 InSAR workflow on an RSLC
  pair and return the path to the produced GUNW.

A bundled production-spec runconfig (``isce3_runconfig_default.yaml``)
is filled in at runtime with the user's RSLC paths, DEM, AOI bbox, and
output EPSG. Override any block via the ``overrides`` argument or supply
your own runconfig via ``runconfig=``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_DEFAULT_RUNCONFIG = Path(__file__).parent / "isce3_runconfig_default.yaml"


def _load_default_runconfig() -> dict:
    with open(_DEFAULT_RUNCONFIG) as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``base`` (overrides win)."""
    result = dict(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _import_insar_workflow():
    """Import ``nisar.workflows.insar`` and friends with a friendly error if missing.

    Returns a tuple ``(insar, InsarRunConfig, Persistence, h5_prep)``.
    """
    try:
        from nisar.workflows import h5_prep, insar
        from nisar.workflows.insar_runconfig import InsarRunConfig
        from nisar.workflows.persistence import Persistence
    except ImportError as exc:
        raise ImportError(
            "nisar_pytools.processing.rslc_to_gunw requires the optional "
            "isce3 + nisar workflow stack. Recommended install (all conda-forge):\n"
            "  mamba install -c conda-forge isce3 libgdal-hdf5 snaphu-py "
            "pygrib pyaps3 raider pysolid netcdf4\n"
            "Or pip-installable bits via 'nisar-pytools[isce3]' (isce3 + "
            "libgdal-hdf5 still need conda).\n"
            f"\nOriginal import error: {exc}"
        ) from exc
    return insar, InsarRunConfig, Persistence, h5_prep


def _utm_epsg_from_polygon(poly) -> int:
    """Return the UTM EPSG code for the centroid of a WGS84 polygon."""
    lon = poly.centroid.x
    lat = poly.centroid.y
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _bbox_from_polygon_in_epsg(poly, epsg: int) -> tuple[float, float, float, float]:
    """Project a WGS84 polygon to ``epsg`` and return (xmin, ymin, xmax, ymax)."""
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tf.transform(*poly.exterior.coords.xy)
    import numpy as np
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))


def rslc_to_gunw(
    reference_rslc: str | os.PathLike,
    secondary_rslc: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    runconfig: str | os.PathLike | None = None,
    dem_file: str | os.PathLike | None = None,
    aoi_bbox_utm: tuple[float, float, float, float] | None = None,
    output_epsg: int | None = None,
    overrides: dict[str, Any] | None = None,
    restart: bool = False,
) -> Path:
    """Process a NISAR RSLC pair into a GUNW via ``nisar.workflows.insar``.

    Imports and calls ``nisar.workflows.insar.run`` in-process. Requires
    the optional ``[isce3]`` dep group plus a conda-installed ``isce3``
    and ``libgdal-hdf5``; see this module's docstring.

    Parameters
    ----------
    reference_rslc, secondary_rslc : path-like
        Reference and secondary L1 RSLC HDF5 files.
    output_dir : path-like
        Directory where the GUNW (``product.h5``) and a ``scratch/`` subdirectory
        will be written. Created if it doesn't exist.
    runconfig : path-like, optional
        YAML runconfig. If omitted, the bundled production-spec default
        (``isce3_runconfig_default.yaml``) is used and filled in with
        ``reference_rslc`` / ``secondary_rslc`` paths, ``dem_file``,
        ``aoi_bbox_utm`` / ``output_epsg`` (auto-detected if missing), and
        ``overrides``.
    dem_file : path-like, optional
        DEM raster (TIF or VRT). If omitted, a Copernicus 30 m DEM
        covering ``reference_rslc``'s bounding polygon is fetched via
        :func:`nisar_pytools.utils.dem.fetch_dem` and saved in ``output_dir``.
    aoi_bbox_utm : (xmin, ymin, xmax, ymax), optional
        Output geocode bbox in the projected CRS. If omitted, defaults to the
        full RSLC bounding polygon projected to ``output_epsg``.
    output_epsg : int, optional
        Output EPSG code. If omitted, picks the UTM zone from the RSLC scene
        centroid.
    overrides : dict, optional
        Nested dict of runconfig overrides; merged on top of the default
        runconfig with the standard "overrides win" rule.
    restart : bool, default False
        If True, re-run all workflow steps even if their outputs exist
        (passed through to ``nisar.workflows.persistence.Persistence``).

    Returns
    -------
    Path
        Path to the produced GUNW (``output_dir/product.h5``).

    Raises
    ------
    ImportError
        If ``nisar.workflows.insar`` cannot be imported.
    RuntimeError
        If the workflow fails or the expected output is missing.
    """
    reference_rslc = Path(reference_rslc).resolve()
    secondary_rslc = Path(secondary_rslc).resolve()
    output_dir = Path(output_dir).resolve()
    if not reference_rslc.exists():
        raise FileNotFoundError(reference_rslc)
    if not secondary_rslc.exists():
        raise FileNotFoundError(secondary_rslc)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

    # Import the workflow up-front so a missing-dep error happens before we
    # spend time fetching DEMs or building the runconfig.
    insar, InsarRunConfig, Persistence, h5_prep = _import_insar_workflow()

    # ---- Auto-fetch DEM if not supplied --------------------------------
    if dem_file is None:
        from nisar_pytools.utils.dem import fetch_dem

        dem_path = output_dir / "dem.tif"
        log.info("DEM not provided; fetching Copernicus 30 m DEM to %s", dem_path)
        fetch_dem(reference_rslc, out_path=dem_path)
        dem_file = dem_path
    dem_file = Path(dem_file).resolve()

    # ---- Auto-detect EPSG / bbox from the reference RSLC ---------------
    if output_epsg is None or aoi_bbox_utm is None:
        from nisar_pytools import open_nisar
        from nisar_pytools.utils.metadata import get_bounding_polygon

        dt = open_nisar(reference_rslc)
        poly = get_bounding_polygon(dt)
        if output_epsg is None:
            output_epsg = _utm_epsg_from_polygon(poly)
            log.info("Auto-detected output EPSG: %d", output_epsg)
        if aoi_bbox_utm is None:
            aoi_bbox_utm = _bbox_from_polygon_in_epsg(poly, output_epsg)
            log.info("Auto-detected AOI bbox (UTM): %s", aoi_bbox_utm)

    # ---- Load runconfig (default or user-supplied) ---------------------
    if runconfig is None:
        cfg = _load_default_runconfig()
    else:
        with open(runconfig) as f:
            cfg = yaml.safe_load(f)

    # ---- Inject paths / bbox / EPSG ------------------------------------
    g = cfg["runconfig"]["groups"]
    g["input_file_group"]["reference_rslc_file"] = str(reference_rslc)
    g["input_file_group"]["secondary_rslc_file"] = str(secondary_rslc)
    g["input_file_group"]["qa_gunw_input_file"] = str(output_dir / "product.h5")
    g["dynamic_ancillary_file_group"]["dem_file"] = str(dem_file)
    g["product_path_group"]["product_path"] = str(output_dir)
    g["product_path_group"]["scratch_path"] = str(scratch_dir)
    g["product_path_group"]["sas_output_file"] = str(output_dir / "product.h5")
    g["product_path_group"]["qa_output_dir"] = str(output_dir / "qa_insar")
    g["logging"]["path"] = str(scratch_dir / "insar.log")

    xmin, ymin, xmax, ymax = aoi_bbox_utm
    g["processing"]["geocode"]["output_epsg"] = int(output_epsg)
    g["processing"]["geocode"]["top_left"] = {"x_abs": float(xmin), "y_abs": float(ymax)}
    g["processing"]["geocode"]["bottom_right"] = {"x_abs": float(xmax), "y_abs": float(ymin)}
    g["processing"]["radar_grid_cubes"]["output_epsg"] = int(output_epsg)

    # ---- Apply user overrides ------------------------------------------
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    # ---- Write merged runconfig ----------------------------------------
    runconfig_path = output_dir / "runconfig.yaml"
    with open(runconfig_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log.info("Runconfig written to %s", runconfig_path)

    # ---- Run the InSAR workflow in-process -----------------------------
    log.info("Running nisar.workflows.insar (this can take 1-2 hours)...")
    args_ns = argparse.Namespace(
        run_config_path=str(runconfig_path),
        log_file=False,
        restart=restart,
    )
    insar_runcfg = InsarRunConfig(args_ns)
    persist = Persistence(insar_runcfg.cfg["logging"]["path"], restart=restart)
    if persist.run:
        _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)
        insar.run(insar_runcfg.cfg, out_paths, persist.run_steps)

    gunw_path = output_dir / "product.h5"
    if not gunw_path.exists():
        raise RuntimeError(
            f"Workflow finished without error but expected GUNW not found at {gunw_path}. "
            f"See log: {scratch_dir / 'insar.log'}"
        )
    log.info("GUNW written: %s", gunw_path)
    return gunw_path
