"""Shared test fixtures: synthetic NISAR HDF5 files."""

import h5py
import numpy as np
import pytest


def _create_frequency_group(parent, name, ny, nx, polarizations=("HH", "HV")):
    """Create a GSLC-style frequency group with coordinate scales and data."""
    grp = parent.create_group(name)

    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * 100.0 + 4000000.0

    xds = grp.create_dataset("xCoordinates", data=x)
    yds = grp.create_dataset("yCoordinates", data=y)
    xds.make_scale("xCoordinates")
    yds.make_scale("yCoordinates")

    for pol in polarizations:
        ds = grp.create_dataset(pol, shape=(ny, nx), dtype="c8", chunks=(4, 4))
        ds.attrs["grid_mapping"] = "projection"
        ds.attrs["description"] = f"Focused SLC image ({pol})"
        ds.dims[0].attach_scale(yds)
        ds.dims[1].attach_scale(xds)

    grp.create_dataset("mask", shape=(ny, nx), dtype="u1", chunks=(4, 4))
    grp.create_dataset("projection", data=np.uint32(32611))
    grp.create_dataset("centerFrequency", data=1.257e9)
    grp.create_dataset("xCoordinateSpacing", data=100.0)
    grp.create_dataset("yCoordinateSpacing", data=100.0)
    pol_arr = np.array([p.encode() for p in polarizations], dtype="S2")
    grp.create_dataset("listOfPolarizations", data=pol_arr)

    return grp


def _create_gunw_subproduct(parent, name, ny, nx, data_vars, pol="HH"):
    """Create a GUNW sub-product group (e.g., unwrappedInterferogram)."""
    sub = parent.create_group(name)

    # Sub-product level coords and mask
    x_sub = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y_sub = np.arange(ny, dtype="f8") * 100.0 + 4000000.0
    sub.create_dataset("xCoordinates", data=x_sub)
    sub.create_dataset("yCoordinates", data=y_sub)
    sub.create_dataset("mask", shape=(ny, nx), dtype="u1", chunks=(4, 4))
    sub.create_dataset("projection", data=np.uint32(32611))

    # Polarization group with its own coords and data
    pol_grp = sub.create_group(pol)
    x = np.arange(nx, dtype="f8") * 100.0 + 500000.0
    y = np.arange(ny, dtype="f8") * 100.0 + 4000000.0
    xds = pol_grp.create_dataset("xCoordinates", data=x)
    yds = pol_grp.create_dataset("yCoordinates", data=y)
    xds.make_scale("xCoordinates")
    yds.make_scale("yCoordinates")
    pol_grp.create_dataset("projection", data=np.uint32(32611))

    for var_name in data_vars:
        dtype = "c8" if "Interferogram" in var_name else "f4"
        ds = pol_grp.create_dataset(var_name, shape=(ny, nx), dtype=dtype, chunks=(4, 4))
        ds.attrs["grid_mapping"] = "projection"
        ds.dims[0].attach_scale(yds)
        ds.dims[1].attach_scale(xds)

    return sub


@pytest.fixture
def gslc_h5(tmp_path):
    """Create a minimal synthetic GSLC HDF5 file."""
    path = tmp_path / "test_gslc.h5"
    ny, nx = 8, 10
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "CF-1.7"
        f.attrs["mission_name"] = "NISAR"

        # Identification
        ident = f.create_group("science/LSAR/identification")
        ident.create_dataset("productType", data=b"GSLC")
        ident.create_dataset("trackNumber", data=np.uint32(77))
        ident.create_dataset("orbitPassDirection", data=b"Ascending")

        # Grids
        grids = f.create_group("science/LSAR/GSLC/grids")
        _create_frequency_group(grids, "frequencyA", ny, nx, ("HH", "HV"))
        _create_frequency_group(grids, "frequencyB", ny, nx // 2, ("HH",))

    return path


@pytest.fixture
def gunw_h5(tmp_path):
    """Create a minimal synthetic GUNW HDF5 file."""
    path = tmp_path / "test_gunw.h5"
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "CF-1.7"

        # Identification
        ident = f.create_group("science/LSAR/identification")
        ident.create_dataset("productType", data=b"GUNW")
        ident.create_dataset("trackNumber", data=np.uint32(149))

        # Grids / frequencyA
        base = f.create_group("science/LSAR/GUNW/grids/frequencyA")
        base.create_dataset("centerFrequency", data=1.257e9)

        # unwrappedInterferogram (6x8)
        _create_gunw_subproduct(
            base,
            "unwrappedInterferogram",
            ny=6,
            nx=8,
            data_vars=["unwrappedPhase", "coherenceMagnitude", "connectedComponents"],
        )

        # wrappedInterferogram (12x16 — different resolution)
        _create_gunw_subproduct(
            base,
            "wrappedInterferogram",
            ny=12,
            nx=16,
            data_vars=["wrappedInterferogram", "coherenceMagnitude"],
        )

        # pixelOffsets (6x8 — same resolution as unwrapped)
        _create_gunw_subproduct(
            base,
            "pixelOffsets",
            ny=6,
            nx=8,
            data_vars=["alongTrackOffset", "slantRangeOffset"],
        )

    return path
