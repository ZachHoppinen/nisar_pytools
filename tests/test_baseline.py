"""Tests for nisar_pytools.processing.baseline."""

import h5py
import numpy as np
import xarray as xr

from nisar_pytools import open_nisar
from nisar_pytools.processing.baseline import compute_baseline


def _make_gslc_with_orbit(path, time_offset=0.0, position_offset=None):
    """Create a minimal GSLC with orbit and radarGrid data."""
    if position_offset is None:
        position_offset = np.array([0.0, 0.0, 0.0])

    ny_rg, nx_rg = 6, 8
    n_heights = 3

    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "CF-1.7"

        # Identification
        ident = f.create_group("science/LSAR/identification")
        ident.create_dataset("productType", data=b"GSLC")
        ident.create_dataset("trackNumber", data=np.uint32(77))
        ident.create_dataset("frameNumber", data=np.uint16(24))
        ident.create_dataset("orbitPassDirection", data=b"Ascending")
        ident.create_dataset("absoluteOrbitNumber", data=np.uint32(1387))
        ident.create_dataset("zeroDopplerStartTime", data=b"2025-11-03T12:46:15.000000000")

        # Orbit
        n_epochs = 20
        orbit = f.create_group("science/LSAR/GSLC/metadata/orbit")
        times = np.arange(n_epochs) * 10.0 + 45000.0 + time_offset
        # Simplified circular-ish orbit
        base_pos = np.column_stack([
            -1300000 + np.arange(n_epochs) * -1200.0,
            -6800000 + np.arange(n_epochs) * 2000.0,
            1700000 + np.arange(n_epochs) * 7000.0,
        ])
        pos = base_pos + position_offset
        vel = np.column_stack([
            np.full(n_epochs, -1250.0),
            np.full(n_epochs, 2050.0),
            np.full(n_epochs, 7170.0),
        ])
        orbit.create_dataset("position", data=pos)
        orbit.create_dataset("velocity", data=vel)
        orbit.create_dataset("time", data=times)
        orbit.create_dataset("interpMethod", data=b"Hermite")
        orbit.create_dataset("orbitType", data=b"POE")

        # radarGrid
        rg = f.create_group("science/LSAR/GSLC/metadata/radarGrid")
        x_rg = np.arange(nx_rg, dtype="f8") * 1000.0 + 500000.0
        y_rg = np.arange(ny_rg, dtype="f8") * -1000.0 + 4500000.0
        heights = np.array([0, 1500, 3000], dtype="f8")

        xds = rg.create_dataset("xCoordinates", data=x_rg)
        yds = rg.create_dataset("yCoordinates", data=y_rg)
        rg.create_dataset("heightAboveEllipsoid", data=heights)

        proj = rg.create_dataset("projection", data=np.uint32(32611))
        proj.attrs["epsg_code"] = 32611

        # LOS and along-track vectors (simplified: nadir-ish)
        for name, val in [
            ("losUnitVectorX", 0.3),
            ("losUnitVectorY", 0.0),
            ("alongTrackUnitVectorX", 0.0),
            ("alongTrackUnitVectorY", 1.0),
        ]:
            ds = rg.create_dataset(
                name, data=np.full((n_heights, ny_rg, nx_rg), val, dtype="f4")
            )
            ds.dims[1].attach_scale(yds)
            ds.dims[2].attach_scale(xds)

        # Azimuth times matching orbit time range
        az = np.full((n_heights, ny_rg, nx_rg), 45050.0, dtype="f8")
        rg.create_dataset("zeroDopplerAzimuthTime", data=az)

        rg.create_dataset(
            "incidenceAngle",
            data=np.full((n_heights, ny_rg, nx_rg), 35.0, dtype="f4"),
        )
        rg.create_dataset(
            "elevationAngle",
            data=np.full((n_heights, ny_rg, nx_rg), 30.0, dtype="f4"),
        )
        rg.create_dataset(
            "groundTrackVelocity",
            data=np.full((n_heights, ny_rg, nx_rg), 7000.0, dtype="f8"),
        )
        rg.create_dataset(
            "slantRange",
            data=np.full((n_heights, ny_rg, nx_rg), 800000.0, dtype="f8"),
        )

        # Grids (minimal)
        ny, nx = 8, 10
        grp = f.create_group("science/LSAR/GSLC/grids/frequencyA")
        gx = np.arange(nx, dtype="f8") * 100.0 + 500000.0
        gy = np.arange(ny, dtype="f8") * 100.0 + 4000000.0
        gxds = grp.create_dataset("xCoordinates", data=gx)
        gyds = grp.create_dataset("yCoordinates", data=gy)
        gxds.make_scale("xCoordinates")
        gyds.make_scale("yCoordinates")
        hh = grp.create_dataset("HH", shape=(ny, nx), dtype="c8", chunks=(4, 4))
        hh.dims[0].attach_scale(gyds)
        hh.dims[1].attach_scale(gxds)
        gproj = grp.create_dataset("projection", data=np.uint32(32611))
        gproj.attrs["epsg_code"] = 32611

    return path


class TestComputeBaseline:
    def test_returns_dataset(self, tmp_path):
        ref = _make_gslc_with_orbit(tmp_path / "ref.h5")
        sec = _make_gslc_with_orbit(
            tmp_path / "sec.h5", position_offset=np.array([100.0, 50.0, -200.0])
        )
        dt_ref = open_nisar(ref)
        dt_sec = open_nisar(sec)
        result = compute_baseline(dt_ref, dt_sec)
        assert isinstance(result, xr.Dataset)
        assert "perpendicular_baseline" in result
        assert "parallel_baseline" in result

    def test_output_shape(self, tmp_path):
        ref = _make_gslc_with_orbit(tmp_path / "ref.h5")
        sec = _make_gslc_with_orbit(
            tmp_path / "sec.h5", position_offset=np.array([100.0, 0.0, 0.0])
        )
        dt_ref = open_nisar(ref)
        dt_sec = open_nisar(sec)
        result = compute_baseline(dt_ref, dt_sec)
        assert result["perpendicular_baseline"].shape == (6, 8)

    def test_zero_baseline_for_same_orbit(self, tmp_path):
        ref = _make_gslc_with_orbit(tmp_path / "ref.h5")
        sec = _make_gslc_with_orbit(tmp_path / "sec.h5")
        dt_ref = open_nisar(ref)
        dt_sec = open_nisar(sec)
        result = compute_baseline(dt_ref, dt_sec)
        np.testing.assert_allclose(result["perpendicular_baseline"].values, 0, atol=1e-3)
        np.testing.assert_allclose(result["parallel_baseline"].values, 0, atol=1e-3)

    def test_nonzero_baseline_with_offset(self, tmp_path):
        ref = _make_gslc_with_orbit(tmp_path / "ref.h5")
        sec = _make_gslc_with_orbit(
            tmp_path / "sec.h5", position_offset=np.array([500.0, 200.0, -300.0])
        )
        dt_ref = open_nisar(ref)
        dt_sec = open_nisar(sec)
        result = compute_baseline(dt_ref, dt_sec)
        b_perp = result["perpendicular_baseline"].values
        b_par = result["parallel_baseline"].values
        # At least one component should be nonzero
        assert np.any(np.abs(b_perp) > 1) or np.any(np.abs(b_par) > 1)

    def test_units_in_meters(self, tmp_path):
        ref = _make_gslc_with_orbit(tmp_path / "ref.h5")
        sec = _make_gslc_with_orbit(
            tmp_path / "sec.h5", position_offset=np.array([100.0, 0.0, 0.0])
        )
        dt_ref = open_nisar(ref)
        dt_sec = open_nisar(sec)
        result = compute_baseline(dt_ref, dt_sec)
        assert result["perpendicular_baseline"].attrs["units"] == "meters"
        assert result["parallel_baseline"].attrs["units"] == "meters"
