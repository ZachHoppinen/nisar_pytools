"""Tests for nisar_pytools.utils.dem (unit tests only, no network)."""

import h5py
import numpy as np
import pytest
import xarray as xr

from nisar_pytools.utils.dem import _bounds_from_dataarray, _bounds_from_h5, _project_bounds


class TestProjectBounds:
    def test_identity_for_4326(self):
        x = np.array([-115.0, -114.0])
        y = np.array([43.0, 44.0])
        bounds = _project_bounds(x, y, epsg=4326, buffer=0.0)
        np.testing.assert_allclose(bounds, [-115.0, 43.0, -114.0, 44.0], atol=1e-6)

    def test_buffer_applied(self):
        x = np.array([-115.0, -114.0])
        y = np.array([43.0, 44.0])
        bounds = _project_bounds(x, y, epsg=4326, buffer=0.1)
        assert bounds[0] < -115.0
        assert bounds[2] > -114.0

    def test_utm_to_latlon(self):
        # UTM zone 11N coords roughly around Idaho
        x = np.array([500000.0, 600000.0])
        y = np.array([4800000.0, 4900000.0])
        bounds = _project_bounds(x, y, epsg=32611, buffer=0.0)
        # Should be reasonable lat/lon values
        assert -120 < bounds[0] < -110
        assert 40 < bounds[1] < 50


class TestBoundsFromH5:
    def test_extracts_bounds(self, tmp_path):
        fp = tmp_path / "test.h5"
        with h5py.File(fp, "w") as f:
            grp = f.create_group("science/LSAR/GSLC/grids/frequencyA")
            grp.create_dataset("xCoordinates", data=np.array([500000.0, 600000.0]))
            grp.create_dataset("yCoordinates", data=np.array([4800000.0, 4900000.0]))
            proj = grp.create_dataset("projection", data=np.uint32(32611))
            proj.attrs["epsg_code"] = 32611

        bounds = _bounds_from_h5(fp, buffer=0.05)
        assert len(bounds) == 4
        assert bounds[0] < bounds[2]
        assert bounds[1] < bounds[3]

    def test_no_coords_raises(self, tmp_path):
        fp = tmp_path / "empty.h5"
        with h5py.File(fp, "w") as f:
            f.create_group("empty")
        with pytest.raises(ValueError, match="No xCoordinates"):
            _bounds_from_h5(fp, buffer=0.0)


class TestBoundsFromDataarray:
    def test_with_crs(self):
        import rioxarray  # noqa: F401
        x = np.arange(10, dtype="f8") * 100.0 + 500000.0
        y = np.arange(8, dtype="f8") * -100.0 + 4500000.0
        da = xr.DataArray(
            np.ones((8, 10)), dims=["y", "x"], coords={"y": y, "x": x}
        )
        da = da.rio.write_crs(32611)
        bounds = _bounds_from_dataarray(da, buffer=0.0)
        assert len(bounds) == 4

    def test_no_coords_raises(self):
        da = xr.DataArray(np.ones((3, 3)), dims=["a", "b"])
        with pytest.raises(ValueError, match="must have 'x' and 'y'"):
            _bounds_from_dataarray(da, buffer=0.0)
