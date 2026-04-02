"""Tests for nisar_pytools.utils.local_incidence_angle."""

import numpy as np
import xarray as xr

from nisar_pytools.utils.local_incidence_angle import (
    compute_surface_normal,
    interpolate_los_to_dem,
    local_incidence_angle,
)


def _make_dem(nx=20, ny=15, flat=True):
    """Create a synthetic DEM DataArray."""
    x = np.arange(nx, dtype="f8") * 30.0 + 500000.0
    y = np.arange(ny, dtype="f8") * -30.0 + 4500000.0  # descending y
    if flat:
        elev = np.full((ny, nx), 1000.0, dtype="f4")
    else:
        # Tilted plane: slopes east and north
        xx, yy = np.meshgrid(x, y)
        elev = ((xx - x[0]) * 0.1 + (yy - y[-1]) * 0.05).astype("f4")
    return xr.DataArray(elev, dims=["y", "x"], coords={"y": y, "x": x})


def _make_los_nadir(n_heights=3, ny=10, nx=12):
    """Create synthetic LOS vectors pointing straight down (nadir)."""
    heights = np.linspace(500, 2000, n_heights)
    x_rg = np.linspace(499900, 500600, nx)
    y_rg = np.linspace(4499600, 4500100, ny)
    los_x = np.zeros((n_heights, ny, nx), dtype="f4")
    los_y = np.zeros((n_heights, ny, nx), dtype="f4")
    los_z = np.ones((n_heights, ny, nx), dtype="f4")
    return los_x, los_y, los_z, heights, x_rg, y_rg


def _make_los_angled(n_heights=3, ny=10, nx=12, angle_deg=30.0):
    """Create synthetic LOS vectors at a fixed incidence angle from nadir."""
    heights = np.linspace(500, 2000, n_heights)
    x_rg = np.linspace(499900, 500600, nx)
    y_rg = np.linspace(4499600, 4500100, ny)
    angle_rad = np.radians(angle_deg)
    los_x = np.full((n_heights, ny, nx), np.sin(angle_rad), dtype="f4")
    los_y = np.zeros((n_heights, ny, nx), dtype="f4")
    los_z = np.full((n_heights, ny, nx), np.cos(angle_rad), dtype="f4")
    return los_x, los_y, los_z, heights, x_rg, y_rg


class TestComputeSurfaceNormal:
    def test_flat_dem(self):
        dem = _make_dem(flat=True)
        nx, ny, nz = compute_surface_normal(dem)
        # Flat surface → normal points straight up
        np.testing.assert_allclose(nx, 0.0, atol=1e-10)
        np.testing.assert_allclose(ny, 0.0, atol=1e-10)
        np.testing.assert_allclose(nz, 1.0, atol=1e-10)

    def test_tilted_dem_has_nonzero_horizontal(self):
        dem = _make_dem(flat=False)
        nx, ny, nz = compute_surface_normal(dem)
        # Tilted surface → normals have horizontal components
        assert np.any(np.abs(nx) > 0.01)
        # All normals should be unit length
        mag = np.sqrt(nx**2 + ny**2 + nz**2)
        np.testing.assert_allclose(mag, 1.0, atol=1e-6)

    def test_output_shapes(self):
        dem = _make_dem(nx=10, ny=8)
        nx, ny, nz = compute_surface_normal(dem)
        assert nx.shape == (8, 10)
        assert ny.shape == (8, 10)
        assert nz.shape == (8, 10)


class TestInterpolateLos:
    def test_nadir_interpolation(self):
        dem = _make_dem(flat=True)
        los_x, los_y, los_z, heights, x_rg, y_rg = _make_los_nadir()
        le, ln, lu = interpolate_los_to_dem(dem, los_x, los_y, los_z, heights, x_rg, y_rg)
        # Nadir LOS → all Z, no X/Y
        valid = ~np.isnan(lu)
        np.testing.assert_allclose(le[valid], 0.0, atol=1e-6)
        np.testing.assert_allclose(ln[valid], 0.0, atol=1e-6)
        np.testing.assert_allclose(lu[valid], 1.0, atol=1e-6)

    def test_output_shape_matches_dem(self):
        dem = _make_dem(nx=20, ny=15)
        los_x, los_y, los_z, heights, x_rg, y_rg = _make_los_nadir()
        le, ln, lu = interpolate_los_to_dem(dem, los_x, los_y, los_z, heights, x_rg, y_rg)
        assert le.shape == (15, 20)


class TestLocalIncidenceAngle:
    def test_nadir_flat_gives_zero(self):
        dem = _make_dem(flat=True)
        los_x, los_y, los_z, heights, x_rg, y_rg = _make_los_nadir()
        lia = local_incidence_angle(dem, los_x, los_y, los_z, heights, x_rg, y_rg)
        valid = ~np.isnan(lia.values)
        # Nadir on flat surface → 0 degrees
        np.testing.assert_allclose(lia.values[valid], 0.0, atol=0.5)

    def test_angled_flat_gives_expected(self):
        dem = _make_dem(flat=True)
        angle = 35.0
        los_x, los_y, los_z, heights, x_rg, y_rg = _make_los_angled(angle_deg=angle)
        lia = local_incidence_angle(dem, los_x, los_y, los_z, heights, x_rg, y_rg)
        valid = ~np.isnan(lia.values)
        # Flat surface with angled LOS → angle should match
        np.testing.assert_allclose(lia.values[valid], angle, atol=1.0)

    def test_output_is_dataarray(self):
        dem = _make_dem()
        los_x, los_y, los_z, heights, x_rg, y_rg = _make_los_nadir()
        lia = local_incidence_angle(dem, los_x, los_y, los_z, heights, x_rg, y_rg)
        assert isinstance(lia, xr.DataArray)
        assert lia.name == "local_incidence_angle"
        assert lia.attrs["units"] == "degrees"

    def test_coords_match_dem(self):
        dem = _make_dem(nx=20, ny=15)
        los_x, los_y, los_z, heights, x_rg, y_rg = _make_los_nadir()
        lia = local_incidence_angle(dem, los_x, los_y, los_z, heights, x_rg, y_rg)
        np.testing.assert_array_equal(lia.x.values, dem.x.values)
        np.testing.assert_array_equal(lia.y.values, dem.y.values)

    def test_epsg_assigned(self):
        dem = _make_dem()
        los_x, los_y, los_z, heights, x_rg, y_rg = _make_los_nadir()
        lia = local_incidence_angle(
            dem, los_x, los_y, los_z, heights, x_rg, y_rg, epsg=32611
        )
        assert lia.rio.crs.to_epsg() == 32611

    def test_values_in_valid_range(self):
        dem = _make_dem(flat=False)
        los_x, los_y, los_z, heights, x_rg, y_rg = _make_los_angled(angle_deg=40.0)
        lia = local_incidence_angle(dem, los_x, los_y, los_z, heights, x_rg, y_rg)
        valid = lia.values[~np.isnan(lia.values)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 180)
