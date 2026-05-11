"""Tests for nisar_pytools.utils.metadata."""

import h5py
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from nisar_pytools import open_nisar
import pytest

from nisar_pytools.utils.metadata import (
    get_acquisition_time,
    get_bounding_polygon,
    get_gunw,
    get_orbit_info,
    get_product_type,
    get_slc,
)


class TestMetadata:
    def test_product_type(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        assert get_product_type(dt) == "GSLC"

    def test_acquisition_time(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        ts = get_acquisition_time(dt)
        assert isinstance(ts, pd.Timestamp)

    def test_orbit_info(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        info = get_orbit_info(dt)
        assert "track_number" in info
        assert "frame_number" in info
        assert "orbit_direction" in info
        assert info["track_number"] == 77

    def test_bounding_polygon_from_fixture(self, gslc_h5):
        # The synthetic fixture now provides a boundingPolygon (see conftest).
        dt = open_nisar(gslc_h5)
        poly = get_bounding_polygon(dt)
        assert isinstance(poly, Polygon)
        assert poly.area > 0

    def test_get_slc_hh(self, gslc_h5):
        # valid_mask defaults to True; the synthetic fixture leaves mask=0
        # everywhere, which means "all invalid". We disable masking here to
        # check shape/dims independently of mask content.
        dt = open_nisar(gslc_h5)
        hh = get_slc(dt, "HH", valid_mask=False)
        assert hh.shape == (8, 10)
        assert hh.dims == ("y", "x")

    def test_get_slc_hv(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        hv = get_slc(dt, "HV", valid_mask=False)
        assert hv.shape == (8, 10)

    def test_get_slc_freq_b(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        hh_b = get_slc(dt, "HH", frequency="frequencyB", valid_mask=False)
        assert hh_b.shape == (8, 5)

    def test_get_slc_missing_pol_raises(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        with pytest.raises(ValueError, match="not found"):
            get_slc(dt, "VV")

    def test_get_slc_missing_freq_raises(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        with pytest.raises(ValueError, match="not found"):
            get_slc(dt, frequency="frequencyC")


class TestGetSlcValidMask:
    """GSLC mask rule: keep pixels where mask != 0 and mask != 255."""

    def _write_mask(self, path, freq, values):
        with h5py.File(path, "r+") as f:
            f[f"science/LSAR/GSLC/grids/{freq}/mask"][...] = values

    def test_valid_mask_drops_invalid_and_fill(self, gslc_h5):
        ny, nx = 8, 10
        mask = np.full((ny, nx), 1, dtype="u1")  # all valid (subswath 1)
        mask[0, 0] = 0     # invalid (partial focus)
        mask[0, 1] = 255   # fill (outside extent)
        self._write_mask(gslc_h5, "frequencyA", mask)

        dt = open_nisar(gslc_h5)
        hh = get_slc(dt, "HH", valid_mask=True).compute()
        assert np.isnan(hh.values[0, 0])
        assert np.isnan(hh.values[0, 1])
        # The rest of the grid is "valid"; underlying HH data is uninitialized
        # zeros, so masked-out positions are NaN and the kept positions are 0+0j.
        assert hh.values[1, 1] == 0 + 0j

    def test_valid_mask_false_preserves_complex_zero(self, gslc_h5):
        # mask=0 everywhere means "all invalid" under the rule, so with
        # valid_mask=True we'd get NaN everywhere. With valid_mask=False the
        # raw zeros pass through.
        dt = open_nisar(gslc_h5)
        hh = get_slc(dt, "HH", valid_mask=False).compute()
        assert not np.any(np.isnan(hh.values))

    def test_valid_mask_true_with_zero_mask_blanks_everything(self, gslc_h5):
        ny, nx = 8, 10
        self._write_mask(gslc_h5, "frequencyA", np.zeros((ny, nx), dtype="u1"))
        dt = open_nisar(gslc_h5)
        hh = get_slc(dt, "HH", valid_mask=True).compute()
        assert np.all(np.isnan(hh.values))


class TestGetGunw:
    def test_unwrapped_phase_default(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        phase = get_gunw(dt, valid_mask=False)
        assert phase.shape == (6, 8)
        assert phase.dims == ("y", "x")

    def test_wrapped_interferogram(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        wifg = get_gunw(
            dt,
            variable="wrappedInterferogram",
            layer="wrappedInterferogram",
            valid_mask=False,
        )
        assert wifg.shape == (12, 16)

    def test_pixel_offsets(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        offs = get_gunw(
            dt,
            variable="alongTrackOffset",
            layer="pixelOffsets",
            valid_mask=False,
        )
        assert offs.shape == (6, 8)

    def test_missing_variable_raises(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        with pytest.raises(ValueError, match="Variable 'foo' not found"):
            get_gunw(dt, variable="foo")

    def test_missing_layer_raises(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        with pytest.raises(ValueError, match="Layer 'bogus' not found"):
            get_gunw(dt, layer="bogus")

    def test_missing_polarization_raises(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        with pytest.raises(ValueError, match="Polarization 'VV' not found"):
            get_gunw(dt, polarization="VV")

    def test_wrong_product_type_raises(self, gslc_h5):
        dt = open_nisar(gslc_h5)
        with pytest.raises(ValueError, match="expects a GUNW DataTree"):
            get_gunw(dt)


class TestGetGunwValidMask:
    """GUNW mask rule: keep pixels where both subswath digits are nonzero and mask != 255."""

    def _write_mask(self, path, layer, values):
        with h5py.File(path, "r+") as f:
            f[f"science/LSAR/GUNW/grids/frequencyA/{layer}/mask"][...] = values

    def test_valid_mask_drops_invalid_subswaths_and_fill(self, gunw_h5):
        ny, nx = 6, 8
        mask = np.full((ny, nx), 11, dtype="u1")  # WRS=011 -> land, ref=1, sec=1
        mask[0, 0] = 10    # WRS=010 -> sec subswath=0 -> invalid
        mask[0, 1] = 1     # WRS=001 -> ref subswath=0 -> invalid
        mask[0, 2] = 255   # fill
        mask[0, 3] = 111   # WRS=111 -> water but valid samples -> kept
        self._write_mask(gunw_h5, "unwrappedInterferogram", mask)

        dt = open_nisar(gunw_h5)
        phase = get_gunw(dt, valid_mask=True).compute()
        assert np.isnan(phase.values[0, 0])
        assert np.isnan(phase.values[0, 1])
        assert np.isnan(phase.values[0, 2])
        assert not np.isnan(phase.values[0, 3])  # water kept
        assert not np.isnan(phase.values[1, 0])

    def test_valid_mask_false_skips_masking(self, gunw_h5):
        dt = open_nisar(gunw_h5)
        phase = get_gunw(dt, valid_mask=False).compute()
        assert not np.any(np.isnan(phase.values))

    def test_integer_variable_promoted_to_float_when_masked(self, gunw_h5):
        # connectedComponents is uint16; masking promotes it to float so NaN fits.
        ny, nx = 6, 8
        mask = np.full((ny, nx), 11, dtype="u1")
        mask[0, 0] = 0  # invalid -> NaN
        self._write_mask(gunw_h5, "unwrappedInterferogram", mask)

        dt = open_nisar(gunw_h5)
        cc = get_gunw(dt, variable="connectedComponents", valid_mask=True).compute()
        assert np.isnan(cc.values[0, 0])
        assert np.issubdtype(cc.dtype, np.floating)
