"""Tests for nisar_pytools.processing.crop_rslc.

The geo2rdr-based window solve (`radar_window_for_aoi`) needs isce3 + a real
orbit, so it is not unit-tested here; instead the pure index math it depends on
(`_window_from_radar_coords`) is tested directly, and the writer (`crop_rslc`)
is tested against a synthetic RSLC. No isce3 dependency.
"""

import h5py
import numpy as np
import pytest

from nisar_pytools.processing import crop_rslc as crop_mod
from nisar_pytools.processing.crop_rslc import (
    _window_from_radar_coords,
    crop_rslc,
    crop_rslc_pair,
)

_SWATHS = "science/LSAR/RSLC/swaths"
_GEOLOC = "science/LSAR/RSLC/metadata/geolocationGrid"

# --- synthetic geometry -----------------------------------------------------
# Clean integer axes so np.searchsorted lands on exact indices.
N_LINES = 100                       # full azimuth lines
N_RGA, N_RGB = 120, 15              # full range samples, freq A / B
SWATH_T = np.arange(N_LINES, dtype="f8")          # 0..99
SLANT_A = np.arange(N_RGA, dtype="f8")            # 0..119
SLANT_B = np.arange(N_RGB, dtype="f8") * 8.0      # 0,8,..,112 (coarse, like real B)
VALID_START, VALID_END = 10, 110                  # validSamplesSubSwath1 row value

# A minimal geolocationGrid for the fixture (no longer used by the window
# solve, but the writer must still copy it verbatim).
N_H, N_AZG, N_RGG = 2, 10, 10
HEIGHTS = np.array([0.0, 1000.0])
AZ_T_G = np.arange(N_AZG, dtype="f8") * 10.0
RG_S_G = np.arange(N_RGG, dtype="f8") * 12.0
LON0, LAT0, DLON, DLAT = -117.0, 44.0, 0.01, 0.01

MARGIN = 4
# Radar-coord extent (az time 30-50 s, slant range 36-60 m) and its hand-derived
# window through _window_from_radar_coords:
#   az: searchsorted(SWATH_T,[30,50])=[30,50] -> +/-4 -> (26,54)
#   A : searchsorted(SLANT_A,[36,60])=[36,60] -> +/-4 -> (32,64)
#   B : searchsorted(SLANT_B, SLANT_A[32]=32)=4 ; searchsorted(SLANT_B,SLANT_A[63]=63)+1=9
AZ_LO, AZ_HI, RG_LO, RG_HI = 30.0, 50.0, 36.0, 60.0
EXPECTED_WINDOW = {"az": (26, 54), "A": (32, 64), "B": (4, 9)}


def _make_rslc(path, pols=("HH", "HV"), n_subswaths=1):
    """Write a tiny but structurally-faithful synthetic RSLC.

    ``pols`` and ``n_subswaths`` vary by acquisition mode in real products, so
    the cropper must read them from the file rather than assume them.
    """
    with h5py.File(path, "w") as f:
        f.attrs["mission_name"] = "NISAR"  # root attr -> must be copied verbatim

        sw = f.create_group(_SWATHS)
        t = sw.create_dataset("zeroDopplerTime", data=SWATH_T)
        t.attrs["units"] = "seconds"       # attr on a cropped axis -> must survive
        sw.create_dataset("zeroDopplerTimeSpacing", data=1.0)

        for fr, slant in (("frequencyA", SLANT_A), ("frequencyB", SLANT_B)):
            g = sw.create_group(fr)
            nrg = slant.size
            # Distinct per-(line,sample) values so slicing is verifiable.
            for k, pol in enumerate(pols):
                img = (np.arange(N_LINES)[:, None] * 1000
                       + np.arange(nrg)[None, :] + k * 0.5j).astype("c8")
                g.create_dataset(pol, data=img, chunks=(8, min(8, nrg)))
            g.create_dataset("slantRange", data=slant)
            g.create_dataset("slantRangeSpacing", data=float(slant[1] - slant[0]))
            vs = np.tile([VALID_START, VALID_END], (N_LINES, 1)).astype("u4")
            for n in range(1, n_subswaths + 1):
                g.create_dataset(f"validSamplesSubSwath{n}", data=vs)
            g.create_dataset(
                "listOfPolarizations", data=np.array([p.encode() for p in pols]))

        # geolocationGrid: radar -> lon/lat lookup the cropper reads.
        gl = f.create_group(_GEOLOC)
        gl.create_dataset("heightAboveEllipsoid", data=HEIGHTS)
        gl.create_dataset("zeroDopplerTime", data=AZ_T_G)
        gl.create_dataset("slantRange", data=RG_S_G)
        lon = LON0 + DLON * np.arange(N_RGG)[None, None, :]
        lat = LAT0 + DLAT * np.arange(N_AZG)[None, :, None]
        gl.create_dataset("coordinateX", data=np.broadcast_to(lon, (N_H, N_AZG, N_RGG)).copy())
        gl.create_dataset("coordinateY", data=np.broadcast_to(lat, (N_H, N_AZG, N_RGG)).copy())
        gl.create_dataset("epsg", data=np.int32(4326))

        # A metadata dataset on no cropped axis -> must be copied byte-for-byte.
        orb = f.create_dataset("science/LSAR/RSLC/metadata/orbit/position",
                               data=np.arange(15, dtype="f8").reshape(5, 3))
        orb.attrs["description"] = "ECEF position"


@pytest.fixture
def rslc_h5(tmp_path):
    path = tmp_path / "synthetic_rslc.h5"
    _make_rslc(path)
    return path


class TestWindowMath:
    """The pure (time, range) -> (line, pixel) index logic, isce3-free."""

    def test_window_matches_hand_derived(self):
        win = _window_from_radar_coords(
            SWATH_T, SLANT_A, SLANT_B, AZ_LO, AZ_HI, RG_LO, RG_HI, margin=MARGIN)
        assert win == EXPECTED_WINDOW

    def test_partial_overlap_is_clamped(self):
        # Range extent starts before the axis -> low bound clamps to 0.
        win = _window_from_radar_coords(
            SWATH_T, SLANT_A, SLANT_B, AZ_LO, AZ_HI, -50.0, RG_HI, margin=MARGIN)
        assert win["A"][0] == 0

    def test_no_overlap_raises(self):
        # Azimuth extent entirely past the end of the swath.
        with pytest.raises(ValueError, match="does not overlap"):
            _window_from_radar_coords(
                SWATH_T, SLANT_A, SLANT_B, 1e6, 2e6, RG_LO, RG_HI, margin=MARGIN)

    def test_min_size_does_not_bind_when_margin_dominates(self):
        # Raw az extent = 20 lines; +MARGIN(4) each side = 28. min_size=10 < 28,
        # so the floor is inert and the window is unchanged.
        win = _window_from_radar_coords(
            SWATH_T, SLANT_A, SLANT_B, AZ_LO, AZ_HI, RG_LO, RG_HI,
            margin=MARGIN, min_size=10)
        assert win == EXPECTED_WINDOW

    def test_min_size_floor_widens_small_window(self):
        # Tiny extent (one cell) with a small margin: min_size forces the span.
        # raw az [40,40] (len 0) -> need pad=ceil((40-0+1)/2)=20 -> [20,60]=40 lines.
        win = _window_from_radar_coords(
            SWATH_T, SLANT_A, SLANT_B, 40.0, 40.0, 48.0, 48.0,
            margin=2, min_size=40)
        a0, a1 = win["az"]
        p0, p1 = win["A"]
        assert a1 - a0 == 40          # floored to min_size, not 2*margin
        assert p1 - p0 == 40
        # margin=2 alone would have given only 4 -> floor clearly bound
        assert a1 - a0 > 2 * 2


class TestCropRslc:
    def test_shapes_and_axes(self, rslc_h5, tmp_path):
        dst = tmp_path / "out.h5"
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW)
        a0, a1 = EXPECTED_WINDOW["az"]
        with h5py.File(dst, "r") as f:
            # Shared azimuth axis sliced once.
            assert f[f"{_SWATHS}/zeroDopplerTime"].shape == (a1 - a0,)
            np.testing.assert_array_equal(
                f[f"{_SWATHS}/zeroDopplerTime"][()], SWATH_T[a0:a1])
            for fr, slant, key in (("frequencyA", SLANT_A, "A"),
                                   ("frequencyB", SLANT_B, "B")):
                p0, p1 = EXPECTED_WINDOW[key]
                assert f[f"{_SWATHS}/{fr}/HH"].shape == (a1 - a0, p1 - p0)
                assert f[f"{_SWATHS}/{fr}/HV"].shape == (a1 - a0, p1 - p0)
                np.testing.assert_array_equal(
                    f[f"{_SWATHS}/{fr}/slantRange"][()], slant[p0:p1])

    def test_image_content_is_the_window(self, rslc_h5, tmp_path):
        dst = tmp_path / "out.h5"
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW)
        a0, a1 = EXPECTED_WINDOW["az"]
        p0, p1 = EXPECTED_WINDOW["A"]
        with h5py.File(rslc_h5, "r") as src, h5py.File(dst, "r") as out:
            expect = src[f"{_SWATHS}/frequencyA/HH"][a0:a1, p0:p1]
            np.testing.assert_array_equal(out[f"{_SWATHS}/frequencyA/HH"][()], expect)

    def test_valid_samples_shifted_and_clipped(self, rslc_h5, tmp_path):
        dst = tmp_path / "out.h5"
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW)
        a0, a1 = EXPECTED_WINDOW["az"]
        p0, p1 = EXPECTED_WINDOW["A"]
        with h5py.File(dst, "r") as f:
            vs = f[f"{_SWATHS}/frequencyA/validSamplesSubSwath1"][()]
        assert vs.shape == (a1 - a0, 2)
        # start = clip(10 - 32, 0, 32) = 0 ; end = clip(110 - 32, 0, 32) = 32
        expect = np.clip(np.array([VALID_START, VALID_END]) - p0, 0, p1 - p0)
        np.testing.assert_array_equal(vs[0], expect)
        assert (vs == expect).all()

    def test_metadata_and_attrs_copied_verbatim(self, rslc_h5, tmp_path):
        dst = tmp_path / "out.h5"
        crop_rslc(rslc_h5, dst, EXPECTED_WINDOW)
        with h5py.File(rslc_h5, "r") as src, h5py.File(dst, "r") as out:
            # Root attribute preserved.
            assert out.attrs["mission_name"] == src.attrs["mission_name"]
            # Uncropped metadata dataset copied byte-for-byte, with its attrs.
            o = out["science/LSAR/RSLC/metadata/orbit/position"]
            np.testing.assert_array_equal(
                o[()], src["science/LSAR/RSLC/metadata/orbit/position"][()])
            assert o.attrs["description"] == "ECEF position"
            # geolocationGrid (a lookup table) is NOT cropped.
            assert out[f"{_GEOLOC}/coordinateX"].shape == (N_H, N_AZG, N_RGG)
            # Attribute on a cropped axis survives.
            assert out[f"{_SWATHS}/zeroDopplerTime"].attrs["units"] == "seconds"


class TestQuadPolMultiSubswath:
    """Every polarization and every subswath table must be cropped.

    NISAR quad-pol products carry HH/HV/VH/VV and up to four subswath tables;
    anything left uncropped stays full-frame while the axes around it shrink.
    """

    @pytest.fixture
    def quad_h5(self, tmp_path):
        path = tmp_path / "quad_rslc.h5"
        _make_rslc(path, pols=("HH", "HV", "VH", "VV"), n_subswaths=4)
        return path

    def test_all_polarizations_cropped(self, quad_h5, tmp_path):
        dst = tmp_path / "out.h5"
        crop_rslc(quad_h5, dst, EXPECTED_WINDOW)
        a0, a1 = EXPECTED_WINDOW["az"]
        with h5py.File(dst, "r") as f:
            for fr, key in (("frequencyA", "A"), ("frequencyB", "B")):
                p0, p1 = EXPECTED_WINDOW[key]
                for pol in ("HH", "HV", "VH", "VV"):
                    assert f[f"{_SWATHS}/{fr}/{pol}"].shape == (a1 - a0, p1 - p0)

    def test_all_subswath_tables_cropped_and_shifted(self, quad_h5, tmp_path):
        dst = tmp_path / "out.h5"
        crop_rslc(quad_h5, dst, EXPECTED_WINDOW)
        a0, a1 = EXPECTED_WINDOW["az"]
        p0, p1 = EXPECTED_WINDOW["A"]
        expect = np.clip(np.array([VALID_START, VALID_END]) - p0, 0, p1 - p0)
        with h5py.File(dst, "r") as f:
            for n in range(1, 5):
                vs = f[f"{_SWATHS}/frequencyA/validSamplesSubSwath{n}"][()]
                assert vs.shape == (a1 - a0, 2)
                assert (vs == expect).all()

    def test_pol_content_is_the_window(self, quad_h5, tmp_path):
        dst = tmp_path / "out.h5"
        crop_rslc(quad_h5, dst, EXPECTED_WINDOW)
        a0, a1 = EXPECTED_WINDOW["az"]
        p0, p1 = EXPECTED_WINDOW["A"]
        with h5py.File(quad_h5, "r") as src, h5py.File(dst, "r") as out:
            for pol in ("VH", "VV"):
                expect = src[f"{_SWATHS}/frequencyA/{pol}"][a0:a1, p0:p1]
                np.testing.assert_array_equal(
                    out[f"{_SWATHS}/frequencyA/{pol}"][()], expect)

    def test_polarizations_discovered_without_list(self, quad_h5, tmp_path):
        """Falls back to the complex 2D datasets when the list is absent."""
        with h5py.File(quad_h5, "r+") as f:
            for fr in ("frequencyA", "frequencyB"):
                del f[f"{_SWATHS}/{fr}/listOfPolarizations"]
        dst = tmp_path / "out.h5"
        crop_rslc(quad_h5, dst, EXPECTED_WINDOW)
        a0, a1 = EXPECTED_WINDOW["az"]
        p0, p1 = EXPECTED_WINDOW["A"]
        with h5py.File(dst, "r") as f:
            for pol in ("HH", "HV", "VH", "VV"):
                assert f[f"{_SWATHS}/frequencyA/{pol}"].shape == (a1 - a0, p1 - p0)


class TestCropPair:
    def test_pair_writes_two_files(self, rslc_h5, tmp_path, monkeypatch):
        # The geo2rdr window solve needs isce3; stub it to a fixed window so
        # the pair plumbing (per-file crop + naming) is testable on its own.
        monkeypatch.setattr(
            crop_mod, "radar_window_for_aoi",
            lambda *a, **k: EXPECTED_WINDOW,
        )
        # A real pair is two distinct files (different acquisition dates);
        # mimic that so the per-file output names don't collide.
        sec = tmp_path / "synthetic_rslc_sec.h5"
        _make_rslc(sec)
        ref_sub, sec_sub = crop_rslc_pair(
            rslc_h5, sec, bbox_utm=None, epsg_aoi=None, dem_file=None,
            out_dir=tmp_path / "subs", margin=MARGIN)
        for p in (ref_sub, sec_sub):
            assert p.exists()
            with h5py.File(p, "r") as f:
                a0, a1 = EXPECTED_WINDOW["az"]
                assert f[f"{_SWATHS}/zeroDopplerTime"].shape == (a1 - a0,)
        assert ref_sub != sec_sub
