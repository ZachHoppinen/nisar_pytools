"""Tests for nisar_pytools.processing.prep_dolphin."""

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import rioxarray  # noqa: F401

from nisar_pytools.processing.prep_dolphin import (
    _set_nested,
    _get_epsg,
    crop_gslc_to_tif,
    _generate_dolphin_config,
    prep_dolphin,
)
from nisar_pytools.utils.overlap import reproject_bbox
from nisar_pytools.utils.validation import validate_frequency, validate_polarization


EPSG = 32611
NX, NY = 30, 20
# UTM grid starting at (500000, 4000000), 5m spacing.
X0, Y0, DX = 500000.0, 4000000.0, 5.0


def _make_gslc_file(
    path: Path,
    date_str: str = "20251128",
    time_str: str | None = None,
    track: int = 77,
    frame: int = 24,
    ny: int = NY,
    nx: int = NX,
    pols: tuple[str, ...] = ("HH", "HV"),
    epsg: int = EPSG,
) -> Path:
    """Create a minimal synthetic GSLC HDF5 matching NISAR naming."""
    if time_str is None:
        time_str = f"{date_str}T124615"
    end_str = time_str.replace("124615", "124650")

    # NISAR-convention filename so parse_filename works.
    fname = (
        f"NISAR_L2_PR_GSLC_004_{track:03d}_A_{frame:03d}_4005_DHDH_A_"
        f"{time_str}_{end_str}_X05009_N_F_J_001.h5"
    )
    fp = path / fname
    with h5py.File(fp, "w") as f:
        ident = f.create_group("science/LSAR/identification")
        ident.create_dataset("productType", data=b"GSLC")
        ident.create_dataset("zeroDopplerStartTime", data=time_str.encode())
        ident.create_dataset("trackNumber", data=np.uint32(track))
        ident.create_dataset("frameNumber", data=np.uint16(frame))

        grp = f.create_group("science/LSAR/GSLC/grids/frequencyA")
        x = np.arange(nx, dtype="f8") * DX + X0
        y = np.arange(ny, dtype="f8") * DX + Y0
        xds = grp.create_dataset("xCoordinates", data=x)
        yds = grp.create_dataset("yCoordinates", data=y)
        xds.make_scale("xCoordinates")
        yds.make_scale("yCoordinates")

        proj = grp.create_dataset("projection", data=np.uint32(epsg))
        proj.attrs["epsg_code"] = epsg

        rng = np.random.default_rng(hash(date_str) % 2**31)
        for pol in pols:
            data = (rng.normal(size=(ny, nx)) +
                    1j * rng.normal(size=(ny, nx))).astype(np.complex64)
            ds = grp.create_dataset(pol, data=data, chunks=(min(4, ny), min(4, nx)))
            ds.dims[0].attach_scale(yds)
            ds.dims[1].attach_scale(xds)

    return fp


@pytest.fixture
def gslc_dir(tmp_path):
    """Create 4 GSLC files with different dates."""
    dates = ["20251128", "20251210", "20251222", "20260115"]
    d = tmp_path / "gslcs"
    d.mkdir()
    paths = [_make_gslc_file(d, date_str=dt) for dt in dates]
    return d, paths, dates


def _mock_dolphin_config(cmd, **kwargs):
    """Mock subprocess.run for `dolphin config`."""
    idx = cmd.index("--outfile")
    outfile = Path(cmd[idx + 1])
    outfile.write_text(
        "input_options:\n"
        "  wavelength: null\n"
        "phase_linking:\n"
        "  half_window:\n"
        "    x: 11\n"
        "    y: 7\n"
    )


# -- _set_nested --

class TestSetNested:
    def test_shallow(self):
        d = {}
        _set_nested(d, ["key"], 42)
        assert d == {"key": 42}

    def test_deep(self):
        d = {}
        _set_nested(d, ["a", "b", "c"], "val")
        assert d == {"a": {"b": {"c": "val"}}}

    def test_overwrites_existing(self):
        d = {"a": {"b": 1}}
        _set_nested(d, ["a", "b"], 2)
        assert d["a"]["b"] == 2


# -- _get_epsg --

class TestGetEpsg:
    def _cache(self):
        import sys
        return sys.modules["nisar_pytools.processing.prep_dolphin"]._epsg_cache

    def test_reads_epsg(self, tmp_path):
        self._cache().clear()
        fp = _make_gslc_file(tmp_path, epsg=32612)
        assert _get_epsg(fp) == 32612

    def test_caches_result(self, tmp_path):
        self._cache().clear()
        fp = _make_gslc_file(tmp_path, epsg=32611)
        _get_epsg(fp)
        assert str(fp) in self._cache()


# -- validate_frequency / validate_polarization --

class TestValidateFrequency:
    def test_valid(self):
        assert validate_frequency("frequencyA") == "frequencyA"
        assert validate_frequency("frequencyB") == "frequencyB"

    def test_invalid(self):
        with pytest.raises(ValueError, match="frequency"):
            validate_frequency("frequencyC")

    def test_case_sensitive(self):
        with pytest.raises(ValueError):
            validate_frequency("FrequencyA")


class TestValidatePolarization:
    def test_valid(self):
        for pol in ("HH", "HV", "VH", "VV"):
            assert validate_polarization(pol) == pol

    def test_invalid(self):
        with pytest.raises(ValueError, match="polarization"):
            validate_polarization("XX")

    def test_lowercase_rejected(self):
        with pytest.raises(ValueError):
            validate_polarization("hh")


# -- reproject_bbox --

class TestReprojectBbox:
    def test_wgs84_to_utm(self):
        bbox = reproject_bbox((-110.0, 44.0, -109.0, 45.0), src_crs=4326, dst_crs=32612)
        xmin, ymin, xmax, ymax = bbox
        assert 400_000 < xmin < 600_000
        assert xmax > xmin
        assert ymax > ymin

    def test_roundtrip(self):
        original = (-110.0, 44.0, -109.0, 45.0)
        utm = reproject_bbox(original, src_crs=4326, dst_crs=32612)
        back = reproject_bbox(utm, src_crs=32612, dst_crs=4326)
        for a, b in zip(original, back):
            assert abs(a - b) < 0.05


# -- crop_gslc_to_tif --

class TestCropGslcToTif:
    def test_full_extent(self, tmp_path):
        fp = _make_gslc_file(tmp_path)
        out = tmp_path / "out.tif"
        crop_gslc_to_tif(fp, out, bbox_utm=None)
        da = rioxarray.open_rasterio(out, masked=False).squeeze()
        assert da.shape == (NY, NX)
        assert da.rio.crs is not None
        assert da.rio.crs.to_epsg() == EPSG

    def test_cropped_bbox(self, tmp_path):
        fp = _make_gslc_file(tmp_path, nx=30, ny=20)
        out = tmp_path / "cropped.tif"
        # Crop to the middle half in x.
        xmin = X0 + 7 * DX
        xmax = X0 + 22 * DX
        ymin = Y0
        ymax = Y0 + (NY - 1) * DX
        crop_gslc_to_tif(fp, out, bbox_utm=(xmin, ymin, xmax, ymax))
        da = rioxarray.open_rasterio(out, masked=False).squeeze()
        assert da.shape[1] < NX
        assert da.shape[0] == NY

    def test_complex_dtype(self, tmp_path):
        fp = _make_gslc_file(tmp_path)
        out = tmp_path / "out.tif"
        crop_gslc_to_tif(fp, out)
        da = rioxarray.open_rasterio(out, masked=False).squeeze()
        assert np.iscomplexobj(da.values)

    def test_hv_polarization(self, tmp_path):
        fp = _make_gslc_file(tmp_path, pols=("HH", "HV"))
        out = tmp_path / "hv.tif"
        crop_gslc_to_tif(fp, out, polarization="HV")
        da = rioxarray.open_rasterio(out, masked=False).squeeze()
        assert da.shape == (NY, NX)

    def test_no_overlap_raises(self, tmp_path):
        fp = _make_gslc_file(tmp_path)
        out = tmp_path / "empty.tif"
        with pytest.raises(ValueError, match="no overlap"):
            crop_gslc_to_tif(fp, out, bbox_utm=(0, 0, 1, 1))


# -- prep_dolphin --

class TestPrepDolphinValidation:
    """Input validation tests for prep_dolphin."""

    def test_empty_paths_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No GSLC paths"):
            prep_dolphin([], tmp_path / "out")

    def test_bad_skip_date_format(self, gslc_dir, tmp_path):
        _, paths, _ = gslc_dir
        with pytest.raises(ValueError, match="YYYYMMDD"):
            prep_dolphin(paths, tmp_path / "out", skip_dates={"2025-12-10"})

    def test_bad_frequency(self, gslc_dir, tmp_path):
        _, paths, _ = gslc_dir
        with pytest.raises(ValueError, match="frequency"):
            prep_dolphin(paths, tmp_path / "out", frequency="frequencyZ")

    def test_bad_polarization(self, gslc_dir, tmp_path):
        _, paths, _ = gslc_dir
        with pytest.raises(ValueError, match="polarization"):
            prep_dolphin(paths, tmp_path / "out", polarization="XX")

    def test_bad_aoi_raises(self, gslc_dir, tmp_path):
        _, paths, _ = gslc_dir
        with pytest.raises(ValueError):
            prep_dolphin(paths, tmp_path / "out",
                         aoi_wgs84=(-200, 44, -109, 45))


class TestPrepDolphin:
    @patch("nisar_pytools.processing.prep_dolphin.subprocess.run",
           side_effect=_mock_dolphin_config)
    def test_basic_export(self, mock_run, gslc_dir, tmp_path):
        _, paths, dates = gslc_dir
        out = tmp_path / "output"
        cfg = prep_dolphin(paths, out)
        assert cfg.exists()
        tifs = sorted((out / "slcs").glob("*.tif"))
        assert len(tifs) == 4
        assert {t.stem for t in tifs} == set(dates)

    @patch("nisar_pytools.processing.prep_dolphin.subprocess.run",
           side_effect=_mock_dolphin_config)
    def test_skip_dates(self, mock_run, gslc_dir, tmp_path):
        _, paths, _ = gslc_dir
        out = tmp_path / "output"
        prep_dolphin(paths, out, skip_dates={"20251210"})
        tifs = sorted((out / "slcs").glob("*.tif"))
        assert len(tifs) == 3
        assert not (out / "slcs" / "20251210.tif").exists()

    @patch("nisar_pytools.processing.prep_dolphin.subprocess.run",
           side_effect=_mock_dolphin_config)
    def test_creates_output_dirs(self, mock_run, gslc_dir, tmp_path):
        _, paths, _ = gslc_dir
        out = tmp_path / "deep" / "nested" / "dir"
        prep_dolphin(paths, out)
        assert out.exists()
        assert (out / "slcs").exists()

    @patch("nisar_pytools.processing.prep_dolphin.subprocess.run",
           side_effect=_mock_dolphin_config)
    def test_skip_existing(self, mock_run, gslc_dir, tmp_path):
        _, paths, _ = gslc_dir
        out = tmp_path / "output"
        # First run creates tifs.
        prep_dolphin(paths, out)
        mtimes = {t.name: t.stat().st_mtime
                  for t in (out / "slcs").glob("*.tif")}
        # Second run should skip all SLC exports.
        prep_dolphin(paths, out)
        for t in (out / "slcs").glob("*.tif"):
            assert t.stat().st_mtime == mtimes[t.name]

    def test_no_valid_files_raises(self, tmp_path):
        out = tmp_path / "output"
        fp = _make_gslc_file(tmp_path, date_str="20251128")
        with pytest.raises(ValueError, match="No valid"):
            prep_dolphin([fp], out, skip_dates={"20251128"})

    @patch("nisar_pytools.processing.prep_dolphin.subprocess.run",
           side_effect=_mock_dolphin_config)
    def test_logs_next_step(self, mock_run, gslc_dir, tmp_path, caplog):
        """Verify the 'Ready. Review the config...' log message."""
        _, paths, _ = gslc_dir
        import logging
        with caplog.at_level(logging.INFO):
            prep_dolphin(paths, tmp_path / "output")
        assert any("dolphin run" in r.message for r in caplog.records)

    @patch("nisar_pytools.processing.prep_dolphin.subprocess.run",
           side_effect=_mock_dolphin_config)
    def test_invalid_h5_skipped(self, mock_run, tmp_path):
        out = tmp_path / "output"
        # Create a valid GSLC and a fake .h5.
        good = _make_gslc_file(tmp_path, date_str="20251128")
        bad = tmp_path / "NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251210T124615_20251210T124650_X05009_N_F_J_001.h5"
        bad.write_text("not an hdf5 file")
        prep_dolphin([good, bad], out)
        tifs = list((out / "slcs").glob("*.tif"))
        assert len(tifs) == 1


# -- _generate_dolphin_config --

class TestDolphinConfig:
    @patch("nisar_pytools.processing.prep_dolphin.subprocess.run",
           side_effect=_mock_dolphin_config)
    def test_config_generated(self, mock_run, tmp_path):
        tifs = [tmp_path / "a.tif", tmp_path / "b.tif"]
        for t in tifs:
            t.touch()
        cfg = _generate_dolphin_config(tifs, tmp_path, tmp_path / "cfg.yaml")
        assert cfg.exists()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "dolphin"

    @patch("nisar_pytools.processing.prep_dolphin.subprocess.run",
           side_effect=_mock_dolphin_config)
    def test_overrides_applied(self, mock_run, tmp_path):
        import yaml
        tifs = [tmp_path / "a.tif"]
        tifs[0].touch()
        overrides = [
            (["phase_linking", "half_window", "x"], 20),
            (["input_options", "wavelength"], 0.235),
        ]
        cfg = _generate_dolphin_config(
            tifs, tmp_path, tmp_path / "cfg.yaml", overrides)
        d = yaml.safe_load(cfg.read_text())
        assert d["phase_linking"]["half_window"]["x"] == 20
        assert d["input_options"]["wavelength"] == 0.235

    @patch("nisar_pytools.processing.prep_dolphin.subprocess.run",
           side_effect=FileNotFoundError("no dolphin"))
    def test_dolphin_not_installed(self, mock_run, tmp_path):
        with pytest.raises(FileNotFoundError, match="dolphin CLI not found"):
            _generate_dolphin_config([], tmp_path, tmp_path / "cfg.yaml")
