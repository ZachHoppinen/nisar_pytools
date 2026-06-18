"""Tests for nisar_pytools.processing.isce3_tools.

Smoke tests only -- the actual workflow run is multi-hour and requires
isce3 + a real RSLC pair; that's covered by an integration script
(scripts/isce3/run_insar.sh), not by pytest.
"""

import pytest
import yaml

from nisar_pytools.processing.isce3_tools import (
    _DEFAULT_RUNCONFIG,
    _bbox_from_polygon_in_epsg,
    _deep_merge,
    _load_default_runconfig,
    _utm_epsg_from_polygon,
)


class TestDefaultRunconfig:
    def test_file_is_packaged(self):
        assert _DEFAULT_RUNCONFIG.exists()

    def test_loads_as_yaml(self):
        cfg = _load_default_runconfig()
        assert "runconfig" in cfg
        groups = cfg["runconfig"]["groups"]
        assert "input_file_group" in groups
        assert "processing" in groups
        # Path fields are null until rslc_to_gunw fills them in
        assert groups["input_file_group"]["reference_rslc_file"] is None
        assert groups["input_file_group"]["secondary_rslc_file"] is None

    def test_production_settings(self):
        cfg = _load_default_runconfig()
        proc = cfg["runconfig"]["groups"]["processing"]
        # JPL production looks: 5x6 crossmul, 13x16 unwrap
        assert proc["crossmul"]["range_looks"] == 5
        assert proc["crossmul"]["azimuth_looks"] == 6
        assert proc["phase_unwrap"]["range_looks"] == 13
        assert proc["phase_unwrap"]["azimuth_looks"] == 16
        # Coregistration on
        assert proc["dense_offsets"]["enabled"] is True
        assert proc["rubbersheet"]["enabled"] is True
        assert proc["fine_resample"]["enabled"] is True
        # Ionosphere split-spectrum on, troposphere off (no ECMWF locally)
        assert proc["ionosphere_phase_correction"]["enabled"] is True
        assert proc["troposphere_delay"]["enabled"] is False


class TestDeepMerge:
    def test_top_level_override_wins(self):
        out = _deep_merge({"a": 1, "b": 2}, {"a": 99})
        assert out == {"a": 99, "b": 2}

    def test_nested_merge(self):
        base = {"x": {"y": 1, "z": 2}}
        out = _deep_merge(base, {"x": {"y": 99}})
        assert out == {"x": {"y": 99, "z": 2}}

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}

    def test_overrides_replace_non_dict_with_dict(self):
        out = _deep_merge({"a": 1}, {"a": {"nested": 2}})
        assert out == {"a": {"nested": 2}}


class TestUtmHelpers:
    def test_utm_north(self):
        from shapely.geometry import box
        # Western US, 45N, 117W -> UTM 11N -> EPSG 32611
        poly = box(-118, 44, -116, 46)
        assert _utm_epsg_from_polygon(poly) == 32611

    def test_utm_south(self):
        from shapely.geometry import box
        # South America, 30S, 70W -> UTM 19S -> EPSG 32719
        poly = box(-72, -32, -70, -28)
        assert _utm_epsg_from_polygon(poly) == 32719

    def test_bbox_projection(self):
        from shapely.geometry import box
        poly = box(-118.0, 44.5, -117.0, 45.5)  # 1° box in west OR
        xmin, ymin, xmax, ymax = _bbox_from_polygon_in_epsg(poly, 32611)
        # Should project to roughly UTM 11N coords near (~500000, ~5000000)
        assert 400000 < xmin < 600000
        assert 4900000 < ymin < 5100000
        assert xmax > xmin
        assert ymax > ymin


class TestRunconfigInjection:
    """Verify rslc_to_gunw injects paths/bbox/EPSG correctly without running isce3."""

    def test_injection_via_inspection(self, tmp_path, monkeypatch):
        """Stub out the workflow imports + DEM fetch + RSLC reading so we
        can inspect the merged runconfig that rslc_to_gunw would feed to
        nisar.workflows.insar."""
        from unittest.mock import MagicMock

        from nisar_pytools.processing import isce3_tools

        ref = tmp_path / "ref.h5"
        sec = tmp_path / "sec.h5"
        ref.touch()
        sec.touch()
        out_dir = tmp_path / "out"

        # Stub: skip DEM auto-fetch (provide explicit dem_file)
        dem = tmp_path / "dem.tif"
        dem.touch()

        # Stub the workflow imports so we don't need isce3 in the test env.
        fake_insar = MagicMock()
        fake_runcfg_cls = MagicMock()
        fake_runcfg_inst = MagicMock()
        fake_runcfg_inst.cfg = {"logging": {"path": str(out_dir / "scratch/insar.log")}}
        fake_runcfg_cls.return_value = fake_runcfg_inst
        fake_persistence_cls = MagicMock()
        fake_persistence_inst = MagicMock()
        fake_persistence_inst.run = True
        fake_persistence_cls.return_value = fake_persistence_inst
        fake_h5_prep = MagicMock()
        fake_h5_prep.get_products_and_paths.return_value = (None, {"GUNW": str(out_dir / "product.h5")})

        # Stub insar.run to also write the expected output product
        def fake_insar_run(cfg, out_paths, run_steps):
            (out_dir / "product.h5").touch()
        fake_insar.run = fake_insar_run

        def fake_import_workflow():
            return fake_insar, fake_runcfg_cls, fake_persistence_cls, fake_h5_prep

        monkeypatch.setattr(isce3_tools, "_import_insar_workflow", fake_import_workflow)

        gunw = isce3_tools.rslc_to_gunw(
            ref, sec, out_dir,
            dem_file=dem,
            aoi_bbox_utm=(100000.0, 4000000.0, 200000.0, 4100000.0),
            output_epsg=32611,
            overrides={"runconfig": {"groups": {"processing": {
                "crossmul": {"range_looks": 10}
            }}}},
        )

        assert gunw == out_dir / "product.h5"

        # Read back the written runconfig and verify everything was injected
        with open(out_dir / "runconfig.yaml") as f:
            cfg = yaml.safe_load(f)
        g = cfg["runconfig"]["groups"]
        assert g["input_file_group"]["reference_rslc_file"] == str(ref.resolve())
        assert g["input_file_group"]["secondary_rslc_file"] == str(sec.resolve())
        assert g["dynamic_ancillary_file_group"]["dem_file"] == str(dem.resolve())
        assert g["processing"]["geocode"]["output_epsg"] == 32611
        assert g["processing"]["geocode"]["top_left"] == {"x_abs": 100000.0, "y_abs": 4100000.0}
        assert g["processing"]["geocode"]["bottom_right"] == {"x_abs": 200000.0, "y_abs": 4000000.0}
        # Override merged in
        assert g["processing"]["crossmul"]["range_looks"] == 10
        # Untouched defaults preserved
        assert g["processing"]["crossmul"]["azimuth_looks"] == 6


def _run_rslc_to_gunw_mocked(tmp_path, monkeypatch, **kwargs):
    """Run rslc_to_gunw with isce3 + DEM fetch stubbed out, then return the
    runconfig dict it wrote to disk. Lets tests inspect injection without a
    real (multi-hour, isce3-dependent) workflow run."""
    from unittest.mock import MagicMock

    from nisar_pytools.processing import isce3_tools

    ref = tmp_path / "ref.h5"
    sec = tmp_path / "sec.h5"
    dem = tmp_path / "dem.tif"
    for f in (ref, sec, dem):
        f.touch()
    out_dir = tmp_path / "out"

    # Stub the workflow imports so the test env doesn't need isce3 installed.
    fake_insar = MagicMock()
    fake_runcfg_inst = MagicMock()
    fake_runcfg_inst.cfg = {"logging": {"path": str(out_dir / "scratch/insar.log")}}
    fake_runcfg_cls = MagicMock(return_value=fake_runcfg_inst)
    fake_persistence_inst = MagicMock(run=True)
    fake_persistence_cls = MagicMock(return_value=fake_persistence_inst)
    fake_h5_prep = MagicMock()
    fake_h5_prep.get_products_and_paths.return_value = (None, {"GUNW": str(out_dir / "product.h5")})
    # insar.run writes the expected output so rslc_to_gunw's existence check passes.
    fake_insar.run = lambda cfg, out_paths, run_steps: (out_dir / "product.h5").touch()

    monkeypatch.setattr(
        isce3_tools, "_import_insar_workflow",
        lambda: (fake_insar, fake_runcfg_cls, fake_persistence_cls, fake_h5_prep),
    )

    isce3_tools.rslc_to_gunw(ref, sec, out_dir, dem_file=dem, **kwargs)
    with open(out_dir / "runconfig.yaml") as f:
        return yaml.safe_load(f)


class TestRadarGridCubes:
    """The radar_grid_cubes group must be geocoded onto the same AOI extent
    and EPSG as the interferogram, so the geometry/metadata cubes (slant
    range, incidence angle, LOS vectors, baseline, ...) co-register with the
    unwrapped phase instead of falling back to isce3's full-scene default."""

    def test_cubes_match_geocode_extent_and_epsg(self, tmp_path, monkeypatch):
        bbox = (100000.0, 4000000.0, 200000.0, 4100000.0)  # xmin, ymin, xmax, ymax
        cfg = _run_rslc_to_gunw_mocked(
            tmp_path, monkeypatch, aoi_bbox_utm=bbox, output_epsg=32611,
        )
        proc = cfg["runconfig"]["groups"]["processing"]
        geocode = proc["geocode"]
        cubes = proc["radar_grid_cubes"]

        # EPSG and extent are injected (not left null) ...
        assert cubes["output_epsg"] == 32611
        assert cubes["top_left"] == {"x_abs": 100000.0, "y_abs": 4100000.0}
        assert cubes["bottom_right"] == {"x_abs": 200000.0, "y_abs": 4000000.0}
        # ... and exactly match the geocode grid the interferogram uses.
        assert cubes["output_epsg"] == geocode["output_epsg"]
        assert cubes["top_left"] == geocode["top_left"]
        assert cubes["bottom_right"] == geocode["bottom_right"]


class TestCrop:
    """crop=True should crop the RSLC pair first and run the workflow on the
    cropped files. The actual crop is mocked here (no real RSLC needed)."""

    def test_crop_requires_bbox_and_epsg(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from nisar_pytools.processing import isce3_tools

        ref, sec, dem = tmp_path / "ref.h5", tmp_path / "sec.h5", tmp_path / "d.tif"
        for f in (ref, sec, dem):
            f.touch()
        monkeypatch.setattr(
            isce3_tools, "_import_insar_workflow",
            lambda: (MagicMock(), MagicMock(), MagicMock(), MagicMock()),
        )
        with pytest.raises(ValueError, match="crop=True requires"):
            isce3_tools.rslc_to_gunw(
                ref, sec, tmp_path / "out", dem_file=dem, crop=True,
            )  # no aoi_bbox_utm / output_epsg

    def test_crop_runs_workflow_on_cropped_pair(self, tmp_path, monkeypatch):
        from nisar_pytools.processing import crop_rslc

        captured = {}

        def fake_crop_pair(ref, sec, bbox, epsg, dem, out_dir, margin, min_size):
            captured["margin"] = margin
            captured["min_size"] = min_size
            out_dir.mkdir(parents=True, exist_ok=True)
            rsub = out_dir / "ref_sub.h5"
            ssub = out_dir / "sec_sub.h5"
            rsub.touch()
            ssub.touch()
            return rsub, ssub

        monkeypatch.setattr(crop_rslc, "crop_rslc_pair", fake_crop_pair)

        cfg = _run_rslc_to_gunw_mocked(
            tmp_path, monkeypatch,
            aoi_bbox_utm=(100000.0, 4000000.0, 200000.0, 4100000.0),
            output_epsg=32611, crop=True, crop_margin=128, crop_min_size=1024,
        )
        g = cfg["runconfig"]["groups"]
        # The workflow must run on the CROPPED files, not the originals.
        assert g["input_file_group"]["reference_rslc_file"].endswith("ref_sub.h5")
        assert g["input_file_group"]["secondary_rslc_file"].endswith("sec_sub.h5")
        assert captured["margin"] == 128
        assert captured["min_size"] == 1024


class TestImportError:
    def test_helpful_message_when_workflow_missing(self, monkeypatch):
        """Verify the import-error wrapping gives users actionable instructions."""
        import builtins
        from nisar_pytools.processing import isce3_tools

        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name.startswith("nisar.workflows"):
                raise ImportError("No module named 'nisar.workflows'")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        try:
            isce3_tools._import_insar_workflow()
        except ImportError as e:
            msg = str(e)
            assert "isce3" in msg
            assert "mamba" in msg or "pip" in msg
        else:
            raise AssertionError("expected ImportError")
