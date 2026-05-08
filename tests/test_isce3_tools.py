"""Tests for nisar_pytools.processing.isce3_tools.

Smoke tests only -- the actual workflow run is multi-hour and requires
isce3 + a real RSLC pair; that's covered by an integration script
(scripts/isce3/run_insar.sh), not by pytest.
"""

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
