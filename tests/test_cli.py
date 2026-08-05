"""Tests for the ``nisar_pytools`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio
import yaml

import nisar_pytools
from nisar_pytools.cli import _RSLC_TO_GUNW_DESCRIPTION, main


def _list_tifs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.tif"))


def test_cli_gslc_default_writes_amplitude(gslc_h5, tmp_path):
    """No --band on a GSLC -> amplitude tif for the first available pol."""
    out_dir = tmp_path / "out"
    main([
        "to-geotiff",
        str(gslc_h5),
        "--output-dir", str(out_dir),
    ])

    tifs = _list_tifs(out_dir)
    assert len(tifs) == 1
    # First pol in the synthetic fixture is HH.
    assert tifs[0].name == f"{gslc_h5.stem}_amplitude_frequencyA_HH.tif"

    with rasterio.open(tifs[0]) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert src.crs is not None


def test_cli_gslc_explicit_pol(gslc_h5, tmp_path):
    """--pol HV should write a single amplitude tif for HV."""
    out_dir = tmp_path / "out"
    main([
        "to-geotiff",
        str(gslc_h5),
        "--band", "amplitude",
        "--pol", "HV",
        "--output-dir", str(out_dir),
    ])

    tifs = _list_tifs(out_dir)
    assert len(tifs) == 1
    assert tifs[0].name == f"{gslc_h5.stem}_amplitude_frequencyA_HV.tif"


def test_cli_gunw_default_writes_all_bands(gunw_h5, tmp_path):
    """No --band on a GUNW -> all four default bands as separate tifs."""
    out_dir = tmp_path / "out"
    main([
        "to-geotiff",
        str(gunw_h5),
        "--output-dir", str(out_dir),
    ])

    tifs = _list_tifs(out_dir)
    names = {p.name for p in tifs}
    expected = {
        f"{gunw_h5.stem}_unwrapped_phase_frequencyA_HH.tif",
        f"{gunw_h5.stem}_wrapped_phase_frequencyA_HH.tif",
        f"{gunw_h5.stem}_coherence_frequencyA_HH.tif",
        f"{gunw_h5.stem}_ionosphere_frequencyA_HH.tif",
    }
    assert names == expected


def test_cli_gunw_single_band(gunw_h5, tmp_path):
    """--band unwrapped_phase writes only that one tif."""
    out_dir = tmp_path / "out"
    main([
        "to-geotiff",
        str(gunw_h5),
        "--band", "unwrapped_phase",
        "--output-dir", str(out_dir),
    ])

    tifs = _list_tifs(out_dir)
    assert len(tifs) == 1
    assert tifs[0].name == f"{gunw_h5.stem}_unwrapped_phase_frequencyA_HH.tif"

    with rasterio.open(tifs[0]) as src:
        # Synthetic GUNW unwrapped grid is 6x8 from conftest.
        assert (src.height, src.width) == (6, 8)
        assert src.crs is not None


def test_cli_gunw_wrapped_phase_is_real(gunw_h5, tmp_path):
    """wrapped_phase comes from np.angle(complex) -> real-valued raster."""
    out_dir = tmp_path / "out"
    main([
        "to-geotiff",
        str(gunw_h5),
        "--band", "wrapped_phase",
        "--output-dir", str(out_dir),
    ])

    tif = _list_tifs(out_dir)[0]
    with rasterio.open(tif) as src:
        assert src.dtypes[0] == "float32"
        # Synthetic wrapped grid is 12x16 from conftest.
        assert (src.height, src.width) == (12, 16)


def test_cli_no_output_dir_writes_next_to_h5(gslc_h5):
    """Without --output-dir, tifs land beside the input h5."""
    main(["to-geotiff", str(gslc_h5)])
    sibling_tifs = sorted(gslc_h5.parent.glob("*.tif"))
    assert len(sibling_tifs) == 1
    assert sibling_tifs[0].parent == gslc_h5.parent


def test_cli_no_args_prints_help(capsys):
    """`nisar_pytools` alone should print top-level help and exit 0."""
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "to-geotiff" in out
    assert "usage:" in out.lower()


def test_cli_invalid_band_for_product_exits(gslc_h5, tmp_path):
    """A GUNW-only band on a GSLC must error out, not silently succeed."""
    with pytest.raises(SystemExit):
        main([
            "to-geotiff",
            str(gslc_h5),
            "--band", "unwrapped_phase",
            "--output-dir", str(tmp_path),
        ])


def test_cli_invalid_pol_exits(gslc_h5, tmp_path):
    """A pol that isn't in the file should error with a clear message."""
    with pytest.raises(SystemExit):
        main([
            "to-geotiff",
            str(gslc_h5),
            "--pol", "VV",
            "--output-dir", str(tmp_path),
        ])


# --- Subsetting (--bbox / --bbox-wgs84) ----------------------------------

# Synthetic GUNW unwrapped grid (per conftest._create_gunw_subproduct):
#   x = 500000 + 100*i for i in 0..7  (-> 500000..500700, ny=6, nx=8)
#   y = 4000000 + 100*j for j in 0..5
# CRS is EPSG:32611 (UTM 11N).
def test_cli_bbox_native_crops_unwrapped(gunw_h5, tmp_path):
    """--bbox in native UTM crops the unwrapped_phase output grid."""
    out_dir = tmp_path / "out"
    # A 3x3 window centered on the synthetic grid.
    main([
        "to-geotiff",
        str(gunw_h5),
        "--band", "unwrapped_phase",
        "--bbox", "500200", "4000200", "500500", "4000500",
        "--output-dir", str(out_dir),
    ])

    tif = _list_tifs(out_dir)[0]
    with rasterio.open(tif) as src:
        # Cropped should be smaller than the full 6x8 grid.
        assert src.height < 6 and src.width < 8
        # Bounds must fall within the requested bbox (within a pixel).
        assert src.bounds.left >= 500100
        assert src.bounds.right <= 500600
        assert src.crs.to_epsg() == 32611


def test_cli_bbox_wgs84_runs(gunw_h5, tmp_path):
    """--bbox-wgs84 reprojects through pyproj and crops to a lat/lon AOI.

    The synthetic grid is positioned at UTM 11N easting 500_000 m,
    northing 4_000_000 m, which is on the central meridian (-117 deg) at
    ~36.12 deg N. Use a bbox wide enough to wholly contain it.
    """
    out_dir = tmp_path / "out"
    main([
        "to-geotiff",
        str(gunw_h5),
        "--band", "unwrapped_phase",
        "--bbox-wgs84", "-118", "36", "-116", "37",
        "--output-dir", str(out_dir),
    ])

    tifs = _list_tifs(out_dir)
    assert len(tifs) == 1
    with rasterio.open(tifs[0]) as src:
        # Reprojected bbox should cover at least some pixels.
        assert src.height > 0 and src.width > 0
        assert src.crs.to_epsg() == 32611


def test_cli_bbox_no_overlap_exits(gunw_h5, tmp_path):
    """A bbox that misses the data should fail with SystemExit."""
    with pytest.raises(SystemExit):
        main([
            "to-geotiff",
            str(gunw_h5),
            "--band", "unwrapped_phase",
            "--bbox", "0", "0", "100", "100",
            "--output-dir", str(tmp_path),
        ])


def test_cli_bbox_and_bbox_wgs84_mutually_exclusive(gunw_h5, tmp_path):
    """Argparse should reject both --bbox and --bbox-wgs84."""
    with pytest.raises(SystemExit):
        main([
            "to-geotiff",
            str(gunw_h5),
            "--bbox", "0", "0", "1", "1",
            "--bbox-wgs84", "0", "0", "1", "1",
            "--output-dir", str(tmp_path),
        ])


# --- info subcommand ----------------------------------------------------

def test_cli_info_gslc_text(gslc_h5, capsys):
    """info on GSLC prints the headline fields users will look for."""
    rc = main(["info", str(gslc_h5)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "GSLC" in out
    assert "track:" in out
    assert "frame:" in out
    assert "Ascending" in out
    assert "frequencyA" in out
    assert "frequencyB" in out
    # GSLC has a single time, not reference/secondary.
    assert "reference:" not in out


def test_cli_info_gunw_text_includes_coherence(gunw_h5, capsys):
    """info on GUNW prints reference/secondary times and coherence stats."""
    rc = main(["info", str(gunw_h5)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "GUNW" in out
    assert "reference:" in out
    assert "secondary:" in out
    assert "temporal baseline:" in out
    assert "coherence" in out
    # The synthetic fixture's coherence is all-zero -> valid stats but min==max==0.
    assert "frequencyA/HH" in out


def test_cli_info_gunw_json(gunw_h5, capsys):
    """--json emits parseable JSON with the expected top-level keys."""
    rc = main(["info", str(gunw_h5), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["product"] == "GUNW"
    assert "acquisition" in payload
    assert "reference_start" in payload["acquisition"]
    # Fixture sets reference=2025-11-03 and secondary=2025-11-15.
    assert payload["acquisition"]["temporal_baseline_days"] == 12
    assert "grids" in payload
    assert "frequencyA" in payload["grids"]
    assert "unwrappedInterferogram" in payload["grids"]["frequencyA"]

    # New stats blocks -- presence + shape, not exact values.
    unw_grid = payload["grids"]["frequencyA"]["unwrappedInterferogram"]
    assert "mask_coverage" in unw_grid
    assert {"valid_count", "total_count", "valid_fraction"} <= unw_grid["mask_coverage"].keys()

    assert "unwrapped_phase" in payload
    assert "frequencyA/HH" in payload["unwrapped_phase"]
    assert {"min", "max", "median"} <= payload["unwrapped_phase"]["frequencyA/HH"].keys()

    assert "connected_components" in payload
    # Synthetic fixture's connectedComponents is all zeros -> no real components,
    # so the dict can be empty (the helper returns None when nothing connects).
    # Still valid; we only require the key exists.

    # Spatial baseline: synthetic fixture now ships pre-computed
    # perpendicular/parallel cubes (see conftest), so the baseline dict
    # must be present with both center and median values.
    assert "baseline" in payload
    assert {
        "perpendicular_center_m",
        "parallel_center_m",
        "perpendicular_median_m",
        "parallel_median_m",
    } <= payload["baseline"].keys()

    # Polarizations are surfaced as a top-level dict, freq -> list.
    assert payload["polarizations"]["frequencyA"] == ["HH"]


def test_cli_output_is_tiled_geotiff(gunw_h5, tmp_path):
    """Output should be a tiled GeoTIFF (streaming-friendly)."""
    out_dir = tmp_path / "out"
    main([
        "to-geotiff",
        str(gunw_h5),
        "--band", "unwrapped_phase",
        "--output-dir", str(out_dir),
    ])
    tif = _list_tifs(out_dir)[0]
    with rasterio.open(tif) as src:
        assert src.is_tiled is True


def test_cli_info_mask_coverage_excludes_fill_value(gslc_h5, capsys):
    """255 is the mask _FillValue (outside the acquisition extent), not valid.

    Real GSLC masks are mostly 255 on the geocoded grid, so counting them
    as valid reports ~97% coverage where the true figure is ~14%.
    """
    with h5py.File(gslc_h5, "r+") as f:
        mask = f["science/LSAR/GSLC/grids/frequencyA/mask"]
        arr = np.full(mask.shape, 255, dtype="u1")
        arr[4:, :] = 1  # in-extent and valid
        arr[4:, :2] = 0  # in-extent but not fully focused
        mask[...] = arr

    rc = main(["info", str(gslc_h5), "--json"])
    assert rc == 0
    mc = json.loads(capsys.readouterr().out)["grids"]["frequencyA"]["mask_coverage"]

    # 8x10 grid: 40 px are 255, of the remaining 40 px 8 are 0 and 32 are 1.
    assert mc["valid_count"] == 32
    assert mc["extent_count"] == 40
    assert mc["total_count"] == 80
    assert mc["valid_fraction"] == pytest.approx(32 / 80)
    assert mc["valid_fraction_in_extent"] == pytest.approx(32 / 40)


def test_cli_freq_in_output_filename(gslc_h5, tmp_path):
    """--freq A and --freq B into one directory must not overwrite each other."""
    out_dir = tmp_path / "out"
    for freq in ("A", "B"):
        main([
            "to-geotiff",
            str(gslc_h5),
            "--output-dir", str(out_dir),
            "--freq", freq,
        ])

    tifs = _list_tifs(out_dir)
    assert len(tifs) == 2
    assert {p.name for p in tifs} == {
        f"{gslc_h5.stem}_amplitude_frequencyA_HH.tif",
        f"{gslc_h5.stem}_amplitude_frequencyB_HH.tif",
    }
    # frequencyB is half as wide in the fixture -- proves they are distinct grids.
    with rasterio.open(out_dir / f"{gslc_h5.stem}_amplitude_frequencyB_HH.tif") as src:
        assert src.width == 5


def test_bundled_runconfig_uses_current_release_id():
    """The bundled runconfig and its help text must name the current release."""
    runconfig = (
        Path(nisar_pytools.__file__).parent
        / "processing"
        / "isce3_runconfig_default.yaml"
    )
    cfg = yaml.safe_load(runconfig.read_text())
    # Dump back out so the comment header (which records the X05010 origin)
    # is dropped and only the settings isce3 actually reads are checked.
    assert "X05010" not in yaml.safe_dump(cfg)
    assert "composite_release_id: P05023" in yaml.safe_dump(cfg)

    assert "P05023" in _RSLC_TO_GUNW_DESCRIPTION
    assert "X05010" not in _RSLC_TO_GUNW_DESCRIPTION


def test_cli_crop_rslc_requires_bbox_and_epsg(tmp_path):
    """--crop-rslc without --bbox/--epsg should exit before any processing."""
    ref = tmp_path / "ref.h5"
    sec = tmp_path / "sec.h5"
    ref.touch()
    sec.touch()
    with pytest.raises(SystemExit):
        main([
            "rslc-to-gunw",
            str(ref),
            str(sec),
            "--output-dir", str(tmp_path / "out"),
            "--crop-rslc",
        ])
