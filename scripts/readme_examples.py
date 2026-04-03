"""Verify all README code examples work against local test files.

Runs each section from the README (except polsar which needs quad-pol)
on small subsets to confirm the API works end-to-end with data quality checks.
"""

import numpy as np

GSLC1 = "local/gslcs/NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001.h5"
GSLC2 = "local/gslcs/NISAR_L2_PR_GSLC_005_077_A_024_4005_DHDH_A_20251115T124615_20251115T124650_X05009_N_F_J_001.h5"
GUNW = "local/gunws/NISAR_L2_PR_GUNW_004_077_A_024_005_4000_SH_20251103T124615_20251103T124650_20251115T124615_20251115T124650_X05010_N_F_J_001.h5"


def section(name):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")


def check_not_all_nan(arr, name):
    vals = np.asarray(arr)
    nan_frac = np.isnan(vals).sum() / vals.size if np.issubdtype(vals.dtype, np.floating) else 0
    assert nan_frac < 0.5, f"{name} is >{nan_frac*100:.0f}% NaN"
    return nan_frac


def check_finite(arr, name):
    vals = np.asarray(arr)
    if np.iscomplexobj(vals):
        assert np.any(np.abs(vals) > 0), f"{name} is all zero"
        assert np.all(np.isfinite(vals[vals != 0])), f"{name} has non-finite values"
    else:
        finite = np.isfinite(vals)
        assert finite.sum() > vals.size * 0.5, f"{name} has <50% finite values"


def test_search():
    section("Search and Download")
    from nisar_pytools import find_nisar

    all_gslcs = find_nisar(
        aoi=[-115, 43, -114, 44],
        start_date="2025-11-01",
        end_date="2025-11-30",
        product_type="GSLC",
    )
    print(f"  Broad search: {len(all_gslcs)} URLs")
    assert len(all_gslcs) >= 2, f"Expected >=2 URLs, got {len(all_gslcs)}"
    assert all(u.endswith(".h5") for u in all_gslcs), "Non-.h5 URLs found"
    assert not any("_QA_" in u for u in all_gslcs), "QA files not filtered"

    track_77 = find_nisar(
        aoi=[-115, 43, -114, 44],
        start_date="2025-11-01",
        end_date="2025-11-30",
        product_type="GSLC",
        path_number=77,
        direction="ASCENDING",
    )
    print(f"  Track 77 search: {len(track_77)} URLs")
    assert len(track_77) >= 2
    assert len(track_77) <= len(all_gslcs), "Narrow search returned more than broad"

    gunws = find_nisar(
        aoi=[-115, 43, -114, 44],
        start_date="2025-11-01",
        end_date="2026-02-01",
        product_type="GUNW",
    )
    print(f"  GUNW search: {len(gunws)} URLs")
    assert len(gunws) >= 1
    print("  PASS")


def test_read():
    section("Read a Single File")
    from nisar_pytools import open_nisar
    from nisar_pytools.utils.metadata import get_slc, get_orbit_info, get_acquisition_time

    dt = open_nisar(GSLC1)

    hh = get_slc(dt, polarization="HH")
    hv = get_slc(dt, polarization="HV")
    hh_b = get_slc(dt, polarization="HH", frequency="frequencyB")

    print(f"  HH: {hh.shape}, CRS={hh.rio.crs}")
    print(f"  HV: {hv.shape}")
    print(f"  HH freq B: {hh_b.shape}")

    assert hh.rio.crs is not None, "CRS not set"
    assert hh.shape[0] > 1000, f"HH too small: {hh.shape}"
    assert np.iscomplexobj(hh), "HH not complex"

    # Check a small subset has actual data (not all zero/nan)
    subset = hh.isel(y=slice(35000, 35064), x=slice(36000, 36064)).compute()
    check_finite(subset.values, "HH subset")
    assert np.mean(np.abs(subset.values)) > 0.001, "HH amplitude suspiciously low"
    print(f"  HH subset amplitude mean: {np.mean(np.abs(subset.values)):.4f}")

    ts = get_acquisition_time(dt)
    info = get_orbit_info(dt)
    print(f"  Acquisition: {ts}")
    print(f"  Orbit: {info}")
    assert info["track_number"] == 77
    assert info["orbit_direction"] == "Ascending"
    assert ts.year == 2025
    print("  PASS")


def test_stack():
    section("Stack GSLCs")
    from nisar_pytools import stack_gslcs

    stack = stack_gslcs([GSLC1, GSLC2], frequency="frequencyA", polarization="HH")
    print(f"  Stack: {stack.shape}, dims={stack.dims}")
    print(f"  Times: {stack.time.values}")
    print(f"  CRS: {stack.rio.crs}")

    assert stack.shape[0] == 2
    assert stack.rio.crs is not None
    assert stack.time.values[0] < stack.time.values[1], "Times not sorted"

    # Verify both time steps have data
    for t in range(2):
        subset = stack.isel(time=t, y=slice(35000, 35032), x=slice(36000, 36032)).compute()
        check_finite(subset.values, f"Stack time={t}")
    print("  Both time steps have valid data")
    print("  PASS")


def test_sar_processing():
    section("SAR Processing")
    from nisar_pytools import stack_gslcs
    from nisar_pytools.processing import (
        interferogram, coherence, multilook, unwrap, calculate_phase
    )

    stack = stack_gslcs([GSLC1, GSLC2], polarization="HH")
    slc1 = stack.isel(time=0, y=slice(35000, 35256), x=slice(36000, 36256)).compute()
    slc2 = stack.isel(time=1, y=slice(35000, 35256), x=slice(36000, 36256)).compute()
    print(f"  Subset: {slc1.shape}")

    # Interferogram
    ifg = interferogram(slc1, slc2)
    assert np.iscomplexobj(ifg.values)
    check_finite(ifg.values, "interferogram")
    ifg_amp = np.abs(ifg.values)
    assert np.mean(ifg_amp) > 0, "Interferogram amplitude is zero"
    print(f"  Interferogram: mean amplitude={np.mean(ifg_amp):.4f}")

    # Multilook
    ml_ifg = multilook(ifg, looks_y=4, looks_x=4)
    assert ml_ifg.shape == (64, 64)
    check_finite(ml_ifg.values, "multilooked ifg")
    print(f"  Multilooked: {ml_ifg.shape}")

    # Coherence
    coh_box = coherence(slc1, slc2, window_size=11)
    coh_gau = coherence(slc1, slc2, window_size=3, method="gaussian")
    for name, coh in [("boxcar", coh_box), ("gaussian", coh_gau)]:
        vals = coh.values
        nan_frac = check_not_all_nan(vals, f"coherence {name}")
        assert np.nanmin(vals) >= 0, f"Coherence {name} has negative values"
        assert np.nanmax(vals) <= 1, f"Coherence {name} exceeds 1"
        assert np.nanmean(vals) > 0.1, f"Coherence {name} suspiciously low: {np.nanmean(vals):.3f}"
        print(f"  Coherence {name}: mean={np.nanmean(vals):.3f}, range=[{np.nanmin(vals):.3f}, {np.nanmax(vals):.3f}], NaN={nan_frac*100:.1f}%")

    # Phase
    phase = calculate_phase(ifg)
    check_not_all_nan(phase.values, "phase")
    assert float(phase.min()) >= -np.pi - 0.01, "Phase below -pi"
    assert float(phase.max()) <= np.pi + 0.01, "Phase above pi"
    print(f"  Phase: range=[{float(phase.min()):.2f}, {float(phase.max()):.2f}] rad")

    # Unwrap
    ml_coh = multilook(coherence(slc1, slc2, window_size=21), looks_y=4, looks_x=4)
    unw, conncomp = unwrap(ml_ifg, ml_coh, nlooks=16.0)
    check_not_all_nan(unw.values, "unwrapped phase")
    assert int(conncomp.max()) >= 1, "No connected components found"
    print(f"  Unwrapped: range=[{float(unw.min()):.2f}, {float(unw.max()):.2f}] rad, components={int(conncomp.max())}")

    print("  PASS")


def test_phase_linking():
    section("Phase Linking")
    from nisar_pytools import stack_gslcs
    from nisar_pytools.processing import phase_link

    stack = stack_gslcs([GSLC1, GSLC2], polarization="HH")
    subset = stack.isel(y=slice(35000, 35008), x=slice(36000, 36008)).compute()
    print(f"  Subset: {subset.shape}")

    linked, temporal_coh = phase_link(subset, search_window=3, confidence=0.95)
    assert linked.shape == subset.shape
    assert np.iscomplexobj(linked.values)

    # Linked output should have nonzero amplitude
    linked_amp = np.abs(linked.values)
    assert np.mean(linked_amp) > 0, "Linked amplitude is all zero"
    print(f"  Linked amplitude mean: {np.mean(linked_amp):.4f}")

    # Temporal coherence: abs(C[0]) stores coherence of first image with all others
    # Diagonal is always 1.0 (self-coherence), so values up to 1.0 are expected
    coh_vals = temporal_coh.values
    valid_coh = coh_vals[coh_vals > 0]
    if len(valid_coh) > 0:
        assert np.all(valid_coh >= 0), "Coherence has negative values"
        assert np.all(valid_coh <= 1.01), f"Coherence exceeds 1: max={np.max(valid_coh)}"
        print(f"  Temporal coherence: mean={np.mean(valid_coh):.3f}, max={np.max(valid_coh):.3f}")
    else:
        print("  Temporal coherence: all zero (small subset)")

    print("  PASS")


def test_local_incidence_angle():
    section("Local Incidence Angle")
    import xarray as xr
    import rioxarray  # noqa: F401
    from nisar_pytools import open_nisar
    from nisar_pytools.utils.local_incidence_angle import local_incidence_angle

    dt = open_nisar(GSLC1)
    rg = dt["science/LSAR/GSLC/metadata/radarGrid"].dataset

    los_x = np.asarray(rg["losUnitVectorX"])
    los_y = np.asarray(rg["losUnitVectorY"])
    los_z = np.sqrt(np.maximum(1.0 - los_x**2 - los_y**2, 0.0))
    heights = np.asarray(rg.coords["z"])
    x_rg = np.asarray(rg.coords["x"])
    y_rg = np.asarray(rg.coords["y"])
    epsg = int(rg.attrs.get("projection"))

    dem = xr.DataArray(
        np.full((len(y_rg), len(x_rg)), 1500.0, dtype=np.float32),
        dims=["y", "x"], coords={"y": y_rg, "x": x_rg},
    ).rio.write_crs(epsg)

    lia = local_incidence_angle(dem, los_x, los_y, los_z, heights, x_rg, y_rg, epsg=epsg)

    nan_frac = check_not_all_nan(lia.values, "LIA")
    lia_valid = lia.values[np.isfinite(lia.values)]
    assert np.min(lia_valid) > 0, f"LIA has values <= 0: {np.min(lia_valid)}"
    assert np.max(lia_valid) < 90, f"LIA has values >= 90: {np.max(lia_valid)}"
    assert 20 < np.mean(lia_valid) < 60, f"LIA mean outside expected range: {np.mean(lia_valid):.1f}"
    print(f"  LIA: range=[{np.min(lia_valid):.1f}, {np.max(lia_valid):.1f}] deg, mean={np.mean(lia_valid):.1f}, NaN={nan_frac*100:.1f}%")
    assert lia.rio.crs is not None, "CRS not set"
    print("  PASS")


def test_visualization():
    section("Visualization")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from nisar_pytools import stack_gslcs
    from nisar_pytools.processing import interferogram, coherence
    from nisar_pytools.viz import plot_amplitude, plot_phase, plot_interferogram, plot_coherence

    stack = stack_gslcs([GSLC1, GSLC2], polarization="HH")
    slc1 = stack.isel(time=0, y=slice(35000, 35128), x=slice(36000, 36128)).compute()
    slc2 = stack.isel(time=1, y=slice(35000, 35128), x=slice(36000, 36128)).compute()

    ifg = interferogram(slc1, slc2)
    coh = coherence(slc1, slc2, window_size=11)

    plots = [
        ("amplitude", plot_amplitude(slc1)),
        ("interferogram", plot_interferogram(ifg)),
        ("coherence", plot_coherence(coh)),
        ("phase", plot_phase(ifg)),
    ]
    for name, fig in plots:
        assert isinstance(fig, Figure), f"{name} plot is not a Figure"
        # Check the plot has actual rendered content (axes with images)
        axes = fig.get_axes()
        assert len(axes) >= 1, f"{name} plot has no axes"
        plt.close(fig)

    print(f"  Generated {len(plots)} plots with rendered content")
    print("  PASS")


def test_export():
    section("Export")
    from nisar_pytools import stack_gslcs
    from nisar_pytools.processing import coherence, interferogram
    from nisar_pytools.io.export import to_netcdf, to_zarr, read_netcdf

    stack = stack_gslcs([GSLC1, GSLC2], polarization="HH")
    slc1 = stack.isel(time=0, y=slice(35000, 35064), x=slice(36000, 36064)).compute()
    slc2 = stack.isel(time=1, y=slice(35000, 35064), x=slice(36000, 36064)).compute()
    coh = coherence(slc1, slc2, window_size=5)
    ifg = interferogram(slc1, slc2)

    # NetCDF with real data
    nc_path = to_netcdf(coh, "local/test_output/readme_coh.nc")
    loaded = read_netcdf(nc_path)
    assert "coherence" in loaded
    np.testing.assert_allclose(loaded["coherence"].values, coh.values, atol=1e-5)
    print(f"  NetCDF roundtrip: values match, {nc_path.stat().st_size} bytes")

    # NetCDF with complex data (split real/imag)
    nc_ifg_path = to_netcdf(ifg, "local/test_output/readme_ifg.nc")
    loaded_ifg = read_netcdf(nc_ifg_path)
    assert "interferogram" in loaded_ifg
    assert np.iscomplexobj(loaded_ifg["interferogram"].values)
    np.testing.assert_allclose(loaded_ifg["interferogram"].values, ifg.values, atol=1e-5)
    print("  NetCDF complex roundtrip: values match")

    # Zarr
    try:
        import zarr  # noqa: F401
        zarr_path = to_zarr(coh.to_dataset(name="coherence"), "local/test_output/readme_coh.zarr")
        import xarray as xr
        loaded_zarr = xr.open_zarr(zarr_path)
        np.testing.assert_allclose(loaded_zarr["coherence"].values, coh.values, atol=1e-5)
        print("  Zarr roundtrip: values match")
    except ImportError:
        print("  Zarr: skipped (not installed)")

    print("  PASS")


def test_baseline():
    section("Baseline")
    from nisar_pytools import open_nisar
    from nisar_pytools.processing.baseline import compute_baseline

    gslc1 = open_nisar(GSLC1)
    gslc2 = open_nisar(GSLC2)

    baselines = compute_baseline(gslc2, gslc1)
    bperp = baselines["perpendicular_baseline"].values
    bpar = baselines["parallel_baseline"].values

    check_not_all_nan(bperp, "B_perp")
    check_not_all_nan(bpar, "B_par")

    bperp_mean = float(np.nanmean(bperp))
    bpar_mean = float(np.nanmean(bpar))
    print(f"  B_perp: mean={bperp_mean:.1f} m, range=[{np.nanmin(bperp):.1f}, {np.nanmax(bperp):.1f}]")
    print(f"  B_par:  mean={bpar_mean:.1f} m, range=[{np.nanmin(bpar):.1f}, {np.nanmax(bpar):.1f}]")

    # Known values from GUNW validation
    assert 50 < abs(bperp_mean) < 200, f"B_perp mean outside expected range: {bperp_mean}"
    assert 10 < abs(bpar_mean) < 100, f"B_par mean outside expected range: {bpar_mean}"
    assert np.std(bperp[np.isfinite(bperp)]) < 50, "B_perp has excessive spatial variation"
    print("  PASS")


def test_metadata():
    section("Metadata & Utilities")
    from nisar_pytools.utils.filename import parse_filename
    from nisar_pytools import open_nisar
    from nisar_pytools.utils.metadata import get_slc
    from nisar_pytools.utils.conversion import to_db, from_db

    # Filename parsing
    info = parse_filename(GSLC1)
    assert info.track == 77
    assert info.frame == 24
    assert info.product_type == "GSLC"
    assert info.direction == "Ascending"
    assert not info.is_qa
    print(f"  Parsed: track={info.track}, frame={info.frame}, type={info.product_type}, dir={info.direction}")

    # dB conversion roundtrip
    dt = open_nisar(GSLC1)
    hh = get_slc(dt, "HH").isel(y=slice(35000, 35064), x=slice(36000, 36064)).compute()
    amp = np.abs(hh)
    assert float(amp.mean()) > 0, "Amplitude is zero"

    db = to_db(amp, power=False)
    assert db.attrs["units"] == "dB"
    assert float(db.mean()) < 50, f"dB values suspiciously high: {float(db.mean())}"
    assert float(db.mean()) > -100, f"dB values suspiciously low: {float(db.mean())}"
    print(f"  dB range: [{float(db.min()):.1f}, {float(db.max()):.1f}]")

    back = from_db(db, power=False)
    np.testing.assert_allclose(back.values, amp.values, rtol=1e-5)
    print(f"  dB roundtrip: max relative error = {np.max(np.abs(back.values - amp.values) / (amp.values + 1e-10)):.2e}")
    print("  PASS")


if __name__ == "__main__":
    tests = [
        test_search,
        test_read,
        test_stack,
        test_sar_processing,
        test_phase_linking,
        test_local_incidence_angle,
        test_visualization,
        test_export,
        test_baseline,
        test_metadata,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  {passed} passed, {failed} failed out of {len(tests)} sections")
    print(f"{'=' * 60}")
