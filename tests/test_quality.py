"""Tests for nisar_pytools.utils.quality on tiny synthetic RSLC, QA_STATS and GUNW files.

QA_STATS download and the L0B stamp check need the network and are not tested here.
"""

import h5py
import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from nisar_pytools.utils import identify_poor_gunw, identify_poorly_focused_rslc
from nisar_pytools.utils import quality

N = 256  # RSLC side; LOOKS=64 gives a 4 x 4 amplitude


def _write_rslc(path, seed, prf=1909.62, isr=0.03, noise=0.003, smear=1, zero_lines=None, empty_lines=None, exception_lines=None, window=(8, N - 8), last_line=N,
                crid="P05023", diagnostic=0, exception_code=4, validity_bits=0):
    """Synthetic RSLC: shared ground reflectivity times per-date speckle, optionally smeared in azimuth."""
    # reflectivity varies at the 64-pixel look scale and is common to every date, like real ground
    ground = np.kron(np.random.default_rng(0).random((N // 64, N // 64)) * 9 + 1, np.ones((64, 64)))
    rng = np.random.default_rng(seed)
    slc = np.sqrt(ground) * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    if smear > 1:
        slc = uniform_filter1d(slc.real, smear, axis=0) + 1j * uniform_filter1d(slc.imag, smear, axis=0)
    # valid samples 8..248 on every line; empty_lines get an empty window, zero_lines zero data
    windows = np.tile(window, (N, 1))
    windows[last_line:] = 0                # no data after last_line
    slc[last_line:] = 0
    if empty_lines is not None:
        windows[empty_lines] = 0
        slc[empty_lines] = 0
    if zero_lines is not None:
        slc[zero_lines] = 0
    with h5py.File(path, "w") as f:
        f["science/LSAR/identification/zeroDopplerStartTime"] = f"2026-01-{seed:02d}T12:00:00".encode()
        f["science/LSAR/identification/compositeReleaseId"] = crid.encode()
        f["science/LSAR/identification/diagnosticModeFlag"] = np.uint32(diagnostic)
        sw = f.create_group("science/LSAR/RSLC/swaths/frequencyA")
        sw["HH"] = slc.astype("c8")
        sw["nominalAcquisitionPRF"] = prf
        sw["validSamplesSubSwath1"] = windows
        # newer isce3 writes uint16 with validity bit 8 (HH focused) set on every valid pixel
        exc = np.full((N, N), validity_bits, "u2" if validity_bits else "u1")
        if exception_lines is not None:
            exc[exception_lines] |= exception_code
        sw["inputDataExceptionMask"] = exc
        f["science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing"] = 0.000658
        sw["sceneCenterGroundRangeSpacing"] = 242e3 / 240
        sw["sceneCenterAlongTrackSpacing"] = 4.45
        meta = "science/LSAR/RSLC/metadata"
        f[f"{meta}/RFI/frequencyA/HH/raw1/interferenceSignalRatio"] = np.full((5, 5), isr, "f4")
        f[f"{meta}/RFI/frequencyA/HH/raw1/hitCount"] = np.zeros((5, 5, 8), "f4")
        f[f"{meta}/calibrationInformation/frequencyA/noiseEquivalentBackscatter/HH"] = np.full((2, 8), noise, "f4")
        f[f"{meta}/processingInformation/inputs/l0bGranules"] = np.array([b"L0B_GRANULE.h5"])
    return path


@pytest.fixture
def stack(tmp_path):
    return [_write_rslc(tmp_path / f"ref{s}.h5", s) for s in (1, 2, 3)]


def _check(path, refs=None):
    return identify_poorly_focused_rslc(path, refs, qa_stats=False)


def test_clean_scene_has_no_flags(tmp_path, stack):
    out = _check(_write_rslc(tmp_path / "t.h5", 4), stack)
    assert out["flags"] == [] and not out["poorly_focused"]


def test_rfi_flagged_without_references(tmp_path):
    out = _check(_write_rslc(tmp_path / "t.h5", 4, isr=1.7))
    assert out["flags"] == ["RFI"]


def test_prf_and_noise_need_references(tmp_path, stack):
    path = _write_rslc(tmp_path / "t.h5", 4, prf=1939.3, noise=0.1)
    assert _check(path)["flags"] == []
    assert {"PRF", "NOISE"} <= set(_check(path, stack)["flags"])


def test_azimuth_smear_flags_focus(tmp_path, stack):
    out = _check(_write_rslc(tmp_path / "t.h5", 4, smear=3), stack)
    assert "FOCUS_ACF" in out["flags"]


def test_references_accept_metric_dicts(tmp_path, stack):
    refs = [quality.rslc_quality_metrics(p, qa_stats=False) for p in stack]
    path = _write_rslc(tmp_path / "t.h5", 4, smear=3)
    assert _check(path, refs)["flags"] == _check(path, stack)["flags"]


def test_interior_zero_lines_flag_missing_data(tmp_path):
    out = _check(_write_rslc(tmp_path / "t.h5", 4, zero_lines=slice(100, 110)))
    assert out["flags"] == ["MISSING_DATA"]
    assert out["gap_lines"] == 10 and out["longest_gap_s"] == pytest.approx(10 * 0.000658)


def test_interior_empty_windows_flag_missing_data(tmp_path):
    out = _check(_write_rslc(tmp_path / "t.h5", 4, empty_lines=slice(120, 125)))
    assert out["flags"] == ["MISSING_DATA"] and out["gap_lines"] == 5


def test_missing_lines_at_scene_edge_not_flagged(tmp_path):
    out = _check(_write_rslc(tmp_path / "t.h5", 4, empty_lines=slice(0, 20), zero_lines=slice(240, N)))
    assert out["flags"] == [] and out["gap_lines"] == 0


def test_exception_codes_flagged(tmp_path):
    out = _check(_write_rslc(tmp_path / "t.h5", 4, exception_lines=slice(100, 140)))
    assert out["exception_frac"] == pytest.approx(40 / N)
    assert out["exception_bits"] == 4
    assert out["flags"] == ["EXCEPTION"]


def test_sample_slip_code_reported_not_flagged(tmp_path):
    out = _check(_write_rslc(tmp_path / "t.h5", 4, exception_lines=slice(100, 140), exception_code=2))
    assert out["exception_frac"] == pytest.approx(40 / N)
    assert out["exception_flag_frac"] == 0.0 and out["exception_bits"] == 2
    assert out["flags"] == []


def test_validity_bits_are_not_exceptions(tmp_path):
    out = _check(_write_rslc(tmp_path / "t.h5", 4, validity_bits=1 << 8))
    assert out["exception_frac"] == 0.0 and out["exception_bits"] == 0
    slip = _check(_write_rslc(tmp_path / "s.h5", 4, exception_lines=slice(100, 140), exception_code=2,
                              validity_bits=1 << 8))
    assert slip["exception_bits"] == 2 and slip["flags"] == []


def test_sample_slip_combined_with_other_code_flags(tmp_path):
    out = _check(_write_rslc(tmp_path / "t.h5", 4, exception_lines=slice(100, 140), exception_code=6))
    assert out["exception_bits"] == 6 and out["flags"] == ["EXCEPTION"]


def test_half_swath_flagged_without_and_with_references(tmp_path, stack):
    path = _write_rslc(tmp_path / "t.h5", 4, window=(8, 128))
    assert _check(path)["swath_width_m"] == pytest.approx(120 * 242e3 / 240)
    assert _check(path)["flags"] == ["HALF_SWATH"]
    assert "HALF_SWATH" in _check(path, stack)["flags"]


def test_partial_coverage_needs_references(tmp_path, stack):
    # data ends a quarter of the way early: no interior gap, but less area than the stack
    path = _write_rslc(tmp_path / "t.h5", 4, last_line=3 * N // 4)
    assert _check(path)["flags"] == []
    # on this tiny scene the missing quarter also moves placement and autocorrelation
    assert "PARTIAL_COVERAGE" in _check(path, stack)["flags"]


def test_crid_and_diagnostic_flags(tmp_path, stack):
    assert _check(_write_rslc(tmp_path / "b.h5", 4, crid="X05009"))["flags"] == ["PRE_PROVISIONAL"]
    assert _check(_write_rslc(tmp_path / "d.h5", 4, diagnostic=1))["flags"] == ["DIAGNOSTIC_MODE"]
    newer = _check(_write_rslc(tmp_path / "n.h5", 4, crid="P05030"), stack)
    assert newer["flags"] == ["MIXED_CRID"]


def test_xcorr_shift_recovers_roll():
    a = np.random.default_rng(0).random((32, 40))
    dy, dx, peak = quality._xcorr_shift(a, np.roll(a, (5, -3), axis=(0, 1)))
    assert (dy, dx) == (5, -3) and peak == pytest.approx(1.0)


def test_qa_stats_metrics(tmp_path):
    path = tmp_path / "qa.h5"
    dens = np.ones(200)
    dens[180:] = 0.5
    rg = np.zeros(2000)
    rg[:200] = -30.0
    with h5py.File(path, "w") as f:
        f["science/LSAR/QA/data/frequencyA/HH/backscatterHistogramDensity"] = dens
        f["science/LSAR/QA/data/frequencyA/HH/rangePowerSpectralDensity"] = rg
    m = quality._qa_stats_metrics(path, "A", "HH")
    assert m["qa_tail_pct"] == pytest.approx(10 / 190 * 100)
    assert m["qa_edge_drop_db"] == pytest.approx(-30.0)


# --- GUNW ------------------------------------------------------------------
DT, V, DR = 0.000658, 6794.0, 3.12  # az pixel = V * DT = 4.47 m


def _write_gunw(path, ref, sec, az_px, rg_px=0.0, coh=0.5, ref_gap_rows=None, empty_rows=None,
                sec_samples=55434, invalid_rows=1, ref_crid="P05023"):
    """Synthetic GUNW with uniform pixel offsets (written in meters) and coherence."""
    grids = "science/LSAR/GUNW/grids/frequencyA"
    with h5py.File(path, "w") as f:
        f["science/LSAR/identification/referenceZeroDopplerStartTime"] = f"{ref}T12:00:00".encode()
        f["science/LSAR/identification/secondaryZeroDopplerStartTime"] = f"{sec}T12:00:00".encode()
        f["science/LSAR/identification/diagnosticModeFlag"] = np.uint32(0)
        f["science/LSAR/identification/referenceHasInputDataException"] = np.uint8(2)
        f["science/LSAR/identification/secondaryHasInputDataException"] = np.uint8(0)
        inputs = "science/LSAR/GUNW/metadata/processingInformation/inputs"
        f[f"{inputs}/l1ReferenceSlcGranules"] = np.array([f"NISAR_L1_PR_RSLC_004_120_A_021_{ref}_{ref_crid}_N_F_J_001.h5".encode()])
        f[f"{inputs}/l1SecondarySlcGranules"] = np.array([f"NISAR_L1_PR_RSLC_008_120_A_021_{sec}_P05023_N_F_J_001.h5".encode()])
        f[f"{grids}/pixelOffsets/HH/alongTrackOffset"] = np.full((6, 6), az_px * V * DT, "f4")
        f[f"{grids}/pixelOffsets/HH/slantRangeOffset"] = np.full((6, 6), rg_px * DR, "f4")
        f[f"{grids}/pixelOffsets/HH/correlationSurfacePeak"] = np.full((6, 6), 0.5, "f4")
        f[f"{grids}/pixelOffsets/HH/xCoordinates"] = np.arange(6) * 80.0
        f[f"{grids}/pixelOffsets/HH/yCoordinates"] = -np.arange(6) * 80.0
        mask = np.full((6, 6), 11, "u4")  # land, subswath 1 in both
        mask[:invalid_rows] = 0           # invalid rows at the frame edge
        if ref_gap_rows is not None:
            mask[ref_gap_rows] = 1        # reference subswath digit 0: reference missing
        f[f"{grids}/pixelOffsets/mask"] = mask
        par = "science/LSAR/GUNW/metadata/processingInformation/parameters/reference/frequencyA"
        f[f"{par}/zeroDopplerTimeSpacing"] = DT
        f[f"{par}/slantRangeSpacing"] = DR
        f[f"{par}/numberOfRangeSamples"] = 55434
        sec_par = par.replace("reference", "secondary")
        f[f"{sec_par}/slantRangeSpacing"] = DR
        f[f"{sec_par}/numberOfRangeSamples"] = sec_samples
        f["science/LSAR/GUNW/metadata/radarGrid/groundTrackVelocity"] = np.full((2, 3, 3), V)
        cohm = np.full((6, 6), coh, "f4")
        phase = np.ones((6, 6), "f4")
        if empty_rows is not None:
            cohm[empty_rows] = np.nan     # mask says valid, layers hold nothing
            phase[empty_rows] = 0
        f[f"{grids}/unwrappedInterferogram/HH/coherenceMagnitude"] = cohm
        f[f"{grids}/unwrappedInterferogram/HH/unwrappedPhase"] = phase
        f[f"{grids}/unwrappedInterferogram/mask"] = mask
    return path


def test_gunw_offsets_converted_to_pixels(tmp_path):
    m = quality.gunw_quality_metrics(_write_gunw(tmp_path / "g.h5", "2025-11-06", "2025-11-18", 0.2, -0.3))
    assert m["az_offset_px_median"] == pytest.approx(0.2, rel=1e-4)
    assert m["rg_offset_px_median"] == pytest.approx(-0.3, rel=1e-4)
    assert m["sec_anomaly_frac"] == 0.0
    assert (m["ref_input_exception"], m["sec_input_exception"]) == (2, 0)


def test_gunw_large_offset_flagged_alone(tmp_path):
    out = identify_poor_gunw(_write_gunw(tmp_path / "g.h5", "2025-11-06", "2025-12-24", -1.7))
    assert out["flags"] == ["AZ_OFFSET"]


def test_gunw_network_pins_bad_date(tmp_path):
    # Dec 24 is 1.6 lines off: negative as secondary, positive as reference
    refs = [
        _write_gunw(tmp_path / "a.h5", "2025-12-24", "2026-01-05", 1.6),
        _write_gunw(tmp_path / "b.h5", "2026-01-05", "2026-01-17", 0.05),
        _write_gunw(tmp_path / "c.h5", "2025-11-06", "2025-11-18", 0.0),
        _write_gunw(tmp_path / "d.h5", "2025-11-18", "2025-12-24", -1.6),
    ]
    out = identify_poor_gunw(_write_gunw(tmp_path / "t.h5", "2025-11-06", "2025-12-24", -1.6), refs)
    assert out["date_az_offset_px"]["2025-12-24"] == pytest.approx(-1.6, abs=0.1)
    assert "BAD_DATE_2025-12-24" in out["flags"]
    assert "BAD_DATE_2025-11-06" not in out["flags"]
    good = identify_poor_gunw(refs[1], refs[:1] + refs[2:])
    assert not any(f.startswith("BAD_DATE") for f in good["flags"])


def test_gunw_low_coherence_needs_references(tmp_path):
    refs = [_write_gunw(tmp_path / f"r{i}.h5", "2026-01-05", "2026-01-17", 0.0, coh=0.6) for i in range(2)]
    path = _write_gunw(tmp_path / "t.h5", "2026-02-10", "2026-02-22", 0.0, coh=0.2)
    assert identify_poor_gunw(path)["flags"] == []
    assert identify_poor_gunw(path, refs)["flags"] == ["LOW_COHERENCE"]


def test_gunw_disconnected_reference_set_does_not_shift_dates(tmp_path):
    # a separate, unconnected pair must not move the zero of the chain holding the bad date
    refs = [
        _write_gunw(tmp_path / "a.h5", "2025-12-24", "2026-01-05", 1.6),
        _write_gunw(tmp_path / "b.h5", "2026-01-05", "2026-01-17", 0.0),
        _write_gunw(tmp_path / "x.h5", "2025-10-01", "2025-10-13", 0.0),
        _write_gunw(tmp_path / "y.h5", "2025-10-13", "2025-10-25", 0.0),
    ]
    out = identify_poor_gunw(_write_gunw(tmp_path / "t.h5", "2025-11-06", "2025-12-24", -1.6), refs)
    assert out["date_az_offset_px"]["2025-11-06"] == pytest.approx(0.0, abs=0.05)
    assert [f for f in out["flags"] if f.startswith("BAD_DATE")] == ["BAD_DATE_2025-12-24"]


def test_gunw_interior_missing_reference_flagged(tmp_path):
    edge = quality.gunw_quality_metrics(_write_gunw(tmp_path / "e.h5", "2025-12-17", "2025-12-29", 0.0))
    assert edge["ref_interior_missing_frac"] == 0.0
    out = identify_poor_gunw(_write_gunw(tmp_path / "g.h5", "2025-12-17", "2025-12-29", 0.0, ref_gap_rows=slice(2, 4)))
    assert out["ref_interior_missing_frac"] == pytest.approx(2 / 5)
    assert out["sec_interior_missing_frac"] == 0.0
    assert out["flags"] == ["MISSING_DATA"]


def test_gunw_valid_but_empty_flagged(tmp_path):
    out = identify_poor_gunw(_write_gunw(tmp_path / "g.h5", "2026-01-05", "2026-01-17", 0.0, empty_rows=slice(3, 5)))
    assert out["valid_but_empty_frac"] == pytest.approx(2 / 5)
    assert out["flags"] == ["EMPTY_VALID"]


def test_gunw_half_swath_scene_flagged(tmp_path):
    out = identify_poor_gunw(_write_gunw(tmp_path / "g.h5", "2026-01-05", "2026-01-17", 0.0, sec_samples=27717))
    assert out["flags"] == ["HALF_SWATH"]


def test_gunw_partial_coverage_needs_references(tmp_path):
    refs = [_write_gunw(tmp_path / f"r{i}.h5", "2026-01-05", "2026-01-17", 0.0) for i in range(2)]
    path = _write_gunw(tmp_path / "t.h5", "2026-02-10", "2026-02-22", 0.0, invalid_rows=3)
    assert quality.gunw_quality_metrics(refs[0])["valid_area_m2"] == pytest.approx(5 * 6 * 80.0 ** 2)
    assert identify_poor_gunw(path)["flags"] == []
    assert identify_poor_gunw(path, refs)["flags"] == ["PARTIAL_COVERAGE"]


def test_gunw_geocoding_fill_is_not_valid(tmp_path):
    # 255 is geocoding fill: no phase there, and it must not count as valid, empty or missing
    path = _write_gunw(tmp_path / "g.h5", "2026-01-05", "2026-01-17", 0.0)
    with h5py.File(path, "a") as f:
        g = "science/LSAR/GUNW/grids/frequencyA"
        for m in (f"{g}/pixelOffsets/mask", f"{g}/unwrappedInterferogram/mask"):
            arr = f[m][()]
            arr[:, :2] = 255
            f[m][...] = arr
        phase = f[f"{g}/unwrappedInterferogram/HH/unwrappedPhase"]
        arr = phase[()]
        arr[:, :2] = np.nan
        phase[...] = arr
    out = identify_poor_gunw(path)
    assert out["valid_but_empty_frac"] == 0.0
    assert out["valid_area_m2"] == pytest.approx(5 * 4 * 80.0 ** 2)
    assert out["flags"] == []


def test_gunw_beta_input_flagged(tmp_path):
    out = identify_poor_gunw(_write_gunw(tmp_path / "g.h5", "2025-10-25", "2025-11-06", 0.0, ref_crid="X05009"))
    assert out["ref_crid"] == "X05009" and out["sec_crid"] == "P05023"
    assert out["flags"] == ["PRE_PROVISIONAL", "MIXED_CRID"]


def test_interior_gap_does_not_move_autocorrelation(tmp_path, stack, monkeypatch):
    # a gap through the scene centre must not change the focus verdict; block smaller than the tiny scene
    monkeypatch.setattr(quality, "ACF_BLOCK", 64)
    out = _check(_write_rslc(tmp_path / "t.h5", 4, empty_lines=slice(120, 136)), stack)
    assert out["flags"] == ["MISSING_DATA"]
    assert out["acf_blocks"] == 4         # the middle block touches the gap and is skipped
