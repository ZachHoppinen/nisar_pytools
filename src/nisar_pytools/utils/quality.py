"""Flag poorly focused NISAR RSLCs and poorly coregistered GUNWs.

Each check reproduces a diagnosis made on the winter 2025-26 Colorado stacks,
with the thresholds that separated the known bad scenes there:

- Phantom time stamps (091_A_022 2026-01-03, Mores Creek 2026-01-14): an extra
  pulse train merged into the L0B time stamps mis-times every pulse. The RSLC
  shows a raised ``nominalAcquisitionPRF`` (1939 vs 1910 Hz), a whole-frame
  placement thousands of lines off its neighbours, and a smeared image.
  The L0B stamps themselves confirm it (``check_l0b=True``).
- Azimuth ghosting (091_A_022 2026-01-15, 120_A_021 2025-12-24): amplitude
  autocorrelation departs from the stack, and JPL's QA_STATS loses its
  brightest backscatter bins and deepens its range-spectrum edge.
- Heavy RFI (track 120 2026-03-06): interference-to-signal ratio median
  1.7 to 2.1 against 0.02 to 0.05, noise 30 to 50 times normal.
- Missing raw data (T019 2025-12-17, 2 s mode-switch segment not on
  Earthdata): azimuth lines 44069-51128 of the RSLC have no valid samples,
  and the GUNWs are invalid across a band of the frame.
- Misregistered RSLCs (T120 2025-12-24, about 2 lines): the GUNW's applied
  coregistration offsets are pixels, not tenths of pixels, with the sign
  flipping with the bad date's role, so a network of GUNWs pins the date.

Most RSLC checks compare against the rest of the stack, so pass the other
scenes of the same track and frame as ``references``. Without references only
the absolute RFI and offset thresholds apply; the metrics are still returned.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)

_RSLC = "science/LSAR/RSLC"
_GUNW = "science/LSAR/GUNW"
_QA = "science/LSAR/QA/data"

LOOKS = 64              # whole-frame amplitude looks for the placement check
ACF_BLOCK = 4096        # side of each single-look block for autocorrelation
ACF_POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)  # block centres as fractions of the frame's lines

# RSLC thresholds
RFI_ISR = 0.3           # interference-to-signal ratio median
RFI_HIT = 0.005         # RFI hit-count median
PRF_TOL_HZ = 1.0        # |PRF - reference median|
NOISE_FACTOR = 5.0      # noise-equivalent backscatter vs reference median
PLACEMENT_LINES = 3000  # frame cuts drift up to ~1700 lines between good passes
PLACEMENT_PEAK = 0.7    # normalized amplitude cross-correlation peak
ACF1_TOL = 0.05         # azimuth lag-1 autocorrelation vs reference median
ACF4_TOL = 0.03         # azimuth lag-4 autocorrelation vs reference median
QA_TAIL_FACTOR = 0.5    # bright histogram tail vs reference median
QA_EDGE_DB = 3.0        # range-spectrum edge drop deeper than reference median
STAMP_SHORT = 0         # stamp intervals under half a PRI (any one is a fault)
STAMP_TAIL_PRI = -0.5   # last stamp before the L0B zeroDopplerEndTime, in PRIs
GAP_LINE_INVALID = 0.5  # line counts as missing when over half its valid window is zero/NaN
EXCEPTION_FRAC = 0.01   # valid samples with a non-benign inputDataExceptionMask code
# qFSP-H1 sample slip (nisar.cal.qfsp_slip.AnomalyCode.SLIP_QFSP_H1): masks only the two DBF beam
# transition bands either side of the H1 processor; no coherence or phase loss seen under it (Taylor 2025-12-05)
BENIGN_EXCEPTION_BITS = 2
PROVISIONAL_CRID = 5023 # P05023: first PROVISIONAL release; lower CRIDs are BETA or earlier
COVERAGE_RATIO = 0.9    # valid area vs reference median (frame cuts drift ~3 %)
SWATH_RATIO = 0.75      # swath width vs reference median or nominal swath
NOMINAL_SWATH_M = 242e3 # NISAR L-band ground swath width

# GUNW thresholds
OFFSET_PX = 0.5         # applied coregistration offset, or per-date offset
COHERENCE_RATIO = 0.75  # median coherence vs reference median
MISSING_FRAC = 0.01     # interior missing fraction of either scene's swath


def _decode(value) -> str:
    """Decode an HDF5 byte string scalar."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _pre_provisional(crid: str) -> bool:
    """True for a CRID (e.g. X05009) older than the first PROVISIONAL release P05023."""
    digits = crid[1:]
    return digits.isdigit() and int(digits) < PROVISIONAL_CRID


def _read_image(
    slc: h5py.Dataset, looks: int, windows: list[np.ndarray], exceptions: h5py.Dataset | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """One pass over the image in row strips.

    Returns the mean amplitude in ``looks`` x ``looks`` blocks; per line the
    fraction of samples inside the valid-sample windows that are zero or
    non-finite (NaN for lines whose windows are empty); per line the number
    of samples inside the windows; and ``inputDataExceptionMask`` statistics
    over in-window samples: the fraction with any code, the fraction with a
    code outside ``BENIGN_EXCEPTION_BITS``, and the bitwise OR of all codes.
    """
    n0, n1 = slc.shape
    ny, nx = n0 // looks, n1 // looks
    amp = np.zeros((ny, nx), np.float32)
    invalid = np.full(n0, np.nan)
    window_samples = np.zeros(n0, int)
    n_exc = n_exc_flag = n_total = exc_bits = 0
    cols = np.arange(n1)
    step = max(looks, 4096 // looks * looks)
    for i in range(0, n0, step):
        blk = np.abs(slc[i:min(i + step, n0)])
        # samples inside any subswath's [start, end) window on each line
        inwin = np.zeros(blk.shape, bool)
        for w in windows:
            inwin |= (cols >= w[i:i + len(blk), :1]) & (cols < w[i:i + len(blk), 1:])
        bad = ~(np.isfinite(blk) & (blk > 0)) & inwin
        n_in = inwin.sum(axis=1)
        invalid[i:i + len(blk)] = np.where(n_in > 0, bad.sum(axis=1) / np.maximum(n_in, 1), np.nan)
        window_samples[i:i + len(blk)] = n_in
        n_total += int(n_in.sum())
        if exceptions is not None:
            # anomaly codes are bits 0-7; newer isce3 writes per-pol validity bits 8-15 (set = focused)
            exc = (exceptions[i:i + len(blk)][inwin] & 0xFF).astype(np.uint8)
            n_exc += int((exc > 0).sum())
            n_exc_flag += int(((exc & ~np.uint8(BENIGN_EXCEPTION_BITS)) > 0).sum())
            exc_bits |= int(np.bitwise_or.reduce(exc)) if exc.size else 0
        # multilook only the rows and columns that fill whole blocks
        b = min(len(blk), ny * looks - i) // looks * looks
        if b > 0:
            ml = np.nan_to_num(blk[:b, :nx * looks]).reshape(b // looks, looks, nx, looks).mean(axis=(1, 3))
            amp[i // looks:(i + b) // looks] = ml
    have = exceptions is not None and n_total > 0
    exc = {
        "exception_frac": n_exc / n_total if have else np.nan,
        "exception_flag_frac": n_exc_flag / n_total if have else np.nan,
        "exception_bits": exc_bits,
    }
    return amp, invalid, window_samples, exc


def _interior_gap_lines(invalid: np.ndarray) -> np.ndarray:
    """Lines between the first and last line with data that are empty or mostly invalid."""
    has_data = np.isfinite(invalid) & (invalid <= GAP_LINE_INVALID)
    rows = np.flatnonzero(has_data)
    if rows.size == 0:
        return np.arange(invalid.size)
    interior = np.arange(rows[0], rows[-1] + 1)
    return interior[~has_data[interior]]


def _longest_run(idx: np.ndarray) -> int:
    """Length of the longest run of consecutive integers in a sorted index array."""
    if idx.size == 0:
        return 0
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, idx.size - 1]
    return int((ends - starts + 1).max())


def _mask_scenes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference-valid, secondary-valid and water pixels from a GUNW mask layer.

    Bits 0-7 hold decimal digits: water, reference subswath, secondary
    subswath (0 = invalid sample). 255 is geocoding fill (isce3 issue #224;
    GPU-geocoded products use it, CPU ones write 0) and is neither scene.
    """
    digits = mask & 255
    data = digits != 255
    return ((digits // 10) % 10 > 0) & data, (digits % 10 > 0) & data, (digits // 100 > 0) & data


def _column_filled(valid: np.ndarray) -> np.ndarray:
    """Pixels between the first and last valid row of each column.

    NISAR tracks run roughly north-south, so grid columns lie close to along
    track and a convex swath fills each column in one interval; invalid pixels
    inside it are data missing in the middle of the swath, not frame edges.
    """
    n = valid.shape[0]
    any_valid = valid.any(axis=0)
    first = np.where(any_valid, valid.argmax(axis=0), n)
    last = np.where(any_valid, n - 1 - valid[::-1].argmax(axis=0), -1)
    rows = np.arange(n)[:, None]
    return (rows >= first) & (rows <= last)


def _acf(x: np.ndarray, lag: int, axis: int) -> float:
    """Normalized autocorrelation of ``x`` at one lag along one axis."""
    a = x - x.mean()
    b = np.roll(a, -lag, axis=axis)
    sl = [slice(None)] * 2
    sl[axis] = slice(0, -lag)
    return float((a[tuple(sl)] * b[tuple(sl)]).mean() / a.var())


def _xcorr_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Integer shift (rows, cols) of ``b`` relative to ``a`` and the normalized peak."""
    n = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a = a[:n[0], :n[1]]
    b = b[:n[0], :n[1]]
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    c = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real / a.size
    k = np.unravel_index(c.argmax(), c.shape)
    sh = [(v if v < m // 2 else v - m) for v, m in zip(k, c.shape)]
    return -sh[0], -sh[1], float(c.max())


def _fetch_qa_stats(path: Path, qa_dir: Path | None) -> Path | None:
    """Local QA_STATS.h5 for an RSLC, downloaded from ASF if not already in ``qa_dir``."""
    import asf_search as asf

    from nisar_pytools.io.download import download_urls

    qa_dir = Path(qa_dir) if qa_dir else path.parent
    local = sorted(qa_dir.glob(f"{path.stem}*QA_STATS.h5"))
    if local:
        return local[0]
    urls = [u for p in asf.granule_search(path.stem) for u in p.find_urls() if u.endswith("QA_STATS.h5")]
    if not urls:
        log.warning("No QA_STATS.h5 on ASF for %s", path.stem)
        return None
    got = download_urls(urls[:1], qa_dir, validate=False)
    return got[0] if got else None


def _qa_stats_metrics(qa_path: Path, frequency: str, pol: str) -> dict:
    """Bright-tail mass and range-spectrum edge drop from JPL's QA_STATS.h5."""
    with h5py.File(qa_path, "r") as h:
        g = h[f"{_QA}/frequency{frequency}/{pol}"]
        dens = g["backscatterHistogramDensity"][()]
        rg = g["rangePowerSpectralDensity"][()]
    n = len(rg)
    return {
        # percent of histogram mass in the brightest 20 of 200 bins
        "qa_tail_pct": float(dens[180:].sum() / dens.sum() * 100),
        # band-edge PSD minus band-centre PSD, dB (more negative = deeper edge)
        "qa_edge_drop_db": float(np.median(rg[:200]) - np.median(rg[n // 2 - 500:n // 2 + 500])),
    }


def _l0b_stamp_metrics(granules: list[str], frequency: str, pol: str) -> dict:
    """Phantom-stamp signatures in the L0B pulse time stamps, streamed from Earthdata."""
    from nisar_pytools.io.stream_l0b import open_remote_l0b

    short, tail = 0, np.inf
    for g in granules:
        with open_remote_l0b(g) as fh:
            tx = fh[f"science/LSAR/RRSD/swaths/frequency{frequency}/tx{pol[0]}"]
            utc = tx["UTCtime"][()]
            epoch = np.datetime64(_decode(tx["UTCtime"].attrs["units"]).split("since ")[1].strip())
            end = np.datetime64(_decode(fh["science/LSAR/identification/zeroDopplerEndTime"][()]))
        t_end = (end - epoch) / np.timedelta64(1, "ns") * 1e-9
        n_short, g_tail = _stamp_faults(utc, t_end)
        log.info("L0B %s: %d short stamp intervals, last stamp %+.2f PRI from end", g[:60], n_short, g_tail)
        short += n_short
        tail = min(tail, g_tail)
    return {"l0b_short_intervals": short, "l0b_tail_pri": tail}


def _stamp_faults(utc: np.ndarray, t_end: float) -> tuple[int, float]:
    """Short (or backward) stamp intervals and the last stamp's offset from the end time, in PRIs."""
    dt = np.diff(utc)
    pri = np.median(dt)
    # a duplicate or interleaved stamp sits closer than half a PRI to its neighbour
    n_short = int((dt < 0.5 * pri).sum())
    # stamps shifted by a duplicate or phantom train leave the last stamp early
    return n_short, float((utc[-1] - t_end) / pri)


def rslc_quality_metrics(
    rslc: str | Path,
    frequency: str = "A",
    pol: str = "HH",
    qa_stats: bool = True,
    qa_dir: str | Path | None = None,
    check_l0b: bool = False,
) -> dict:
    """Per-scene focus, timing and RFI metrics of one RSLC.

    Reads the whole image once for a ``LOOKS`` x ``LOOKS`` amplitude, which is
    kept in the result (``amp``) so the dict can serve as a reference later
    without re-reading the file.

    Parameters
    ----------
    rslc : str or Path
        RSLC HDF5 file.
    frequency, pol : str
        Which image to check.
    qa_stats : bool
        Fetch JPL's QA_STATS.h5 for this granule from ASF (cached in ``qa_dir``).
    qa_dir : str or Path, optional
        Where QA_STATS.h5 files live or are downloaded to. Defaults to the
        RSLC's directory.
    check_l0b : bool
        Stream the source L0B time stamps from Earthdata and test for phantom,
        duplicate or backward stamps. Needs an Earthdata login.

    Returns
    -------
    dict
        Scalars per metric plus ``amp`` (multilooked amplitude).
    """
    path = Path(rslc)
    with h5py.File(path, "r") as f:
        if f"{_RSLC}/swaths/frequency{frequency}" not in f:
            raise ValueError(f"{path.name} has no frequency{frequency} swath (side-band-only acquisition?)")
        freq = f[f"{_RSLC}/swaths/frequency{frequency}"]
        meta = f"{_RSLC}/metadata"
        out = {
            "file": path.name,
            "date": _decode(f["science/LSAR/identification/zeroDopplerStartTime"][()])[:10],
            "prf_hz": float(freq["nominalAcquisitionPRF"][()]),
            "crid": _decode(f["science/LSAR/identification/compositeReleaseId"][()]),
            "diagnostic_mode": int(f["science/LSAR/identification/diagnosticModeFlag"][()]),
        }
        # RFI layers sit under the raw datatake id (e.g. raw44376); pool every datatake
        rfi_path = f"{meta}/RFI/frequency{frequency}/{pol}"
        if rfi_path in f:
            rfi = f[rfi_path]
            out["isr_median"] = float(np.nanmedian(np.concatenate(
                [rfi[k]["interferenceSignalRatio"][()].ravel() for k in rfi])))
            out["hit_median"] = float(np.nanmedian(np.concatenate(
                [rfi[k]["hitCount"][()].ravel() for k in rfi])))
        else:
            out["isr_median"] = out["hit_median"] = np.nan
        neb_path = f"{meta}/calibrationInformation/frequency{frequency}/noiseEquivalentBackscatter/{pol}"
        out["noise_median"] = float(np.nanmedian(f[neb_path][()])) if neb_path in f else np.nan
        l0bs = [_decode(g) for g in f[f"{meta}/processingInformation/inputs/l0bGranules"][()]]

        slc = freq[pol]
        dt = float(f[f"{_RSLC}/swaths/zeroDopplerTimeSpacing"][()])
        windows = [freq[k][()] for k in sorted(freq) if k.startswith("validSamplesSubSwath")]
        if not windows:
            # no valid-sample metadata: treat every whole line as the window
            windows = [np.tile([0, slc.shape[1]], (slc.shape[0], 1))]
        # read from the RSLC: GUNW masks can lose these bits (isce3 issue #335)
        exceptions = freq["inputDataExceptionMask"] if "inputDataExceptionMask" in freq else None
        out["amp"], invalid, window_samples, exc = _read_image(slc, LOOKS, windows, exceptions)
        out.update(exc)
        # lines missing inside the scene: empty valid-sample windows or mostly zero/NaN samples
        gaps = _interior_gap_lines(invalid)
        out["gap_lines"] = int(gaps.size)
        out["longest_gap_s"] = _longest_run(gaps) * dt
        # coverage and swath width on the ground, from scene-centre pixel spacings (approximate:
        # ground range spacing varies a few percent across the swath with incidence angle)
        dx_ground = float(freq["sceneCenterGroundRangeSpacing"][()])
        dy_ground = float(freq["sceneCenterAlongTrackSpacing"][()])
        has_data = np.isfinite(invalid) & (invalid <= GAP_LINE_INVALID)
        valid_samples = np.nansum(window_samples * (1 - invalid))
        out["valid_area_m2"] = float(valid_samples * dx_ground * dy_ground)
        out["swath_width_m"] = float(np.median(window_samples[has_data]) * dx_ground) if has_data.any() else 0.0
        # amplitude autocorrelation in single-look blocks at fixed positions along the frame (same ground
        # in every scene of a stack), centred in range; blocks touching missing lines are skipped
        n0, n1 = slc.shape
        side = min(ACF_BLOCK, n0, n1)
        acfs = []
        for frac in ACF_POSITIONS:
            a0 = min(max(int(frac * n0) - side // 2, 0), n0 - side)
            if not has_data[a0:a0 + side].all():
                continue
            x = np.abs(slc[a0:a0 + side, n1 // 2 - side // 2:n1 // 2 + side // 2]).astype(np.float64)
            acfs.append((_acf(x, 1, 0), _acf(x, 2, 0), _acf(x, 4, 0), _acf(x, 1, 1)))
    out["acf_blocks"] = len(acfs)
    med = np.median(acfs, axis=0) if acfs else np.full(4, np.nan)
    out["az_acf1"], out["az_acf2"], out["az_acf4"], out["rg_acf1"] = (float(v) for v in med)

    out["qa_tail_pct"] = out["qa_edge_drop_db"] = np.nan
    if qa_stats:
        qa_path = _fetch_qa_stats(path, qa_dir)
        if qa_path is not None:
            out.update(_qa_stats_metrics(qa_path, frequency, pol))
    if check_l0b:
        out.update(_l0b_stamp_metrics(l0bs, frequency, pol))

    log.info(
        "%s: PRF %.2f Hz, ISR %.3f, hit %.4f, noise %.3g, az acf1 %.3f acf4 %.3f, QA tail %.4f%% edge %+.1f dB, "
        "%d gap lines (longest %.2f s), exception codes on %.4f, valid area %.0f km2, swath %.1f km",
        out["date"], out["prf_hz"], out["isr_median"], out["hit_median"], out["noise_median"],
        out["az_acf1"], out["az_acf4"], out["qa_tail_pct"], out["qa_edge_drop_db"],
        out["gap_lines"], out["longest_gap_s"], out["exception_frac"],
        out["valid_area_m2"] / 1e6, out["swath_width_m"] / 1e3,
    )
    return out


def identify_poorly_focused_rslc(
    rslc: str | Path | dict,
    references: list[str | Path | dict] | None = None,
    frequency: str = "A",
    pol: str = "HH",
    qa_stats: bool = True,
    qa_dir: str | Path | None = None,
    check_l0b: bool = False,
) -> dict:
    """Flag an RSLC that is defocused, mis-timed or RFI-contaminated.

    Flags (in ``result["flags"]``):

    - ``RFI``: interference-to-signal ratio median > 0.3 or hit median > 0.005.
    - ``MISSING_DATA``: any line between the first and last line with data
      whose valid-sample window is empty, or whose samples in that window are
      more than half zero or NaN (T019 2025-12-17: 7060 empty lines, 4.6 s).
    - ``EXCEPTION``: more than 1 % of valid samples carry an
      ``inputDataExceptionMask`` code other than the benign ones in
      ``BENIGN_EXCEPTION_BITS``. Code 2 (qFSP-H1 sample slip) is reported in
      ``exception_frac`` / ``exception_bits`` but not flagged: on T019
      2025-12-05 it covers 18 % of samples with no coherence or phase-spread
      loss under it.
    - ``HALF_SWATH``: valid ground swath width (median valid samples per line
      x scene-centre ground range spacing) below 0.75 x the reference median,
      or below 0.75 x the nominal 242 km without references (full scenes
      measure 242.6 km). Half-swath scenes get a wrong IMAGEN ionosphere LUT
      in production (isce3 issue #153).
    - ``PRE_PROVISIONAL``: CRID older than P05023 (BETA, e.g. X05009).
    - ``DIAGNOSTIC_MODE``: ``diagnosticModeFlag`` nonzero; ASF lists the
      calibration-mode frames 161/174, 161/175, 169/090, 169/091 as unusable
      before August 2026.
    - ``L0B_STAMPS`` (``check_l0b``): any L0B stamp interval under half a PRI
      (duplicate, backward or phantom stamps), or the last stamp of any L0B more
      than half a PRI before its end time. Catches the 091_A_022 2026-01-03
      phantom burst and the 027_D_068 2026-02-04 duplicate stamp, after which
      every stamp is one pulse early.

    With references (other scenes of the same track and frame):

    - ``PRF``: nominalAcquisitionPRF more than 1 Hz from the reference median.
    - ``NOISE``: noise-equivalent backscatter above 5 x the reference median.
    - ``MIXED_CRID``: CRID differs from the most common reference CRID.
    - ``PARTIAL_COVERAGE``: valid area below 0.9 x the reference median, e.g.
      raw data missing at the start or end of the frame while the product
      still says ``isFullFrame`` (isce3 issue #342).
    - ``PLACEMENT``: whole-frame amplitude shifted more than 3000 lines from the
      references, or cross-correlation peak below 0.7 (median over references).
    - ``FOCUS_ACF``: azimuth amplitude autocorrelation more than 0.05 (lag 1)
      or 0.03 (lag 4) from the reference median.
    - ``FOCUS_QA``: QA_STATS bright tail below 0.5 x the reference median, or
      range-spectrum edge more than 3 dB deeper than the reference median.

    Parameters
    ----------
    rslc : str, Path or dict
        RSLC file, or a result of :func:`rslc_quality_metrics`.
    references : list of str, Path or dict, optional
        Other RSLCs of the same track and frame, as files or precomputed
        metric dicts. Pass dicts when checking a whole stack so each file is
        read once.
    frequency, pol, qa_stats, qa_dir, check_l0b
        See :func:`rslc_quality_metrics`. ``check_l0b`` applies to the target only.

    Returns
    -------
    dict
        The target's metrics plus ``flags`` (list of str) and
        ``poorly_focused`` (bool).
    """
    def metrics(x, l0b):
        if isinstance(x, dict):
            return x
        return rslc_quality_metrics(x, frequency, pol, qa_stats, qa_dir, l0b)

    out = dict(metrics(rslc, check_l0b))
    refs = [metrics(r, False) for r in references or []]
    flags = []

    # absolute checks, no references needed
    if out["isr_median"] > RFI_ISR or out["hit_median"] > RFI_HIT:
        flags.append("RFI")
    if out["gap_lines"] > 0:
        flags.append("MISSING_DATA")
    if out["exception_flag_frac"] > EXCEPTION_FRAC:
        flags.append("EXCEPTION")
    # half swath: against the stack when given, else against the nominal NISAR swath
    swath_ref = float(np.median([r["swath_width_m"] for r in refs])) if refs else NOMINAL_SWATH_M
    if out["swath_width_m"] < SWATH_RATIO * swath_ref:
        flags.append("HALF_SWATH")
    if out.get("l0b_short_intervals", 0) > STAMP_SHORT or out.get("l0b_tail_pri", 0) < STAMP_TAIL_PRI:
        flags.append("L0B_STAMPS")
    if _pre_provisional(out["crid"]):
        flags.append("PRE_PROVISIONAL")
    if out["diagnostic_mode"] != 0:
        flags.append("DIAGNOSTIC_MODE")

    if refs:
        def ref_median(key):
            return float(np.nanmedian([r[key] for r in refs]))

        # a different processing release than most of the stack (BETA scenes sat 1 px off PROVISIONAL)
        ref_crids = [r["crid"] for r in refs]
        if out["crid"] != max(set(ref_crids), key=ref_crids.count):
            flags.append("MIXED_CRID")

        if abs(out["prf_hz"] - ref_median("prf_hz")) > PRF_TOL_HZ:
            flags.append("PRF")
        if out["noise_median"] > NOISE_FACTOR * ref_median("noise_median"):
            flags.append("NOISE")
        if out["valid_area_m2"] < COVERAGE_RATIO * ref_median("valid_area_m2"):
            flags.append("PARTIAL_COVERAGE")

        # whole-frame placement against every reference, in single-look lines
        shifts = [_xcorr_shift(r["amp"], out["amp"]) for r in refs]
        out["placement_az_lines"] = int(np.median([s[0] for s in shifts]) * LOOKS)
        out["placement_rg_samples"] = int(np.median([s[1] for s in shifts]) * LOOKS)
        out["placement_peak"] = float(np.median([s[2] for s in shifts]))
        if abs(out["placement_az_lines"]) > PLACEMENT_LINES or out["placement_peak"] < PLACEMENT_PEAK:
            flags.append("PLACEMENT")

        if (abs(out["az_acf1"] - ref_median("az_acf1")) > ACF1_TOL
                or abs(out["az_acf4"] - ref_median("az_acf4")) > ACF4_TOL):
            flags.append("FOCUS_ACF")
        # QA_STATS check only when the target and some references have one
        qa_refs = [r for r in refs if np.isfinite(r["qa_tail_pct"])]
        if np.isfinite(out["qa_tail_pct"]) and qa_refs:
            tail_med = float(np.median([r["qa_tail_pct"] for r in qa_refs]))
            edge_med = float(np.median([r["qa_edge_drop_db"] for r in qa_refs]))
            if out["qa_tail_pct"] < QA_TAIL_FACTOR * tail_med or out["qa_edge_drop_db"] < edge_med - QA_EDGE_DB:
                flags.append("FOCUS_QA")

    out["flags"] = flags
    out["poorly_focused"] = bool(flags)
    log.info("%s: %s", out["date"], ", ".join(flags) or "ok")
    return out


def gunw_quality_metrics(gunw: str | Path, frequency: str = "A", pol: str = "HH") -> dict:
    """Coregistration offsets, correlation peak and coherence of one GUNW.

    The ``pixelOffsets`` layers are the rubbersheet offsets production applied
    to the secondary on top of geometric coregistration, i.e. the timing or
    geometry error of the input RSLCs. isce3 writes them in meters as
    ``pixels * groundTrackVelocity * zeroDopplerTimeSpacing`` (along track) and
    ``pixels * slantRangeSpacing`` (range); this converts them back to pixels
    using the median of the radarGrid groundTrackVelocity cube, which varies
    by about 0.25 % over a frame.

    Returns
    -------
    dict
        ``reference_date``, ``secondary_date``, offset medians and 5/95
        percentiles in pixels, ``corr_peak_median``, ``coherence_median`` and
        the fraction of valid pixels with RSLC anomaly bits set.
    """
    path = Path(gunw)
    grids = f"{_GUNW}/grids/frequency{frequency}"
    with h5py.File(path, "r") as f:
        ident = f["science/LSAR/identification"]
        po = f[f"{grids}/pixelOffsets/{pol}"]
        az_m = po["alongTrackOffset"][()]
        rg_m = po["slantRangeOffset"][()]
        peak = po["correlationSurfacePeak"][()]
        mask = f[f"{grids}/pixelOffsets/mask"][()]
        x = po["xCoordinates"][()]
        y = po["yCoordinates"][()]
        params = f"{_GUNW}/metadata/processingInformation/parameters"
        par = f[f"{params}/reference/frequency{frequency}"]
        dt = float(par["zeroDopplerTimeSpacing"][()])
        dr = float(par["slantRangeSpacing"][()])
        # slant-range extent of each input RSLC grid, to spot a half-swath scene
        extents = {}
        for scene in ("reference", "secondary"):
            p = f[f"{params}/{scene}/frequency{frequency}"]
            extents[scene] = float(p["numberOfRangeSamples"][()]) * float(p["slantRangeSpacing"][()])
        v = float(np.nanmedian(f[f"{_GUNW}/metadata/radarGrid/groundTrackVelocity"][()]))
        coh = f[f"{grids}/unwrappedInterferogram/{pol}/coherenceMagnitude"][()]
        unw = f[f"{grids}/unwrappedInterferogram/{pol}/unwrappedPhase"][()]
        unw_mask = f[f"{grids}/unwrappedInterferogram/mask"][()]
        out = {
            "file": path.name,
            "reference_date": _decode(ident["referenceZeroDopplerStartTime"][()])[:10],
            "secondary_date": _decode(ident["secondaryZeroDopplerStartTime"][()])[:10],
            "diagnostic_mode": int(ident["diagnosticModeFlag"][()]),
            # input RSLC exception codes from identification (survives isce3 issue #335)
            "ref_input_exception": int(ident["referenceHasInputDataException"][()]),
            "sec_input_exception": int(ident["secondaryHasInputDataException"][()]),
        }
        # each input RSLC's CRID, from its granule name (..._P05023_N_F_J_001)
        inputs = f[f"{_GUNW}/metadata/processingInformation/inputs"]
        for name, key in (("ref", "l1ReferenceSlcGranules"), ("sec", "l1SecondarySlcGranules")):
            granule = _decode(np.atleast_1d(inputs[key][()])[0])
            match = re.search(r"_([A-Z]\d{5})_", granule)
            out[f"{name}_crid"] = match.group(1) if match else ""

    ref_ok, sec_ok, water = _mask_scenes(mask)
    valid = ref_ok & sec_ok & ~water
    valid &= np.isfinite(az_m) & np.isfinite(rg_m)
    az_px = az_m[valid] / (v * dt)
    rg_px = rg_m[valid] / dr
    for name, arr in (("az", az_px), ("rg", rg_px)):
        out[f"{name}_offset_px_median"] = float(np.median(arr))
        out[f"{name}_offset_px_p05"], out[f"{name}_offset_px_p95"] = (float(q) for q in np.percentile(arr, [5, 95]))
    out["corr_peak_median"] = float(np.nanmedian(peak[valid]))
    # area where both scenes are valid (water included), and each scene's slant-range extent
    pixel_area = abs((x[1] - x[0]) * (y[1] - y[0]))
    out["valid_area_m2"] = float((ref_ok & sec_ok).sum() * pixel_area)
    out["ref_slant_extent_m"] = extents["reference"]
    out["sec_slant_extent_m"] = extents["secondary"]
    # samples missing inside each scene's swath (a raw-data gap), not at the frame edges
    for name, scene in (("ref", ref_ok), ("sec", sec_ok)):
        filled = _column_filled(scene)
        out[f"{name}_interior_missing_frac"] = float((filled & ~scene).sum() / max(filled.sum(), 1))
    # bits 8-15 flag secondary RSLC anomalies, bits 16-23 reference
    out["sec_anomaly_frac"] = float((((mask >> 8) & 255)[valid] > 0).mean())
    out["ref_anomaly_frac"] = float((((mask >> 16) & 255)[valid] > 0).mean())
    out["coherence_median"] = float(np.median(coh[np.isfinite(coh) & (coh > 0)]))
    # both scenes valid per the mask, yet no phase or coherence (isce3 issue #62: bad dense offsets)
    unw_ref, unw_sec, _ = _mask_scenes(unw_mask)
    both = unw_ref & unw_sec
    empty = ~np.isfinite(unw) | (unw == 0) | ~np.isfinite(coh)
    out["valid_but_empty_frac"] = float((empty & both).sum() / max(both.sum(), 1))

    log.info(
        "%s_%s: az offset %+.2f px, rg offset %+.2f px, corr peak %.2f, coherence %.3f, interior missing ref %.4f sec %.4f",
        out["reference_date"], out["secondary_date"], out["az_offset_px_median"],
        out["rg_offset_px_median"], out["corr_peak_median"], out["coherence_median"],
        out["ref_interior_missing_frac"], out["sec_interior_missing_frac"],
    )
    return out


def _date_offsets(pairs: list[dict], key: str) -> dict[str, float]:
    """Per-date offsets whose secondary-minus-reference differences fit each pair's median offset.

    Least squares, then shifted so the median date of each connected set of
    dates sits at zero (each set is only known up to its own constant): a
    single mis-timed date stands out against the rest.
    """
    dates = sorted({p["reference_date"] for p in pairs} | {p["secondary_date"] for p in pairs})
    idx = {d: i for i, d in enumerate(dates)}
    g = np.zeros((len(pairs), len(dates)))
    for row, p in enumerate(pairs):
        g[row, idx[p["reference_date"]]] = -1.0
        g[row, idx[p["secondary_date"]]] = 1.0
    e = np.linalg.lstsq(g, np.array([p[key] for p in pairs]), rcond=None)[0]

    # label connected sets of dates by following pair links to a root date
    parent = list(range(len(dates)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for p in pairs:
        parent[root(idx[p["reference_date"]])] = root(idx[p["secondary_date"]])
    roots = np.array([root(i) for i in range(len(dates))])
    for r in set(roots):
        e[roots == r] -= np.median(e[roots == r])
    return {d: float(e[idx[d]]) for d in dates}


def identify_poor_gunw(
    gunw: str | Path | dict,
    references: list[str | Path | dict] | None = None,
    frequency: str = "A",
    pol: str = "HH",
) -> dict:
    """Flag a GUNW built from a misregistered or mis-timed RSLC.

    Flags (in ``result["flags"]``):

    - ``AZ_OFFSET`` / ``RG_OFFSET``: median applied coregistration offset above
      0.5 px. Production corrects this with a degree-2 polynomial, which can
      miss the local offset by 0.4 px over a small AOI.
    - ``MISSING_DATA``: more than 1 % of either scene's swath invalid in the
      middle of the frame (T019 2025-12-17 reference: 13 %; good scenes
      0.01 % or less), from the pixelOffsets mask subswath digits.
    - ``EMPTY_VALID``: more than 1 % of pixels valid in both scenes per the
      mask have NaN or zero unwrapped phase or NaN coherence, the symptom of
      blown-up dense offsets in isce3 issue #62 (checked GUNWs: 0.02 % or less).
    - ``HALF_SWATH``: one input RSLC's slant-range extent is under 0.75 x the
      other's (or, with references, under 0.75 x the reference median).
      Production's IMAGEN ionosphere LUT assumes a full swath (isce3 issue #153).
    - ``PRE_PROVISIONAL`` / ``MIXED_CRID``: an input RSLC older than P05023,
      or the two inputs from different releases (ASF warns against mixing
      BETA and PROVISIONAL; East River BETA scenes sat 1 range px off).
    - ``DIAGNOSTIC_MODE``: ``diagnosticModeFlag`` nonzero.

    The input RSLCs' exception codes are reported, not flagged: as
    ``ref_input_exception`` / ``sec_input_exception`` from identification, and
    as the fraction of pixels carrying them (``ref_anomaly_frac``,
    ``sec_anomaly_frac``) from the mask bits, which GUNWs made with NumPy 2
    lose (isce3 issue #335). :func:`identify_poorly_focused_rslc` flags
    non-benign codes from the RSLC itself.

    With references (other GUNWs of the same track and frame):

    - ``BAD_DATE_<date>``: solving every pair's median offset for one offset per
      date puts one of this GUNW's dates more than 0.5 px from the stack, in
      azimuth or range. Needs a reference that shares a date with this GUNW.
    - ``LOW_COHERENCE``: median coherence below 0.75 x the reference median.
    - ``PARTIAL_COVERAGE``: area valid in both scenes below 0.9 x the
      reference median (isce3 issue #342: partial frames labelled full).

    Parameters
    ----------
    gunw : str, Path or dict
        GUNW file, or a result of :func:`gunw_quality_metrics`.
    references : list of str, Path or dict, optional
        Other GUNWs of the same track and frame, as files or metric dicts.
    frequency, pol : str
        Which layers to check.

    Returns
    -------
    dict
        The target's metrics plus ``date_az_offset_px`` and
        ``date_rg_offset_px`` (with references), ``flags`` and
        ``poorly_coregistered``.
    """
    def metrics(x):
        return x if isinstance(x, dict) else gunw_quality_metrics(x, frequency, pol)

    out = dict(metrics(gunw))
    refs = [metrics(r) for r in references or []]
    flags = []
    if abs(out["az_offset_px_median"]) > OFFSET_PX:
        flags.append("AZ_OFFSET")
    if abs(out["rg_offset_px_median"]) > OFFSET_PX:
        flags.append("RG_OFFSET")
    if max(out["ref_interior_missing_frac"], out["sec_interior_missing_frac"]) > MISSING_FRAC:
        flags.append("MISSING_DATA")
    if out["valid_but_empty_frac"] > MISSING_FRAC:
        flags.append("EMPTY_VALID")
    # half swath: the two scenes disagree, or either is narrow against the stack
    ext = (out["ref_slant_extent_m"], out["sec_slant_extent_m"])
    half = min(ext) < SWATH_RATIO * max(ext)
    if refs:
        ref_ext = float(np.median([e for r in refs for e in (r["ref_slant_extent_m"], r["sec_slant_extent_m"])]))
        half |= min(ext) < SWATH_RATIO * ref_ext
    if half:
        flags.append("HALF_SWATH")
    if _pre_provisional(out["ref_crid"]) or _pre_provisional(out["sec_crid"]):
        flags.append("PRE_PROVISIONAL")
    if out["ref_crid"] != out["sec_crid"]:
        flags.append("MIXED_CRID")
    if out["diagnostic_mode"] != 0:
        flags.append("DIAGNOSTIC_MODE")

    if refs:
        if out["coherence_median"] < COHERENCE_RATIO * float(np.median([r["coherence_median"] for r in refs])):
            flags.append("LOW_COHERENCE")
        if out["valid_area_m2"] < COVERAGE_RATIO * float(np.median([r["valid_area_m2"] for r in refs])):
            flags.append("PARTIAL_COVERAGE")
        own = {out["reference_date"], out["secondary_date"]}
        if any(own & {r["reference_date"], r["secondary_date"]} for r in refs):
            pairs = [out, *refs]
            out["date_az_offset_px"] = _date_offsets(pairs, "az_offset_px_median")
            out["date_rg_offset_px"] = _date_offsets(pairs, "rg_offset_px_median")
            for d in sorted(own):
                if abs(out["date_az_offset_px"][d]) > OFFSET_PX or abs(out["date_rg_offset_px"][d]) > OFFSET_PX:
                    flags.append(f"BAD_DATE_{d}")
        else:
            log.warning("No reference GUNW shares a date with %s; skipping per-date offsets", out["file"])

    out["flags"] = flags
    out["poorly_coregistered"] = bool(flags)
    log.info("%s_%s: %s", out["reference_date"], out["secondary_date"], ", ".join(flags) or "ok")
    return out
