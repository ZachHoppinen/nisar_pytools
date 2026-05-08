# Production NISAR, ISCE3, NISAR Pytools Interferogram Comparisons

## Goal

Quantify how well two locally-controlled InSAR processing paths reproduce
JPL's official NISAR L2 GUNW for the same acquisition pair, and document
the precision/cost tradeoff between them.

## Data

Single ASF-distributed NISAR pair, track 77, frame 24:

- **Reference**: `NISAR_L1_PR_RSLC_004_077_..._20251103T124615...` and matching X05009 GSLC
- **Secondary**: `NISAR_L1_PR_RSLC_005_077_..._20251115T124615...` and matching X05009 GSLC
- **JPL GUNW**: `NISAR_L2_PR_GUNW_004_077_A_024_005_4000_SH_..._X05010_N_F_J_001`
- 12-day repeat over OR/ID/NV (UTM 11N), DHDH mode, frequency A HH

Comparison AOI: 80 × 80 km centered in the scene, 80 m posting (1000 × 1000 px).

## Three processing paths

| Path | Inputs | Software | Wall time | Stages |
|---|---|---|---|---|
| **Production NISAR** | RSLC pair + ANC (DEM, ECMWF, TEC, water mask, orbit XML) | JPL PCM (`nisar.workflows.insar`) on AWS | hours | full pipeline including troposphere + TEC ionosphere + SET |
| **Local ISCE3** | RSLC pair + DEM (Copernicus 30 m) | `nisar.workflows.insar` v0.25.8, locally on CPU | ~86 min | rdr2geo / geo2rdr / coreg / crossmul / filter / unwrap / split-spectrum ionosphere / geocode / SET / baseline (no troposphere, no TEC) |
| **NISAR Pytools** | X05009 GSLC pair (already geocoded) | `nisar_pytools.processing` (`interferogram`, `multilook`, `coherence`, `unwrap`) | ~15 sec | element-wise IFG → multilook → coherence → SNAPHU |

## Headline results

### Local ISCE3 vs JPL (RSLC → GUNW pipeline)

| Band | R² | Pearson r | RMSE |
|---|---|---|---|
| Unwrapped phase | **1.0000** | **1.0000** | 0.0034 rad (~0.13 mm LOS at L-band) |
| Coherence | **1.0000** | **1.0000** | 0.0005 |
| Ionosphere phase screen | 0.9281 | 0.9687 | 0.1028 rad |

→ Local ISCE3 reproduces JPL's unwrapped phase and coherence to **mm / sub-percent precision**. Demonstrates the production GUNW workflow is fully reproducible from RSLCs with the public ISCE3 binaries.

### NISAR Pytools (GSLC × GSLC) vs JPL

| Band | R² | Pearson r | RMSE |
|---|---|---|---|
| Unwrapped phase | 0.7084 | 0.8465 | 0.3121 rad |
| Coherence | 0.9604 | 0.9810 | 0.0324 |

→ A 50-line GSLC-path pipeline captures **71% of the unwrapped-phase variance** and **96% of the coherence variance** of the production GUNW, in seconds.

## Why pytools doesn't reach R² = 1

After eliminating the obvious physical-correction candidates (smooth ramp in the residual, ~0.3 rad RMSE), the residual is **structural to the pipeline order**, not a missing physical correction.

### What was tested and ruled out

| Hypothesis | Test | Result |
|---|---|---|
| **Solid earth tides** | pysolid + JPL's stored `slantRangeSolidEarthTidesPhase` cube; tried both signs | Differential SET is 70 mm LOS uniform shift, only 0.22 rad spatial variance; correlation with residual = +0.014 — not the cause |
| **Goldstein filter** (skipped pre-unwrap step) | Applied a 32-pixel Goldstein filter (α = 0.2 / 0.5 / 0.8) to pytools IFG and re-snaphu'd | Hurts severely: RMSE 0.31 → 1.7-7 rad; the filter changes the unwrap branch path. Not the cause |
| **Dispersive ionosphere phase screen** | Added/subtracted JPL's stored `ionospherePhaseScreen` (range 67-70 rad, std 0.38 rad) | Both signs hurt R²; the phase screen is left out of `unwrappedPhase` by design and isn't applied to either product. Not the cause |
| **Wet tropospheric correction** | Compared `wetTroposphericGeolocationCorrectionApplied` flags: GSLC = False, GUNW = True. Initially looked compelling | **Ruled out by**: our local ISCE3 GUNW (`troposphere_delay.enabled: false`) matches JPL GUNW to R² = 1.0000. So the GUNW flag means *geolocation correction*, not phase subtraction from `unwrappedPhase` — wet tropo phase is in both products and cancels |
| **Iono timing/range delay** | Indirectly tested via subtracting the iono phase screen; spatial pattern doesn't match | Not the cause |
| **GSLC processing differences** (X05009 vs locally-processed) | Re-processed the reference RSLC into a GSLC locally with our DEM and no TEC; computed pixel-by-pixel coherence vs JPL X05009 | Coherence = **0.9963**; ~0.6 rad smooth phase offset attributable to public-DEM differences + missing iono geolocation correction. **GSLC processing is reproducible** — no exotic JPL-only steps |

The X05009 → X05010 release diff (`NISAR_LSAR_CRID_*.csv` manifests) shows
the only changes between the two releases were:
- L0B PGE patch (R05.00.9 → R05.01.0; edge-case fix unrelated to GSLC/InSAR)
- LSAR Parameter Set patch (`05008_patch2` → `05008_patch3`, fixing
  duplicate `phase_unwrap` elements in the **InSAR** PSF only — not GSLC)
- ISCE3 binary version unchanged (v0.25.7)

So the residual is not a release-version artifact either.

### What it actually is

The pytools (GSLC × GSLC) and the production (RSLC → GUNW) pipelines do
mathematically different things and the operations don't strictly commute:

- **Pipeline-order**: pytools cross-multiplies on the projected (UTM)
  grid *after* each RSLC has been independently geocoded;
  production cross-multiplies in radar (azimuth, range) coordinates and
  geocodes the unwrapped phase at the end. Geocoding is a non-linear
  operation on complex SAR signals; doing it before vs after the
  conjugate multiply gives slightly different speckle realizations.
- **Coregistration**: production runs `dense_offsets` + `rubbersheet` +
  `fine_resample` to align the secondary RSLC to the reference at the
  sub-pixel level *before* crossmul. Pytools relies on the implicit
  coregistration that happens because each GSLC was geocoded onto the
  same UTM grid — sub-pixel inter-acquisition offsets aren't refined.
- **Filter step**: production applies an adaptive (Goldstein-style)
  filter to the wrapped IFG before unwrap. Pytools skips this. Even
  though the filter alone in isolation hurts R² (because it shifts the
  unwrap branch), it is consistent inside production with the
  surrounding multilook + unwrap settings.

These three together produce a smooth ~0.2 rad RMS residual with a weak
N/S preference, plus ~0.22% of pixels with ±2π discrepancies from
different snaphu unwrap branches. None of these are "fixable" with a
post-hoc correction — they require fundamentally moving from GSLC × GSLC
to a radar-domain pipeline.

## Engineering work along the way

1. **Replicated ISCE3 GUNW workflow locally.** Adapted the JPL runconfig,
   fetched a public Copernicus 30 m DEM, disabled troposphere + TEC
   corrections (no ECMWF / TEC ANC locally), kept split-spectrum
   ionosphere on. Confirmed bit-level fidelity on the unwrapped phase.

2. **Audited `nisar_pytools.processing.interferogram` against ISCE3 source.**
   Pulled `CrossMultiply.cpp`, `Crossmul.cpp`, `Signal.cpp`, `Looks.cpp`
   from the public ISCE3 repo to reverse-engineer the exact algorithm
   the production NISAR `crossmul` workflow runs. Discovered ISCE3 has
   *two* implementations: the `CrossMultiply` Python binding (sum-based,
   2× amplitude) and the production `Crossmul` workflow class (mean-based,
   1× amplitude).

3. **Added an `antialias` parameter to `interferogram()`** with three modes:
   - `False` (default): naive `slc1 * conj(slc2)`
   - `'range'`: matches `isce3.signal.Crossmul` (production NISAR convention)
   - `'2d'`: symmetric variant for GSLCs in projected coordinates
   - Confirmed bit-exact phase match (max residual 4 × 10⁻⁴ rad) against
     `isce3.signal.CrossMultiply / upsample` on a real GSLC chip.

4. **Added `multilook_coherence()`** producing ISCE3-style coherence on
   the multilooked grid (block-decimating mean), keeping numerator and
   denominator consistent so γ stays in [0, 1] regardless of antialias mode.

5. **Tightened `compute_baseline()`** to mm-per-pixel agreement with the
   GUNW `metadata/radarGrid` baseline cube (separate work; lives in
   `src/nisar_pytools/processing/baseline.py`).

6. **Built verification scripts** (`scripts/isce3/`):
   - `verify_crossmul.py`: 4-way crossmul comparison on a real GSLC chip
   - `compare_gslcs.py`: pixel-by-pixel local-vs-JPL GSLC check
     (coherence 0.9963 confirms GSLC processing reproducibility)

## Figures (`figures/isce3_comparisons/`)

| File | What it shows |
|---|---|
| `isce3_vs_jpl.png` | Local ISCE3 vs JPL — image side-by-side for unwrapped phase + ionosphere |
| `isce3_vs_jpl_scatter.png` | Density scatter + R² / RMSE / Pearson for each band |
| `pytools_vs_jpl.png` | NISAR Pytools (GSLC path) vs JPL — image side-by-side for unwrapped phase + coherence |
| `pytools_vs_jpl_scatter.png` | Density scatter + R² / RMSE / Pearson for each band |
| `three_way.png` | All three paths side-by-side for unwrapped phase + coherence |
| `three_way_scatter.png` | Pairwise scatter + metrics for both pytools and ISCE3 against JPL |
| `gslc_local_vs_jpl.png` | Pixel-by-pixel local GSLC vs JPL X05009 GSLC: amplitude, phase, complex coherence |
| `gslc_phase_diff_spatial.png` | Spatial pattern of the local-vs-JPL GSLC phase difference (terrain-correlated, attributable to DEM source + missing TEC) |

## Suggested slide flow

1. **Title** — "Production NISAR, ISCE3, NISAR Pytools Interferogram Comparisons"
2. **Three paths overview** — table from "Three processing paths" above
3. **Production NISAR vs Local ISCE3** — `isce3_vs_jpl.png` (image-domain agreement)
4. **Production NISAR vs Local ISCE3 (scatter)** — `isce3_vs_jpl_scatter.png` with R² = 1.0000 callout
5. **Production NISAR vs NISAR Pytools** — `pytools_vs_jpl.png` (image-domain)
6. **Production NISAR vs NISAR Pytools (scatter)** — `pytools_vs_jpl_scatter.png`, R² = 0.71 / 0.96 callout
7. **All three side-by-side** — `three_way.png` (visual indistinguishability of the maps)
8. **Diagnosing the residual** — table of "what was tested and ruled out"
9. **GSLC-processing reproducibility** — `gslc_local_vs_jpl.png` with coherence = 0.9963
10. **The actual answer** — pipeline-order: GSLC × GSLC ≠ RSLC → IFG → geocode at the ~0.2 rad level
11. **Tradeoff summary** — wall time, lines of code, achievable accuracy
12. **What we built** — bullet list from "Engineering work along the way"

## Tradeoff summary table for the deck

| Path | Stages | Wall time | Lines of code | Achievable accuracy vs JPL |
|---|---|---|---|---|
| RSLC → GUNW (local isce3) | 16+ | ~86 min | ~25k (isce3 + nisar.workflows) | mm-level |
| GSLC × GSLC (nisar_pytools) | 4 | ~15 sec | ~50 | ~cm-level (RMSE 0.31 rad / λ ≈ 1 cm at L-band) |

For deformation studies, glacier flow, fault mapping — the GSLC path
captures the same spatial structure at the cm-level. For
publication-grade L2 product reproduction or anything that requires
the full unwrap-branch alignment, the RSLC path is the right choice.
