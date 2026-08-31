# Processing reference

Interferometric and polarimetric processing with `nisar_pytools`. Most users
who only need to open a product and pull out layers can stay in the
[README](../README.md); this page is for building products rather than
reading them.

Everything here beyond `interferogram`/`coherence`/`multilook` needs the
isce3 stack from `environment.yml`. See
[Installation](../README.md#installation).

## Command line

### RSLC → GUNW (production InSAR pipeline)

Wrapper around `nisar.workflows.insar` (the production isce3 InSAR
workflow) that takes an RSLC pair and produces a GUNW. Requires the
isce3 stack from environment.yml (see Installation above).

```bash
# Minimal: auto-fetch DEM, auto-detect UTM zone + bbox, production-spec runconfig
nisar_pytools rslc-to-gunw ref.h5 sec.h5 --output-dir gunw_run/

# Bring your own DEM + crop output to a smaller AOI in UTM 11N
nisar_pytools rslc-to-gunw ref.h5 sec.h5 \
    --output-dir gunw_run/ \
    --dem dem.tif \
    --bbox 464000 4884080 544000 4964080 \
    --epsg 32611

# Use a custom runconfig (overrides the bundled default)
nisar_pytools rslc-to-gunw ref.h5 sec.h5 --output-dir gunw_run/ --runconfig my.yaml

# Force a full re-run, ignoring previously-cached step outputs
nisar_pytools rslc-to-gunw ref.h5 sec.h5 --output-dir gunw_run/ --restart
```

Outputs `gunw_run/product.h5` (the GUNW), with intermediates and the
log under `gunw_run/scratch/`. Multi-hour wall time on CPU.

## Python

### Basic SAR Processing

```python
from nisar_pytools.processing import (
    interferogram, coherence, multilook, unwrap, calculate_phase
)

# Interferogram (validates matching grids)
ifg = interferogram(slc1, slc2)
ifg_aa = interferogram(slc1, slc2, antialias="range")  # ISCE3 RSLC convention
ifg_2d = interferogram(slc1, slc2, antialias="2d")     # symmetric, right for GSLCs

# Multilooked interferogram
ml_ifg = multilook(ifg, looks_y=4, looks_x=4)

# Coherence estimation
coh = coherence(slc1, slc2, window_size=11)

# Phase unwrapping with SNAPHU
unw, conncomp = unwrap(ifg, coh, nlooks=20.0)
```

#### Crossmul antialiasing

`interferogram()` accepts an `antialias` argument with three modes:

| `antialias` | What it does | When to use |
|---|---|---|
| `False` (default) | Naive `slc1 * conj(slc2)` | Followed by multilook (which absorbs alias noise). Fast, dask-friendly. |
| `'range'` | FFT 2× upsample → multiply → mean-of-pairs along axis=1 only. Matches the production NISAR `crossmul` workflow (`isce3.signal.Crossmul`) to numerical roundoff. Output amplitude is ~1× naive. | RSLCs in radar coordinates, where range is critically sampled and azimuth is over-sampled. |
| `'2d'` | Same algorithm applied symmetrically along both axes. Output amplitude is ~1× naive. | GSLCs (or any product on a projected x–y grid) where the SAR bandlimit is rotated diagonally and neither axis is privileged. |

The naive (default) and `'range'` outputs differ at full resolution by ~0.75 rad phase std on a representative GSLC chip; after 16×16 multilook the residual drops below the science noise floor at L-band, so the default is fine for the typical RSLC→multilook→unwrap pipeline. Set `antialias` only if you need the full-resolution wrapped phase clean of alias noise.

> Note: `isce3.signal.CrossMultiply` (a separate, simpler ISCE3 class with a numpy-array API) uses `multilookSummed` instead of mean and gives 2× amplitude. We deliberately match the NISAR production `Crossmul` (mean) so coherence calculations stay bounded to [0, 1].

#### Coherence — sliding window vs multilooked

Two coherence functions for two use cases:

```python
from nisar_pytools.processing import coherence, multilook_coherence

# Sliding-window estimate at SLC resolution (boxcar or gaussian)
coh = coherence(slc1, slc2, window_size=11)

# ISCE3 production-style: non-overlapping mean blocks, output on the
# multilooked grid -- matches RIFG/RUNW/GUNW coherenceMagnitude bands
coh_ml = multilook_coherence(slc1, slc2, looks_y=16, looks_x=5)
coh_aa = multilook_coherence(slc1, slc2, looks_y=16, looks_x=5, antialias="range")
```

`coherence()` is for visualization and analysis at full SLC resolution. `multilook_coherence()` produces the same estimator on the multilooked grid that the ISCE3 GUNW reports — numerator and denominator are kept consistent (both mean-based, both share the same multilook factors), so values stay bounded to [0, 1] regardless of antialias mode.

### RSLC → GUNW (production InSAR pipeline)

`rslc_to_gunw` wraps `nisar.workflows.insar` to take an L1 RSLC pair and
produce an L2 GUNW. The bundled `environment.yml` installs the full
isce3 stack; users on a custom env need at minimum:

```sh
mamba install -c conda-forge isce3 libgdal-hdf5 snaphu pygrib pyaps3 raider pysolid netcdf4
pip install 'nisar-pytools[isce3]'   # for the rest via pip if you skip mamba
```

Minimal call — auto-fetches a Copernicus 30 m DEM, auto-detects the UTM
zone + bbox from the reference RSLC, applies the bundled production-spec
runconfig (JPL P05023 settings, 5×6 / 13×16 looks, full coregistration,
split-spectrum ionosphere on, troposphere off):

```python
from nisar_pytools.processing import rslc_to_gunw

gunw = rslc_to_gunw(
    reference_rslc="NISAR_L1_PR_RSLC_..._20251103...h5",
    secondary_rslc="NISAR_L1_PR_RSLC_..._20251115...h5",
    output_dir="gunw_run/",
)
print(gunw)  # gunw_run/product.h5
```

More control — supply your own DEM, crop the output, and tweak processing
parameters via the merged-with-defaults `overrides` dict:

```python
gunw = rslc_to_gunw(
    "ref.h5", "sec.h5", "gunw_run/",
    dem_file="dem.tif",
    aoi_bbox_utm=(464000, 4884080, 544000, 4964080),  # (xmin, ymin, xmax, ymax)
    output_epsg=32611,
    overrides={"runconfig": {"groups": {"processing": {
        # Faster + lower-res run: bigger looks, no rubbersheet
        "crossmul":      {"range_looks": 10, "azimuth_looks": 12},
        "phase_unwrap":  {"range_looks": 26, "azimuth_looks": 32},
        "rubbersheet":   {"enabled": False},
        "fine_resample": {"enabled": False},
    }}}},
)
```

Or pass an entirely user-supplied runconfig (paths and AOI bbox are still
auto-injected on top):

```python
gunw = rslc_to_gunw("ref.h5", "sec.h5", "gunw_run/", runconfig="my_runconfig.yaml")
```

Outputs land in `gunw_run/`:

- `product.h5` — the GUNW
- `runconfig.yaml` — the resolved runconfig actually fed to the workflow
- `scratch/` — intermediate RIFG/RUNW HDF5s, rdr2geo cache, log

Set `restart=True` to ignore cached step outputs and re-run from
scratch (forwarded to `nisar.workflows.persistence.Persistence`).

### Prepare GSLCs for dolphin

[dolphin](https://github.com/isce-framework/dolphin) is an open-source InSAR
time-series tool (phase linking, unwrapping, network inversion) from the
OPERA/ISCE framework. `prep_dolphin` crops your GSLCs to an AOI, exports them
as complex GeoTIFFs, and generates a ready-to-run dolphin config YAML.

```python
from pathlib import Path
from nisar_pytools.processing import prep_dolphin

gslc_files = sorted(Path("gslcs/").glob("NISAR_L2_PR_GSLC_*.h5"))

config = prep_dolphin(
    gslc_paths=gslc_files,
    out_dir="dolphin_run/",
    aoi_wgs84=(-110.54, 44.65, -109.63, 45.18),    # Greater Yellowstone
    skip_dates={"20251104"},                          # drop early-season date
    dolphin_overrides=[
        (["phase_linking", "half_window", "y"], 20),
        (["phase_linking", "half_window", "x"], 20),
        (["output_options", "strides", "y"], 5),
        (["output_options", "strides", "x"], 5),
        (["unwrap_options", "unwrap_method"], "spurt"),
    ],
)
# Logs: "Ready. Review the config, then run:
#          dolphin run dolphin_run/dolphin_config.yaml"
```

You can also use `crop_gslc_to_tif` standalone to extract a single GSLC:

```python
from nisar_pytools.processing import crop_gslc_to_tif

crop_gslc_to_tif("NISAR_L2_PR_GSLC_...h5", "out.tif",
                 bbox_utm=(540000, 4950000, 560000, 4970000))
```

### Phase Linking

```python
from nisar_pytools.processing import phase_link

# EMI phase linking on a GSLC stack
linked, temporal_coh = phase_link(stack, search_window=11, confidence=0.95)
```

### Polarimetric Decomposition

Requires a **quad-pol acquisition** (HH + HV + VV). NISAR acquires quad-pol
over specific regions — check `listOfPolarizations` in the frequency group
to confirm your data has all three channels. Dual-pol (HH + HV only) data
cannot be used for H-A-alpha decomposition.

```python
from nisar_pytools import open_nisar
from nisar_pytools.utils.metadata import get_slc
from nisar_pytools.processing import h_a_alpha

dt = open_nisar("NISAR_L2_PR_GSLC_...h5")

# Extract the three quad-pol channels (must call .compute() for in-memory)
hh = get_slc(dt, "HH").compute()
hv = get_slc(dt, "HV").compute()
vv = get_slc(dt, "VV").compute()  # only available in quad-pol mode

# H-A-alpha decomposition (Cloude-Pottier)
ds = h_a_alpha(hh, hv, vv)
# Returns Dataset with: entropy, anisotropy, alpha, mean_alpha
```

### Local Incidence Angle

```python
import numpy as np
from nisar_pytools import open_nisar
from nisar_pytools.utils.local_incidence_angle import local_incidence_angle

# Open a GSLC or GUNW — both have radarGrid with LOS vectors
dt = open_nisar("NISAR_L2_PR_GSLC_...h5")

# Extract LOS vectors from the radarGrid metadata
rg = dt["science/LSAR/GSLC/metadata/radarGrid"].dataset
los_x = np.asarray(rg["losUnitVectorX"])  # shape: (n_heights, ny, nx)
los_y = np.asarray(rg["losUnitVectorY"])
los_z = np.sqrt(np.maximum(1.0 - los_x**2 - los_y**2, 0.0))  # derive Z

heights = np.asarray(rg.coords["z"])      # height layers
x_rg = np.asarray(rg.coords["x"])         # radarGrid x coordinates
y_rg = np.asarray(rg.coords["y"])         # radarGrid y coordinates
epsg = int(rg.attrs.get("projection"))     # CRS

# Compute LIA using a DEM (must be in projected CRS, same as radarGrid)
lia = local_incidence_angle(dem, los_x, los_y, los_z, heights, x_rg, y_rg, epsg=epsg)
# lia is an xr.DataArray in degrees with CRS set
```
