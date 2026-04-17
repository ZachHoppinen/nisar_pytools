# nisar_pytools

Open source Python tools for working with NISAR datasets.

## About

`nisar_pytools` provides utilities for searching, downloading, reading, and processing data products from NASA's [NISAR](https://nisar.jpl.nasa.gov/) (NASA-ISRO Synthetic Aperture Radar) mission.

### Supported Products

- **GSLC** - Geocoded Single Look Complex
- **GUNW** - Geocoded Unwrapped Interferogram

Additional NISAR product types will be added over time.

## Getting Started

### Prerequisites

- Python 3.10+
- [Miniforge](https://github.com/conda-forge/miniforge) (recommended)
- NASA Earthdata login (for downloading from ASF)

### Installation

**From PyPI:**
```sh
pip install nisar-pytools
```

With optional extras:
```sh
pip install nisar-pytools[dem]       # DEM fetching (dem_stitcher)
pip install nisar-pytools[dolphin]   # dolphin InSAR time-series prep
pip install nisar-pytools[viz]       # Visualization (matplotlib)
pip install nisar-pytools[all]       # Everything
```

**From source (for development):**
```sh
git clone https://github.com/zmhoppinen/nisar_pytools.git
cd nisar_pytools
mamba env create -f environment.yml
conda activate nisar_pytools
```

## Usage

### Search and Download

```python
from nisar_pytools import find_nisar, download_urls

# find_nisar() searches ASF for NISAR products. Parameters:
#   aoi          - area of interest: [xmin, ymin, xmax, ymax], shapely geometry,
#                  or dict like {"west": -115, "south": 43, "east": -114, "north": 44}
#   start_date   - start of temporal search window (ISO string or datetime)
#   end_date     - end of temporal search window

#   product_type - "GSLC", "GUNW", "RSLC", "GCOV", "RIFG", "RUNW", "ROFF", "GOFF"
#   path_number  - relative orbit / track number (optional)
#   frame        - frame number (optional)
#   direction    - "ASCENDING" or "DESCENDING" (optional)
#   include_qa   - if True, include QA files (default False)

# Broad search: find all GSLC data over an area (any track, any direction)
all_gslcs = find_nisar(
    aoi=[-115, 43, -114, 44],
    start_date="2025-06-01",
    end_date="2025-12-01",
    product_type="GSLC",
)

# Narrow search: specific track and direction for time-series analysis
track_77 = find_nisar(
    aoi=[-115, 43, -114, 44],
    start_date="2025-06-01",
    end_date="2025-12-01",
    product_type="GSLC",
    path_number=77,
    direction="ASCENDING",
)

# Search for GUNW interferograms instead of SLCs
gunws = find_nisar(
    aoi=[-115, 43, -114, 44],
    start_date="2025-11-01",
    end_date="2026-02-01",
    product_type="GUNW",
)

# Download in parallel with automatic HDF5 validation
fps = download_urls(track_77, "local/gslcs/")
```

### Read a Single File

```python
from nisar_pytools import open_nisar
from nisar_pytools.utils.metadata import get_slc, get_orbit_info, get_acquisition_time

dt = open_nisar("NISAR_L2_PR_GSLC_...h5")

# Quick access to a polarization channel
hh = get_slc(dt, polarization="HH")           # lazy, CRS set
hv = get_slc(dt, polarization="HV")
hh_b = get_slc(dt, polarization="HH", frequency="frequencyB")

# Metadata without navigating the tree
print(get_acquisition_time(dt))  # 2025-11-03 12:46:15
print(get_orbit_info(dt))        # {'track_number': 77, 'frame_number': 24, ...}
print(hh.rio.crs)                # EPSG:32611

# Or access the full DataTree directly
freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
```

### Stack GSLCs into a Time Series

```python
from nisar_pytools import stack_gslcs

# Stack multiple same-track GSLCs into a (time, y, x) DataArray
stack = stack_gslcs(
    ["gslc_date1.h5", "gslc_date2.h5", "gslc_date3.h5"],
    frequency="frequencyA",
    polarization="HH",
)
# Sorted by time, grid-validated, dask-backed, CRS assigned
```

### SAR Processing

```python
from nisar_pytools.processing import (
    interferogram, coherence, multilook, unwrap, calculate_phase
)

# Interferogram (validates matching grids)
ifg = interferogram(slc1, slc2)

# Multilooked interferogram
ml_ifg = multilook(ifg, looks_y=4, looks_x=4)

# Coherence estimation
coh = coherence(slc1, slc2, window_size=11)

# Phase unwrapping with SNAPHU
unw, conncomp = unwrap(ifg, coh, nlooks=20.0)
```

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

## Roadmap

- [x] Lazy HDF5 reader returning xarray DataTree with CRS
- [x] Prep dolphin to run GSLC to dolphin ready yaml + geotiffs
- [x] ASF search and parallel download with validation
- [x] GSLC time-series stacking
- [x] Interferogram, coherence, multilooking, phase extraction
- [x] Phase unwrapping (SNAPHU)
- [x] Phase linking (EMI with SHP selection)
- [x] Polarimetric decomposition (H-A-alpha)
- [x] Local incidence angle computation
- [x] Visualization helpers (amplitude, phase, interferogram, coherence)
- [ ] Support for additional NISAR product types

## Contributing

Contributions are welcome! Please fork the repo and open a pull request, or open an issue to suggest improvements.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
