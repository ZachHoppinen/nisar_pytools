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

# Search ASF for GSLC products over an area of interest
urls = find_nisar(
    aoi=[-115, 43, -114, 44],
    start_date="2025-06-01",
    end_date="2025-12-01",
    product_type="GSLC",
    path_number=77,
    direction="ASCENDING",
)

# Download in parallel with automatic validation
fps = download_urls(urls, "local/gslcs/")
```

### Read a Single File

```python
from nisar_pytools import open_nisar

dt = open_nisar("NISAR_L2_PR_GSLC_...h5")

# Access a frequency/polarization group
freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
hh = freq_a["HH"]  # lazy dask-backed DataArray with CRS set

# Coordinates and CRS are assigned automatically
print(hh.rio.crs)  # e.g. EPSG:32611
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

### Phase Linking

```python
from nisar_pytools.processing import phase_link

# EMI phase linking on a GSLC stack
linked, temporal_coh = phase_link(stack, window_size=11, confidence=0.95)
```

### Polarimetric Decomposition

```python
from nisar_pytools.processing import h_a_alpha

# H-A-alpha decomposition from quad-pol SLC channels
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
- [x] GSLC and GUNW support
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
