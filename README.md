# nisar_pytools

Open source Python tools for working with NISAR datasets.

## About

`nisar_pytools` opens NASA [NISAR](https://nisar.jpl.nasa.gov/) (NASA-ISRO
Synthetic Aperture Radar) HDF5 products as lazy `xarray` objects, so you can
search for a product, open it, and pull out the layer you want without
learning the HDF5 layout.

Reading is the common case and it needs nothing beyond the base install.
Building products (interferograms, RSLC to GUNW, dolphin prep, polarimetry)
lives in the [processing reference](docs/processing.md).

### Supported Products

- **GSLC** - Geocoded Single Look Complex
- **GUNW** - Geocoded Unwrapped Interferogram

`find_nisar` searches for other types (RSLC, GCOV, RIFG, RUNW, ROFF, GOFF),
and RSLC products can be streamed and cropped. Full reader support for
additional types will be added over time.

## Quick start

```python
from nisar_pytools import open_nisar
from nisar_pytools.utils.metadata import get_gunw

dt = open_nisar("NISAR_L2_PR_GUNW_...h5")

# One layer, all of its variables, on one grid, invalid pixels blanked.
ds = get_gunw(dt)
ds.unwrappedPhase.plot()
ds.coherenceMagnitude.plot()
```

```
<xarray.Dataset>
Dimensions:  (y: 4347, x: 4410)
Data variables:
    unwrappedPhase                    (y, x) float32
    coherenceMagnitude                (y, x) float32
    connectedComponents               (y, x) float32
    ionospherePhaseScreen             (y, x) float32
    ionospherePhaseScreenUncertainty  (y, x) float32
```

Nothing is read from disk until you plot or `.compute()`, so this works the
same on a 2 GB product as on a small one.

## Getting Started

### Prerequisites

- Python 3.10+
- [Miniforge](https://github.com/conda-forge/miniforge) (recommended)
- NASA Earthdata login (for downloading from ASF)

### Installation

```sh
pip install nisar-pytools
```

With optional extras:

```sh
pip install nisar-pytools[dem]       # DEM fetching (dem_stitcher)
pip install nisar-pytools[dolphin]   # dolphin InSAR time-series prep
pip install nisar-pytools[viz]       # Visualization (matplotlib)
pip install nisar-pytools[all]       # Everything available from PyPI
```

**From source (for development):**

```sh
git clone https://github.com/zmhoppinen/nisar_pytools.git
cd nisar_pytools
mamba env create -f environment.yml
conda activate nisar_pytools
```

The bundled `environment.yml` also installs the ISCE3 RSLC to GUNW stack.
You only need that for the [processing reference](docs/processing.md);
reading products does not require it.

## Reading products

### How a NISAR file is laid out

`open_nisar` returns an `xarray.DataTree` that mirrors the HDF5 groups, so a
path in the tree is the same path you would use in `h5dump` or the NISAR
product spec. NISAR stores nothing at the root, so printing the tree shows
`Data variables: (0)` and a single group. That is expected. The data is one
level down:

```
/science/LSAR/identification            <- track, frame, times, polygon (attrs)
/science/LSAR/GUNW/grids/frequencyA/
        unwrappedInterferogram/HH       <- unwrapped phase, coherence, ...
        wrappedInterferogram/HH         <- wrapped interferogram, coherence
        pixelOffsets/HH                 <- along-track / slant-range offsets
```

To see every group that actually holds arrays:

```python
[node.path for node in dt.subtree if node.dataset.data_vars]
```

The accessors below save you from walking this by hand, but the tree is
always there if you want it:

```python
freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
```

### GUNW layers

`get_gunw` returns a whole layer by default, since the variables in one layer
share a grid:

```python
from nisar_pytools.utils.metadata import get_gunw

dt = open_nisar("NISAR_L2_PR_GUNW_...h5")

# Defaults: unwrappedInterferogram / HH, masked.
ds = get_gunw(dt)
ds.unwrappedPhase
ds.coherenceMagnitude
ds.connectedComponents

# A single variable comes back as a DataArray.
unw = get_gunw(dt, variable="unwrappedPhase")

# Raw samples, nothing blanked.
ds_raw = get_gunw(dt, valid_mask=False)
```

The three layers are **not** on a common grid. In a typical product the
unwrapped interferogram and pixel offsets are posted at 80 m while the
wrapped interferogram is at 20 m, so they cannot be merged into one Dataset
without resampling. Ask for one layer at a time:

```python
wrapped = get_gunw(dt, layer="wrappedInterferogram")   # 20 m grid
offsets = get_gunw(dt, layer="pixelOffsets")           # 80 m grid
```

### GSLC channels

```python
from nisar_pytools.utils.metadata import get_slc

dt = open_nisar("NISAR_L2_PR_GSLC_...h5")

hh = get_slc(dt, polarization="HH")                        # lazy, CRS set, masked
hv = get_slc(dt, polarization="HV")
hh_b = get_slc(dt, polarization="HH", frequency="frequencyB")
hh_raw = get_slc(dt, polarization="HH", valid_mask=False)   # opt out of masking
```

### Metadata

```python
from nisar_pytools.utils.metadata import (
    get_orbit_info, get_bounding_polygon,
)

print(get_orbit_info(dt))         # {'track_number': 77, 'frame_number': 24, ...}
print(get_bounding_polygon(dt))   # shapely Polygon in WGS84
print(hh.rio.crs)                 # EPSG:32611
```

`get_acquisition_time` returns both passes, so it reads the same way whether
the product came from one acquisition or two:

```python
from nisar_pytools.utils.metadata import get_acquisition_time

t = get_acquisition_time(gslc)
t.reference                       # 2025-11-03 12:46:15
t.secondary                       # None, a GSLC is one acquisition

t = get_acquisition_time(gunw)
t.reference                       # 2026-02-15 12:11:20
t.secondary                       # 2026-02-27 12:11:20
(t.secondary - t.reference).days  # 12, the temporal baseline
```

### Valid mask semantics

When `valid_mask=True` (the default), invalid pixels become `NaN`.
Integer-typed variables (e.g. `connectedComponents`, uint16) are promoted
to float so `NaN` fits.

- **GSLC** `mask` (uint8) encodes the subswath number for valid samples.
  - `0` = at least one RSLC pixel in the interpolation window was
    partially-focused or invalid, so the sample is dropped.
  - `1..N` = valid subswath number, kept.
  - `255` = outside the radar acquisition extent, dropped.
- **GUNW** `mask` (uint8) is a three-digit `WRS` integer combining a water
  flag and the reference/secondary RSLC subswath numbers.
  - `W` = reference water flag (1 = water, 0 = land). Water pixels are
    **kept**, so mask water separately if you need to drop it.
  - `R` = reference RSLC subswath number; `0` means the sample is invalid
    in the reference, so it is dropped.
  - `S` = secondary RSLC subswath number; `0` means the sample is invalid
    in the secondary, so it is dropped.
  - `255` = fill outside the acquisition extent, dropped.

  In code: kept where `(mask // 10) % 10 != 0 and mask % 10 != 0 and mask != 255`.

  Each GUNW layer carries its own mask on its own grid; the layer mask is
  shared across polarizations.

## Command line

### GeoTIFF export

Quick export of commonly used bands from a NISAR HDF5, handy for pulling into
QGIS. Installed with the package as the `nisar_pytools` console script; run
`nisar_pytools to-geotiff --help` for the full band catalog.

```bash
# Default-all: write every default band for the product, next to the .h5
# GUNW -> unwrapped_phase, wrapped_phase, coherence, ionosphere
# GSLC -> amplitude (10*log10(|SLC|^2) in dB)
nisar_pytools to-geotiff NISAR_L2_PR_GUNW_...h5

# One band, explicit polarization, custom output directory
nisar_pytools to-geotiff NISAR_L2_PR_GUNW_...h5 \
    --band unwrapped_phase --pol HH --output-dir /tmp/gunw_tifs

# Subset a multi-GB GSLC to a lat/lon AOI -- streamed, low memory
nisar_pytools to-geotiff NISAR_L2_PR_GSLC_...h5 \
    --bbox-wgs84 -118.5 41.0 -118.3 41.2

# Same crop in the file's native CRS (UTM meters here)
nisar_pytools to-geotiff NISAR_L2_PR_GSLC_...h5 \
    --bbox 380000 4540000 400000 4565000

# GSLC amplitude on frequency B
nisar_pytools to-geotiff NISAR_L2_PR_GSLC_...h5 --freq B
```

Outputs are tiled GeoTIFFs named `<h5_stem>_<band>_<frequency>_<pol>.tif`.
Writes stream chunk-by-chunk via dask + rioxarray, so a full-resolution
41 GB GSLC processes with ~330 MB peak memory.

### File summary

Product type/version, file size, acquisition time(s), track/frame/direction,
polarizations, per-grid shape and resolution, native + WGS84 extent, and (for
GUNW) coherence / unwrapped-phase stats, connected-component summary, and
perpendicular + parallel baselines from the radarGrid cube.

```bash
nisar_pytools info NISAR_L2_PR_GUNW_...h5
nisar_pytools info NISAR_L2_PR_GSLC_...h5 --json
```

## Search and download

```python
from nisar_pytools import find_nisar, download_urls

# aoi          - [xmin, ymin, xmax, ymax], shapely geometry, or
#                {"west": -115, "south": 43, "east": -114, "north": 44}
# product_type - "GSLC", "GUNW", "RSLC", "GCOV", "RIFG", "RUNW", "ROFF", "GOFF"
# path_number  - relative orbit / track number (optional)
# frame        - frame number (optional)
# direction    - "ASCENDING" or "DESCENDING" (optional)
# maturity     - "validated", "provisional" or "beta" (optional)
# include_qa   - if True, include QA files (default False)

all_gslcs = find_nisar(
    aoi=[-115, 43, -114, 44],
    start_date="2025-06-01",
    end_date="2025-12-01",
    product_type="GSLC",
)

# Narrow to one track and direction for time-series analysis
track_77 = find_nisar(
    aoi=[-115, 43, -114, 44],
    start_date="2025-06-01",
    end_date="2025-12-01",
    product_type="GSLC",
    path_number=77,
    direction="ASCENDING",
)

# Download in parallel with automatic HDF5 validation
fps = download_urls(track_77, "local/gslcs/")
```

NISAR products are published at several processing maturities, which cover
different date ranges rather than reprocessing the same acquisitions. Leaving
`maturity` unset searches all of them and warns when results span more than
one.

## Stack GSLCs into a time series

```python
from nisar_pytools import stack_gslcs

stack = stack_gslcs(
    ["gslc_date1.h5", "gslc_date2.h5", "gslc_date3.h5"],
    frequency="frequencyA",
    polarization="HH",
)
# Sorted by time, grid-validated, dask-backed, CRS assigned
```

## Processing

Interferogram formation, coherence, multilooking, unwrapping, the production
RSLC to GUNW pipeline, dolphin prep, phase linking, polarimetric
decomposition, and local incidence angle are documented in the
[processing reference](docs/processing.md).

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
