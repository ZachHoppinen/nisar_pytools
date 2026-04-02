# nisar_pytools

Open source Python tools for working with NISAR datasets.

## About

`nisar_pytools` provides utilities for accessing, reading, and analyzing data products from NASA's [NISAR](https://nisar.jpl.nasa.gov/) (NASA-ISRO Synthetic Aperture Radar) mission.

### Supported Products

- **GSLC** - Geocoded Single Look Complex
- **GUNW** - Geocoded Unwrapped Interferogram

Additional NISAR product types will be added over time.

## Getting Started

### Prerequisites

- Python 3.10+
- [Miniforge](https://github.com/conda-forge/miniforge) (recommended)

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/zmhoppinen/nisar_pytools.git
   cd nisar_pytools
   ```
2. Create the conda environment
   ```sh
   mamba env create -f environment.yml
   conda activate nisar_pytools
   ```

## Usage

```python
from nisar_pytools import open_nisar

# Open a NISAR HDF5 file as a lazy xarray DataTree
dt = open_nisar("NISAR_L2_PR_GSLC_...h5")

# Access a specific frequency group
freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
print(freq_a)

# Data is lazily loaded — compute when needed
phase = freq_a["HH"].values
```

All data arrays are dask-backed and nothing is loaded into memory until explicitly computed. Coordinates (`x`, `y`) are assigned as proper xarray dimension coordinates.

## Roadmap

- [x] Lazy HDF5 reader returning xarray DataTree
- [x] GSLC support
- [x] GUNW support (multi-resolution sub-products)
- [ ] Zarr export utilities
- [ ] Visualization helpers
- [ ] Support for additional NISAR product types

## Contributing

Contributions are welcome! Please fork the repo and open a pull request, or open an issue to suggest improvements.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
