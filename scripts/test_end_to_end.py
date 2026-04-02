"""End-to-end test: search → open → stack → interferogram → coherence → phase.

Uses local GSLC files (already downloaded).
"""

from nisar_pytools import find_nisar, open_nisar, stack_gslcs
from nisar_pytools.processing import interferogram, coherence, calculate_phase, multilook
from nisar_pytools.utils.metadata import get_acquisition_time, get_orbit_info
from nisar_pytools.io._export import to_netcdf
import numpy as np

AOI = [-119.15, 42.85, -114.62, 46.05]
GSLC_DIR = "local/gslcs"
GSLC_FILES = [
    f"{GSLC_DIR}/NISAR_L2_PR_GSLC_004_077_A_024_4005_DHDH_A_20251103T124615_20251103T124650_X05009_N_F_J_001.h5",
    f"{GSLC_DIR}/NISAR_L2_PR_GSLC_005_077_A_024_4005_DHDH_A_20251115T124615_20251115T124650_X05009_N_F_J_001.h5",
]

print("=" * 60)
print("1. SEARCH")
print("=" * 60)
urls = find_nisar(
    aoi=AOI,
    start_date="2025-11-01",
    end_date="2025-11-30",
    product_type="GSLC",
    path_number=77,
    frame=24,
    direction="ASCENDING",
)
print(f"   Found {len(urls)} GSLC URLs")
assert len(urls) >= 2, f"Expected at least 2 URLs, got {len(urls)}"
print("   PASS")

print()
print("=" * 60)
print("2. OPEN SINGLE FILE")
print("=" * 60)
dt = open_nisar(GSLC_FILES[0])
print(f"   Product type: {dt.attrs['product_type']}")
print(f"   Acquisition:  {get_acquisition_time(dt)}")
print(f"   Orbit info:   {get_orbit_info(dt)}")
freq_a = dt["science/LSAR/GSLC/grids/frequencyA"].dataset
print(f"   Freq A vars:  {list(freq_a.data_vars)}")
print(f"   HH shape:     {freq_a['HH'].shape}")
print(f"   HH CRS:       {freq_a['HH'].rio.crs}")
print(f"   HH chunks:    {freq_a['HH'].data.chunksize}")
assert freq_a["HH"].rio.crs is not None, "CRS not set"
print("   PASS")

print()
print("=" * 60)
print("3. STACK GSLCs")
print("=" * 60)
stack = stack_gslcs(GSLC_FILES, frequency="frequencyA", polarization="HH")
print(f"   Stack shape:  {stack.shape}")
print(f"   Stack dims:   {stack.dims}")
print(f"   Times:        {stack.time.values}")
print(f"   CRS:          {stack.rio.crs}")
assert stack.shape[0] == 2, f"Expected 2 time steps, got {stack.shape[0]}"
assert stack.rio.crs is not None, "CRS not set on stack"
print("   PASS")

print()
print("=" * 60)
print("4. INTERFEROGRAM (small subset)")
print("=" * 60)
# Use a small subset to avoid loading 23GB into memory
slc1 = stack.isel(time=0, y=slice(0, 512), x=slice(0, 512)).compute()
slc2 = stack.isel(time=1, y=slice(0, 512), x=slice(0, 512)).compute()
print(f"   Subset shape: {slc1.shape}")

ifg = interferogram(slc1, slc2)
print(f"   IFG shape:    {ifg.shape}")
print(f"   IFG dtype:    {ifg.dtype}")
assert np.iscomplexobj(ifg.values), "Interferogram not complex"
print("   PASS")

print()
print("=" * 60)
print("5. COHERENCE")
print("=" * 60)
coh = coherence(slc1, slc2, window_size=11)
print(f"   Coh shape:    {coh.shape}")
print(f"   Coh range:    [{float(coh.min()):.3f}, {float(coh.max()):.3f}]")
assert coh.min() >= 0 and coh.max() <= 1, "Coherence out of range"
print("   PASS")

print()
print("=" * 60)
print("6. MULTILOOK + PHASE")
print("=" * 60)
ml_ifg = multilook(ifg, looks_y=4, looks_x=4)
phase = calculate_phase(ml_ifg)
print(f"   ML shape:     {ml_ifg.shape}")
print(f"   Phase range:  [{float(phase.min()):.3f}, {float(phase.max()):.3f}] rad")
assert ml_ifg.shape[0] == 512 // 4, "Multilook y dimension wrong"
print("   PASS")

print()
print("=" * 60)
print("7. EXPORT")
print("=" * 60)
out = to_netcdf(coh, "local/test_output/coherence.nc")
print(f"   Saved to:     {out}")
assert out.exists(), "Output file not created"
print("   PASS")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
