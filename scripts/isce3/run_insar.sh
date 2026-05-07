#!/usr/bin/env bash
# Run the NISAR L2 InSAR workflow end-to-end on the local RSLC pair.
#
# Inputs (paths set in runconfig.yaml):
#   - local/rslc/NISAR_L1_PR_RSLC_004_077_..._20251103...h5  (reference)
#   - local/rslc/NISAR_L1_PR_RSLC_005_077_..._20251115...h5  (secondary)
#   - scripts/isce3/inputs/dem.tif                            (Copernicus 30m)
#
# Outputs (under scripts/isce3/output/):
#   - product.h5  (the GUNW we just produced)
#   - rifg.h5, runw.h5, gunw.h5 (intermediates)
#
# Notes:
#   - Multi-hour runtime expected on CPU. Stream the log to watch progress.
#   - Troposphere correction: disabled (no ECMWF locally)
#   - Ionosphere: split-spectrum (main_diff_ms_band) -- enabled

set -e
cd "$(dirname "$0")/../.."

PYTHON=/Users/zmhoppinen/miniforge3/envs/isce3/bin/python

mkdir -p scripts/isce3/output scripts/isce3/scratch

echo "Starting nisar.workflows.insar at $(date)"
$PYTHON -m nisar.workflows.insar scripts/isce3/runconfig.yaml 2>&1 | tee scripts/isce3/scratch/run.log
echo "Finished at $(date)"
