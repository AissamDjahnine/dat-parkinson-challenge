#!/usr/bin/env bash
# Shared preprocessing for all three models (~50 min, CPU). Run once.
# Expects data/niftis/*.nii.gz and data/train_labels.csv at the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-uv run python}

$PY scripts/inventory.py                                               # header inventory
$PY scripts/artifact_audit.py                                          # per-scan intensity stats
$PY scripts/make_folds.py                                              # grouped folds - never regenerate
$PY scripts/build_cache.py --voxel-mm 2.0    --crop-mm 128 --tag v1    # 64^3, SBR features
$PY scripts/build_cache.py --voxel-mm 1.3333 --crop-mm 128 --tag hires # 96^3, all CNNs
