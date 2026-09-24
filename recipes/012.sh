#!/usr/bin/env bash
# 012 - 011's 75 CNN checkpoints unchanged; SBR head refitted on a 50-feature multiscale set, and
# an RAS orientation guard at load time. CPU only. Grouped CV 0.2426 / AUROC 0.9622.
# Needs the outputs of recipes/008a.sh and recipes/011.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-uv run python}

$PY scripts/train_sbr_v2.py --arm v2_feat          # 15 boosters -> artifacts/ckpt/sbr_v2_feat
$PY scripts/verify_orientation_guard.py            # guard is a bit-exact no-op on RAS scans

bash scripts/pack.sh 012
