#!/usr/bin/env bash
# 008a - four 3D CNN families (80%) + SBR LightGBM (20%). ~2.5 GPU-hours on one 16 GB GPU.
# Grouped CV 0.2472 / AUROC 0.9607.  Run recipes/common.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-uv run python}
AUG="--aug orig --rot 15.0 --trans 0.090 --scale 0.075 --intensity 0.1 --smooth 0.0"
RUN="--cubes cubes_hires.npy --epochs 30 --seeds 0 1 2 --tta-flip"

$PY scripts/train_sbr.py                                                        # SBR head, CPU
$PY scripts/train_cnn2.py --arch datnet $AUG --stem-stride 3 $RUN \
    --tag ckpt_sm0 --save-dir artifacts/ckpt/sm0                                # sm0
$PY scripts/train_cnn2.py --arch siamese --aug fixed --rot-plane axial --stem-stride 3 $RUN \
    --tag ckpt_siamese --save-dir artifacts/ckpt/siamese                        # siamese
$PY scripts/train_slicenet.py $AUG $RUN \
    --tag ckpt_slice128 --save-dir artifacts/ckpt/slice128                      # slicenet
bash scripts/fetch_medicalnet.sh
$PY scripts/train_med3d.py --depth 10 --standardise --batch 16 $AUG $RUN \
    --tag ckpt_med3d --save-dir artifacts/ckpt/med3d                            # med3d
$PY scripts/eval_tta.py --family siamese --save                                 # score on shipped TTA
$PY scripts/eval_tta_sm0.py --save

bash scripts/pack.sh 008a
