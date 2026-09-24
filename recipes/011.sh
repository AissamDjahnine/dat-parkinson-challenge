#!/usr/bin/env bash
# 011 - 008a with sm0 retrained under an added count-noise augmentation, two draws pooled
# (30 checkpoints). ~1 extra GPU-hour. Grouped CV 0.2448 / AUROC 0.9613.
# Needs the siamese, slicenet, med3d and SBR outputs of recipes/008a.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-uv run python}
BASE="--cubes cubes_hires.npy --stem-stride 3 --aug orig --rot 15.0 --trans 0.090 --scale 0.075 \
--intensity 0.1 --smooth 0.0 --epochs 30 --tta-flip"
NOISE="--bn-m 2.5 --bn-p 0.9 --bn-no-blur"

$PY scripts/train_bn.py $BASE --seeds 0 1 2 $NOISE \
    --tag bn_n25 --save-dir artifacts/ckpt/bn_n25                               # draw 1
$PY scripts/make_folds_ext.py
$PY scripts/train_bn_confirm.py $BASE --fold-seeds 0 1 2 --train-seed-offset 1000 $NOISE \
    --tag s2b_n25 --save-dir artifacts/ckpt/s2b_n25                             # draw 2
$PY scripts/eval_tta_n25.py --save                                              # both, shipped TTA

bash scripts/pack.sh 011
