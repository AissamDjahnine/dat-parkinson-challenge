"""022 Stage 4 — train an SBR arm on the exact grouped folds. CPU only, no new image cache.

    python scripts/train_sbr_v2.py --arm v1_anchor   # ANCHOR: must reproduce sbr_lgb
    python scripts/train_sbr_v2.py --arm v2          # ★ PRIMARY
    python scripts/train_sbr_v2.py --arm v2_mid      # attribution: midline only
    python scripts/train_sbr_v2.py --arm v2_feat     # attribution: feature set only

012 ships the `v2_feat` arm. This script writes an OOF csv, 15 fold boosters to
`artifacts/ckpt/sbr_<arm>/` and diagnostics; `scripts/pack.sh 012` stages the boosters.

Identical LightGBM PARAMS, identical SEEDS and identical folds to `train_sbr.py`, so the ONLY
difference between an arm's OOF and the shipped one is the feature matrix. That is what makes the four
arms a clean 2x2 of {v1 plane, robust plane} x {v1 features, v2 features}:

    v1_anchor   v1 plane   v1 features   -> ANCHOR, must reproduce sbr_lgb
    v2_mid      robust     v1 features   -> isolates the MIDLINE
    v2_feat     v1 plane   v2 features   -> isolates the FEATURE SET
    v2          robust     v2 features   -> PRIMARY (both changes)

★ MEMORY SAFETY. Features are computed from `cubes_v1.npy` via mmap in chunks of 128 scans. Two
system-RAM OOMs on 2026-08-04 came from CPU-side work materialising a whole cache at once, and a GPU
trainer is usually running alongside this. `train_sbr_folds.py:61` materialises all 1362 as float32
(1.43 GB) in one go; this does not. Do not remove the chunking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datpark.features_v2 import (  # noqa: E402
    FEATURE_NAMES_V1,
    FEATURE_NAMES_V2,
    feature_matrix_v1_at_plane,
    feature_matrix_v2,
)
from datpark.metrics import score  # noqa: E402

CACHE = ROOT / "artifacts" / "cache"
OUT = ROOT / "artifacts" / "022"
SEEDS = (0, 1, 2)              # exactly train_sbr_folds.py
CHUNK = 128

# byte-for-byte the PARAMS block from train_sbr.py / train_sbr_folds.py. Frozen: the pre-registration forbids any
# hyperparameter search in this experiment.
PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=25,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    n_estimators=400, verbose=-1,
)

ARMS = {
    #  arm        -> (extractor,                 robust_midline, feature names)
    "v1_anchor":  (feature_matrix_v1_at_plane, False, FEATURE_NAMES_V1),
    "v2_mid":     (feature_matrix_v1_at_plane, True,  FEATURE_NAMES_V1),
    "v2_feat":    (feature_matrix_v2,          False, FEATURE_NAMES_V2),
    "v2":         (feature_matrix_v2,          True,  FEATURE_NAMES_V2),
}


def build_features(arm: str, uids: list[str]):
    extractor, robust, names = ARMS[arm]
    order = pd.read_csv(CACHE / "meta_v1.csv").uid.tolist()
    pos = {u: i for i, u in enumerate(order)}
    if set(order) != set(uids):
        raise SystemExit("folds uids do not match the cube cache")
    idx = [pos[u] for u in uids]
    cubes = np.load(CACHE / "cubes_v1.npy", mmap_mode="r")

    blocks, mls = [], []
    for s in range(0, len(idx), CHUNK):
        blk = np.asarray(cubes[idx[s:s + CHUNK]], dtype=np.float32)
        F, m = extractor(blk, 2.0, robust)
        blocks.append(F)
        mls.extend(m)
        del blk
        print(f"    features {min(s + CHUNK, len(idx))}/{len(idx)}", end="\r", flush=True)
    print(" " * 44, end="\r")
    X = np.concatenate(blocks, axis=0)
    if X.shape[1] != len(names):
        raise SystemExit(f"{arm}: {X.shape[1]} features vs manifest {len(names)}")
    return X, mls, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--folds", default="folds.csv")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"sbr_{args.arm}"
    folds = pd.read_csv(ROOT / "artifacts" / args.folds)
    y = folds.y.to_numpy()

    extractor, robust, names = ARMS[args.arm]
    print(f"{tag}: plane={'ROBUST' if robust else 'v1'}  features={len(names)}  "
          f"folds={args.folds}  seeds={list(SEEDS)}")
    X, mls, names = build_features(args.arm, folds.uid.tolist())
    if not np.isfinite(X).all():
        raise SystemExit(f"{tag}: non-finite features")

    rel = np.array([m.reliable for m in mls])
    shifts = np.array([m.shift_mm for m in mls], dtype=np.float64)
    print(f"  midline reliable on {rel.sum()}/{len(rel)} scans ({rel.mean():.1%})")
    if rel.any():
        a = np.abs(shifts[rel])
        print(f"  |shift| on reliable scans: mean {a.mean():.2f} mm  median {np.median(a):.2f}  "
              f"p90 {np.percentile(a, 90):.2f}  max {a.max():.2f}")

    cols = {}
    ckdir = ROOT / "artifacts" / "ckpt" / tag
    ckdir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        oof = np.full(len(y), np.nan)
        fold = folds[f"fold_s{seed}"].to_numpy()
        for k in range(args.n_splits):
            va = fold == k
            clf = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            clf.fit(X[~va], y[~va], feature_name=list(names))
            oof[va] = clf.predict_proba(X[va])[:, 1]
            clf.booster_.save_model(str(ckdir / f"lgb_s{seed}_f{k}.txt"))
        assert not np.isnan(oof).any(), "a fold column does not partition all rows"
        cols[f"oof_s{seed}"] = oof
        print(f"  seed {seed}: {score(y, oof)['log_loss']:.4f}")

    mean = np.mean(list(cols.values()), axis=0)
    s = score(y, mean)
    print(f"  seed-averaged: {s['log_loss']:.4f} log loss  {s['auroc']:.4f} AUROC")

    out = ROOT / "artifacts" / "oof" / f"{tag}.csv"
    # `oof_raw` AND `oof_mean` so the blend code, which reads either, needs no special case
    pd.DataFrame({"uid": folds.uid, "y": y, **cols,
                  "oof_raw": mean, "oof_mean": mean}).to_csv(out, index=False)
    print(f"  OOF -> {out}")

    # diagnostics stay LOCAL: uid-level rows must never enter a submission archive or public report
    pd.DataFrame({"uid": folds.uid, "group": folds.group,
                  "shift_mm": shifts, "abs_shift_mm": np.abs(shifts),
                  "separation_mm": [m.separation_mm for m in mls],
                  "reliable": rel, "reason": [m.reason for m in mls]}
                 ).to_csv(OUT / f"midline_diagnostics_{args.arm}.csv", index=False)
    (OUT / f"feature_manifest_{args.arm}.json").write_text(json.dumps(
        dict(arm=args.arm, robust_midline=bool(robust), n_features=len(names),
             feature_names=list(names), lgb_params=PARAMS, seeds=list(SEEDS),
             folds=args.folds, n_splits=args.n_splits,
             standalone={"log_loss": s["log_loss"], "auroc": s["auroc"]}), indent=2))
    print(f"  diagnostics + manifest -> {OUT}")


if __name__ == "__main__":
    main()
