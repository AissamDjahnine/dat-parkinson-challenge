"""Train the SBR + LightGBM track and export submission assets.

Deliberately pickle-free. LightGBM's text format is version-independent, and the
calibrator is two floats in a JSON file — so nothing here can break because the
container ships a different scikit-learn than we trained with.

One booster per (seed, fold); inference averages all of them. The calibrator is
fitted on out-of-fold training predictions only, which keeps the whole pipeline
deterministic with respect to the test set.

    python scripts/train_sbr.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datpark.cv import N_SPLITS, SEEDS  # noqa: E402
from datpark.features import FEATURE_NAMES, feature_matrix  # noqa: E402
from datpark.metrics import report, score  # noqa: E402
from datpark.preprocess import PreprocessConfig  # noqa: E402

CACHE = ROOT / "artifacts" / "cache"
ASSETS = ROOT / "artifacts" / "sbr"   # staged into inference/assets/models by scripts/pack.sh

PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=25,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=400,
    verbose=-1,
)


def platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit p = sigmoid(a * logit + b) by simple Newton/IRLS-free optimisation."""
    from scipy.optimize import minimize

    def nll(theta):
        a, b = theta
        z = np.clip(a * logits + b, -30, 30)
        return float(np.mean(np.log1p(np.exp(z)) - y * z))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 5000})
    return float(res.x[0]), float(res.x[1])


def main() -> None:
    folds = pd.read_csv(ROOT / "artifacts" / "folds.csv")
    order = pd.read_csv(CACHE / "meta_v1.csv").uid.tolist()
    cubes = np.load(CACHE / "cubes_v1.npy", mmap_mode="r")
    pos = {u: i for i, u in enumerate(order)}

    idx = [pos[u] for u in folds.uid]
    print(f"computing {len(FEATURE_NAMES)} features for {len(idx)} scans...")
    X = feature_matrix(np.asarray(cubes[idx], dtype=np.float32))
    y = folds.y.to_numpy()

    ASSETS.mkdir(parents=True, exist_ok=True)
    models_dir = ASSETS / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir()

    oof_per_seed = []
    n_models = 0
    for seed in SEEDS:
        oof = np.full(len(y), np.nan)
        fold = folds[f"fold_s{seed}"].to_numpy()
        for k in range(N_SPLITS):
            va = fold == k
            clf = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            clf.fit(X[~va], y[~va], feature_name=list(FEATURE_NAMES))
            oof[va] = clf.predict_proba(X[va])[:, 1]
            clf.booster_.save_model(str(models_dir / f"lgb_s{seed}_f{k}.txt"))
            n_models += 1
        assert not np.isnan(oof).any()
        oof_per_seed.append(oof)
        print(f"  seed {seed}: {score(y, oof)['log_loss']:.4f} log loss")

    oof_mean = np.mean(oof_per_seed, axis=0)
    raw = score(y, oof_mean)

    logits = np.log(np.clip(oof_mean, 1e-6, 1 - 1e-6) / (1 - np.clip(oof_mean, 1e-6, 1 - 1e-6)))
    a, b = platt(logits, y.astype(float))
    cal = 1 / (1 + np.exp(-(a * logits + b)))
    calibrated = score(y, cal)

    print(f"\nuncalibrated OOF : log loss {raw['log_loss']:.4f}  AUROC {raw['auroc']:.4f}  ECE {raw['ece']:.4f}")
    print(f"calibrated   OOF : log loss {calibrated['log_loss']:.4f}  AUROC {calibrated['auroc']:.4f}  ECE {calibrated['ece']:.4f}")
    print(f"Platt: a={a:.4f} b={b:.4f}   (gain {raw['log_loss'] - calibrated['log_loss']:+.4f})")
    print("\nNOTE: this Platt fit is evaluated on the same OOF it was fitted on, so the")
    print("      number above is mildly optimistic. It is never applied at inference:")
    print("      every shipped config sets calibration_applied = false.")

    report(y, cal, folds.group.to_numpy(), title="calibrated OOF by acquisition group")

    cfg = PreprocessConfig()
    (ASSETS / "config.json").write_text(json.dumps({
        "preprocess": cfg.to_dict(),
        "feature_names": list(FEATURE_NAMES),
        "calibration": {"kind": "platt", "a": a, "b": b},
        "n_models": n_models,
        "train_prevalence": float(y.mean()),
        "clip": [1e-4, 1 - 1e-4],
        "oof": {"log_loss": calibrated["log_loss"], "auroc": calibrated["auroc"]},
    }, indent=2))

    size = sum(p.stat().st_size for p in ASSETS.rglob("*")) / 1e6
    print(f"\nexported {n_models} boosters + config to {ASSETS}  ({size:.2f} MB)")

    pd.DataFrame({"uid": folds.uid, "y": y, "oof_raw": oof_mean, "oof_cal": cal}).to_csv(
        ROOT / "artifacts" / "oof" / "sbr_lgb.csv", index=False)


if __name__ == "__main__":
    main()
