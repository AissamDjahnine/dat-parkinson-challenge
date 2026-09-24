"""Scoring. Log loss ranks the competition; everything else is diagnosis.

AUROC and log loss are reported separately on purpose: AUROC measures the model,
log loss measures model AND calibration. Conflating them hides which one is
costing you (see STRATEGY.md -- one leaderboard entry is losing 0.041 purely to
miscalibration).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def clipped(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-width binned |confidence - accuracy|, weighted by bin population."""
    p = clipped(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def score(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = clipped(p)
    out = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_pred": float(p.mean()),
    }
    # what a perfectly recalibrated version of these same scores would achieve;
    # the difference is the calibration headroom
    out["log_loss_baseline"] = float(log_loss(y, np.full(len(y), y.mean()), labels=[0, 1]))
    return out


def report(y, p, groups=None, title: str = "") -> pd.DataFrame:
    overall = score(y, p)
    if title:
        print(f"\n=== {title} ===")
    print(
        f"log loss {overall['log_loss']:.4f}   AUROC {overall['auroc']:.4f}   "
        f"Brier {overall['brier']:.4f}   ECE {overall['ece']:.4f}   "
        f"(base-rate log loss {overall['log_loss_baseline']:.4f})"
    )
    rows = [{"group": "ALL", **overall}]
    if groups is not None:
        g = np.asarray(groups)
        print(f"\n{'group':<10}{'n':>5}{'prev':>7}{'log loss':>10}{'AUROC':>8}{'ECE':>7}")
        print("-" * 47)
        for name in pd.Series(g).value_counts().index:
            m = g == name
            if m.sum() < 5:
                continue
            s = score(np.asarray(y)[m], np.asarray(p)[m])
            rows.append({"group": name, **s})
            auc = "  n/a" if np.isnan(s["auroc"]) else f"{s['auroc']:.3f}"
            print(f"{name:<10}{s['n']:>5}{s['prevalence']:>7.2f}"
                  f"{s['log_loss']:>10.4f}{auc:>8}{s['ece']:>7.3f}")
    return pd.DataFrame(rows)


def summarise_seeds(per_seed: list[dict]) -> str:
    """mean ± sd across repeated CV seeds — never report a single run."""
    ll = np.array([d["log_loss"] for d in per_seed])
    au = np.array([d["auroc"] for d in per_seed])
    return (f"log loss {ll.mean():.4f} ± {ll.std():.4f}   "
            f"AUROC {au.mean():.4f} ± {au.std():.4f}   (n_seeds={len(per_seed)})")
