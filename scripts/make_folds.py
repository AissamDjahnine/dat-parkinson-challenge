"""Freeze the cross-validation folds once, so every model track shares them.

Writes artifacts/folds.csv. Out-of-fold predictions from different tracks can
only be honestly compared or ensembled if they were produced on identical splits.

    python scripts/make_folds.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datpark.cv import N_SPLITS, SEEDS, add_group, make_folds  # noqa: E402

OUT = ROOT / "artifacts" / "folds.csv"


def main() -> None:
    inv = pd.read_csv(ROOT / "artifacts" / "header_inventory.csv")
    lab = pd.read_csv(ROOT / "data" / "train_labels.csv")
    aud = pd.read_csv(ROOT / "artifacts" / "artifact_audit.csv")[["uid", "p995"]]

    df = inv.merge(lab, on="uid").merge(aud, on="uid")
    df["slice_mm"] = df.zooms.map(lambda z: round(ast.literal_eval(z)[2], 2))
    df["y"] = (df.is_pathologic == 1.0).astype(int)
    df = add_group(df)

    folds = make_folds(df)
    folds.to_csv(OUT, index=False)

    print(f"{len(folds)} scans, {folds.group.nunique()} acquisition groups, "
          f"{N_SPLITS} folds x {len(SEEDS)} seeds -> {OUT}\n")

    for seed in SEEDS:
        col = f"fold_s{seed}"
        t = folds.groupby(col).agg(n=("uid", "size"), prevalence=("y", "mean"),
                                   groups=("group", "nunique"))
        spread = t.prevalence.max() - t.prevalence.min()
        print(f"seed {seed}:  fold sizes {list(t.n)}   "
              f"prevalence {t.prevalence.min():.3f}-{t.prevalence.max():.3f} "
              f"(spread {spread:.3f})")

    # a group must never be split across folds, or the scheme is pointless
    for seed in SEEDS:
        per_group = folds.groupby("group")[f"fold_s{seed}"].nunique()
        bad = per_group[per_group > 1]
        assert bad.empty, f"seed {seed}: groups split across folds: {list(bad.index)}"
    print("\nverified: no acquisition group is split across folds, for any seed")

    sigs = {s: tuple(folds[f"fold_s{s}"]) for s in SEEDS}
    distinct = len(set(sigs.values()))
    print(f"verified: {distinct}/{len(SEEDS)} seeds give distinct partitions"
          + ("" if distinct == len(SEEDS) else "   <-- WARNING: repeated CV will understate variance"))

    print(f"\n{'group':<10}{'n':>6}{'prev':>7}   fold per seed")
    print("-" * 46)
    for name, sub in folds.groupby("group"):
        assign = "  ".join(str(sub[f'fold_s{s}'].iloc[0]) for s in SEEDS)
        print(f"{name:<10}{len(sub):>6}{sub.y.mean():>7.2f}   {assign}")


if __name__ == "__main__":
    main()
