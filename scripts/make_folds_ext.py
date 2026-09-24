"""Write artifacts/folds_ext.csv, which scripts/train_bn_confirm.py (011, draw 2) requires.

It preserves folds.csv's row order and frozen fold_s0-2 columns and adds fold_s3-5; draw 2 uses
only the frozen three. It was first written to give the confirmation draw three FRESH fold
partitions. Measurement showed the partitions are not fresh and cannot be, so the draw holds the
partitions fixed and re-rolls the TRAINING randomness instead.

THE FINDING, worth keeping because it is a property of the dataset and not of this code:

    agreement with the frozen seeds 0/1/2, after trying every relabelling of the 5 fold ids
        fold_s3   90.9 %   91.4 %   88.3 %
        fold_s4   98.2 %   98.8 %   94.3 %   <- a near-duplicate of fold_s1
        fold_s5   88.3 %   88.9 %   91.1 %
    for scale, the frozen seeds agree with EACH OTHER at 93.2-98.8 %
    over 37 candidate seeds the lowest achievable worst-case agreement is ~90.6 %,
    and the best available triple (9, 30, 31) is 98.6-99.3 % self-similar

Cause: 4 of the 13 acquisition groups hold 72 % of the 1362 scans, so any grouped 5-fold split
must place those four blocks, and there are few distinct ways to do it. The space of distinct
grouped partitions is nearly exhausted by the three we already use.

Consequence for the original design: "three independent partitions" was never available, so the
"all 3 seeds agree" clause would have been worth roughly ONE effective draw, and a stage-1 winner
whose luck came from partition 1 would have had that luck handed back to it by seed 4.

The deeper reason re-drawing was unnecessary: every stage-1 arm is scored on the SAME partitions,
so partition difficulty is common to arm and control and subtracts out of the paired delta.

Original docstring follows.

Extend the frozen fold partition with three FRESH seeds, for stage-2 confirmation only.

    uv run python scripts/make_folds_ext.py      # writes artifacts/folds_ext.csv

★ ADDITIVE. `artifacts/folds.csv` is never read for writing and never modified. Every number
reported anywhere on this project is on folds.csv; this file exists solely so a stage-1 winner
can be re-tested on partitions it has never seen.

── WHY FRESH FOLDS, AND NOT JUST FRESH INIT SEEDS ──────────────────────────────────────────
Stage 1 sweeps 5 arms and calls anything past +0.010 a winner. With a paired-delta SE of about
0.0045, roughly 6.5 % of pure-noise sweeps hand back a "winner" anyway. That false positive is
produced by the interaction of one arm's randomness with THESE THREE fold partitions. Re-running
with a different torch seed on the same partitions would only re-roll half of that; it leaves the
partition luck in place. Fresh partitions re-roll all of it, which is what makes this a genuine
replication rather than a repeat.

The cost of that choice, stated plainly: **stage-2 absolute numbers are NOT comparable to stage-1
absolute numbers.** A different partition moves the whole scale — the same configuration scores
0.2694 on folds.csv and 0.2804 on the merged-group folds10.csv. So stage 2 must be read only as
an arm-minus-control DELTA on its own partitions, which is why the runner always re-measures the
control on the same fresh folds instead of reusing the stage-1 control number.

── THE SELF-CHECK THAT MATTERS ─────────────────────────────────────────────────────────────
Seeds 0-2 are regenerated here alongside 3-5 and asserted to be BIT-IDENTICAL to folds.csv. If
that assertion fails, the grouping or the splitter has drifted since folds.csv was frozen, and
every historical number on this project is suspect — a far bigger problem than this experiment.
It is cheap to check and it would be negligent not to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datpark.cv import N_SPLITS, make_folds  # noqa: E402

FROZEN = ROOT / "artifacts" / "folds.csv"
OUT = ROOT / "artifacts" / "folds_ext.csv"
ALL_SEEDS = (0, 1, 2, 3, 4, 5)
NEW_SEEDS = (3, 4, 5)


def main() -> None:
    frozen = pd.read_csv(FROZEN)
    print(f"  frozen partition: {len(frozen)} scans, {frozen.group.nunique()} acquisition groups")

    ext = make_folds(frozen[["uid", "y", "group"]].copy(), seeds=ALL_SEEDS)

    # ---- self-check: the regenerated old seeds must reproduce the frozen file exactly --------
    for s in (0, 1, 2):
        col = f"fold_s{s}"
        same = (ext[col].to_numpy() == frozen[col].to_numpy()).all()
        assert same, (
            f"fold_s{s} does NOT reproduce artifacts/folds.csv. The grouping or the splitter has "
            f"drifted since the partition was frozen; every number on this project is affected. "
            f"Do not proceed with stage 2 — investigate this first."
        )
    print("  self-check: seeds 0-2 reproduce folds.csv bit-identically")

    # ---- the new seeds must obey the same grouped constraint --------------------------------
    for s in NEW_SEEDS:
        per_group = ext.groupby("group")[f"fold_s{s}"].nunique()
        bad = per_group[per_group > 1]
        assert bad.empty, f"seed {s}: acquisition groups split across folds: {list(bad.index)}"
        sizes = ext[f"fold_s{s}"].value_counts().sort_index().tolist()
        prev = ext.groupby(f"fold_s{s}").y.mean().round(3).tolist()
        assert len(sizes) == N_SPLITS, f"seed {s}: got {len(sizes)} folds, expected {N_SPLITS}"
        print(f"  seed {s}: fold sizes {sizes}   prevalence {prev}")
    print("  verified: no acquisition group is split across folds, for any new seed")

    # ---- the new partitions must actually be new --------------------------------------------
    sigs = {s: tuple(ext[f"fold_s{s}"]) for s in ALL_SEEDS}
    distinct = len(set(sigs.values()))
    assert distinct == len(ALL_SEEDS), (
        f"only {distinct}/{len(ALL_SEEDS)} partitions are distinct — a repeated partition would "
        f"make stage 2 a repeat rather than a replication, understating the variance it exists "
        f"to measure."
    )
    print(f"  verified: all {distinct} partitions are distinct")

    ext.to_csv(OUT, index=False)
    print(f"\n  wrote {OUT}  ({len(ext)} rows, seeds {list(ALL_SEEDS)})")
    print("  NOTE: stage-2 scores are NOT comparable to stage-1 scores in absolute terms.")
    print("        Read arm-minus-control on these folds only.")


if __name__ == "__main__":
    main()
