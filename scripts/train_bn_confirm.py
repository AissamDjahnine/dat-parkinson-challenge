"""Stage-2 confirmation of an 011 blur+noise arm: same folds, independent training draw.

    python scripts/train_bn_confirm.py --bn-m 2.5 --bn-p 0.9 --tag s2b_m25

**The training loop is imported, not copied.** `train_fold` comes straight from `train_bn.py`, so
there is exactly one implementation of the thing being tested and it cannot drift from the one
draw 1 used. Only `main()` differs: it keeps the frozen fold columns and re-rolls the training
randomness through `--train-seed-offset`.

── WHAT THIS SCRIPT IS FOR ─────────────────────────────────────────────────────────────────
Stage 1 screens several arms and reports the best. The best of k noisy measurements is biased
upward, so the winner must be re-measured independently before it is believed.

── ⚠ THE ORIGINAL DESIGN WAS WRONG, AND THE REASON IS WORTH KEEPING ────────────────────────
This script first re-tested the winner on THREE FRESH FOLD PARTITIONS (`folds_ext.csv`, seeds
3/4/5), on the theory that re-drawing the partitions re-rolls all of the winner's luck. Two
measurements killed that:

  1. **The partitions were not fresh.** Compared after trying every relabelling of the 5 fold
     ids, `fold_s4` agreed with `fold_s1` on **98.8 %** of scans — as similar as s0 and s1 are to
     each other. Sweeping 37 seeds, the LOWEST achievable disagreement with the frozen three is
     ~9 %, and any triple of new seeds is 98.6-99.3 % self-similar. With 13 acquisition groups of
     which 4 hold 72 % of scans, the space of distinct grouped partitions is nearly exhausted.
     "Three independent partitions" was never available, so the "all 3 seeds agree" clause was
     worth roughly ONE effective draw, not three.
  2. **Partition luck largely cancels anyway.** Every stage-1 arm is scored on the SAME three
     partitions, so partition difficulty is common to arm and control and subtracts out of the
     paired delta. What differs between arms is their training randomness — model init and the
     augmentation draws. That is the noise source that produced the winner, and it is what has to
     be re-rolled.

So stage 2 now holds the partitions FIXED and re-rolls the training randomness via
`--train-seed-offset`. Two consequences, both improvements: stage-2 numbers are directly
comparable to stage-1 numbers (identical validation scans), and the stage-2 control measured
against the stage-1 control is a direct measurement of the training noise, which calibrates the
threshold empirically instead of by assumption.

`scripts/make_folds_ext.py` is retained, unused, with this finding recorded in it.

── THE RULE ────────────────────────────────────────────────────────────────────────────────
Fixed in `run_aug_stage2.sh` before any number exists. A threshold chosen after seeing the
result is not a confirmation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datpark.cv import N_SPLITS  # noqa: E402
from datpark.metrics import report, score  # noqa: E402
from train_bn import train_fold  # noqa: E402  ★ the SAME loop stage 1 measured
from train_cnn import load_data  # noqa: E402
from train_cnn2 import make_grid_fn  # noqa: E402

FOLDS_EXT = ROOT / "artifacts" / "folds_ext.csv"


def build_parser() -> argparse.ArgumentParser:
    """Mirrors train_bn.py's parser, plus the fold and seed-offset controls."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bn-m", type=float, default=0.0)
    ap.add_argument("--bn-p", type=float, default=0.9)
    ap.add_argument("--bn-no-blur", action="store_true")
    ap.add_argument("--bn-no-noise", action="store_true")
    # ★ BUG FIX 2026-08-04 (018a Stage 2). `train_bn.train_fold` — which this script calls — reads
    # `args.bn_blur_2d` unconditionally at train_bn.py:94, inside an expression that is evaluated even
    # when blur is disabled. That flag was added to train_bn.py LATER, for an earlier bn2d experiment, and
    # this parser was never updated. So ANY invocation with --bn-m > 0 crashed with
    # `AttributeError: 'Namespace' object has no attribute 'bn_blur_2d'`, which means the SHIPPED 011
    # member `s2b_n25` could no longer be reproduced from source at all.
    # This is purely additive and provably behaviour-preserving: `store_true` defaults to False, which
    # is the value train_bn.py's own default supplies, so every previously-working invocation is
    # byte-identical. Same help text as train_bn.py:131 deliberately — one flag, one meaning.
    ap.add_argument("--bn-blur-2d", action="store_true",
                    help="blur the AXIAL PLANE ONLY, leaving superior-inferior untouched")
    ap.add_argument("--voxel-mm", type=float, default=1.3333)
    ap.add_argument("--focal-gamma", type=float, default=1.0)
    ap.add_argument("--focal-raw", action="store_true")
    ap.add_argument("--arch", default="datnet")
    ap.add_argument("--aug", default="orig", choices=["fixed", "orig"])
    ap.add_argument("--rot-plane", default="axial", choices=["axial", "sagittal"])
    ap.add_argument("--cubes", default="cubes_hires.npy")
    ap.add_argument("--rot", type=float, default=15.0)
    ap.add_argument("--trans", type=float, default=0.090)
    ap.add_argument("--scale", type=float, default=0.075)
    ap.add_argument("--intensity", type=float, default=0.1)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--stem-stride", type=int, default=3)
    ap.add_argument("--widths", default=None)
    ap.add_argument("--loss", default="bce")
    ap.add_argument("--loss-q", type=float, default=0.7)
    ap.add_argument("--loss-beta", type=float, default=0.5)
    ap.add_argument("--loss-drop", type=float, default=0.05)
    ap.add_argument("--fold-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="WHICH fold columns of folds.csv to use. Defaults to the frozen three, "
                         "so stage-2 numbers are directly comparable to stage-1 numbers.")
    ap.add_argument("--train-seed-offset", type=int, default=1000,
                    help="added to the fold seed before deriving the torch seed, so the model "
                         "init and augmentation draws are an INDEPENDENT draw while the "
                         "validation partitions stay identical")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="deprecated alias for --fold-seeds")
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--save-dir", default=None, help="see train_aug2.py; always set it")
    return ap


def load_ext_folds(folds_frozen: pd.DataFrame) -> pd.DataFrame:
    """folds_ext.csv, verified to be row-aligned with the cache ordering load_data used.

    load_data returns X reordered to folds.csv's uid order. folds_ext.csv was generated from
    folds.csv preserving row order, so the same X lines up — but a silent misalignment here would
    scramble labels invisibly and produce a plausible wrong number, so it is asserted, not
    assumed.
    """
    if not FOLDS_EXT.exists():
        raise SystemExit(
            f"{FOLDS_EXT} not found. Run:  python scripts/make_folds_ext.py"
        )
    ext = pd.read_csv(FOLDS_EXT)
    assert len(ext) == len(folds_frozen), (
        f"folds_ext.csv has {len(ext)} rows, folds.csv has {len(folds_frozen)}"
    )
    assert (ext.uid.to_numpy() == folds_frozen.uid.to_numpy()).all(), (
        "folds_ext.csv is not row-aligned with folds.csv, so it does not line up with the cube "
        "order load_data returned. Regenerate it with scripts/make_folds_ext.py."
    )
    assert (ext.y.to_numpy() == folds_frozen.y.to_numpy()).all(), "labels disagree"
    return ext


def main() -> None:
    args = build_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds_frozen, X, y = load_data(args.cubes)
    # ★ the FROZEN partitions, deliberately. See the docstring: re-drawing partitions was the
    # original plan and it does not work on this data, and partition luck cancels in a paired
    # arm-minus-control comparison anyway. What gets re-rolled here is the training randomness.
    folds = folds_frozen
    grid_fn = make_grid_fn(args)
    yv = y.numpy().astype(np.float64)

    if args.seeds is not None:
        args.fold_seeds = args.seeds
    missing = [s for s in args.fold_seeds if f"fold_s{s}" not in folds.columns]
    if missing:
        raise SystemExit(f"no fold column(s) for seed(s) {missing}")

    print(f"STAGE 2  tag={args.tag}  bn_m={args.bn_m} bn_p={args.bn_p}  "
          f"fold_seeds={args.fold_seeds}  train_seed_offset={args.train_seed_offset}  "
          f"folds=folds.csv (frozen)", flush=True)
    if args.bn_m == 0:
        print("  (all flags zero: this is the stage-2 CONTROL — the only legitimate baseline "
              "for the arms below, since these partitions are new)", flush=True)

    oof = np.full((len(folds), len(args.fold_seeds)), np.nan, dtype=np.float64)
    t0 = time.time()
    for si, seed in enumerate(args.fold_seeds):
        col = f"fold_s{seed}"
        for k in range(N_SPLITS):
            va = (folds[col] == k).to_numpy()
            # The FOLD comes from `seed`; the TRAINING randomness comes from
            # `seed + offset`. Stage 1 used offset 0, so this is an independent draw of model
            # init and augmentation draws over the IDENTICAL validation partitions.
            tseed = (seed + args.train_seed_offset) * 100 + k
            p, model = train_fold(X[~va], y[~va], X[va], args, device, tseed, grid_fn)
            oof[va, si] = p
            if args.save_dir:
                sd = Path(args.save_dir)
                sd.mkdir(parents=True, exist_ok=True)
                torch.save({k2: v.cpu() for k2, v in model.state_dict().items()},
                           sd / f"{args.arch}_s{seed}_f{k}.pt")
            print(f"  seed {seed} fold {k}: n_va={len(p):4d}  "
                  f"logloss {score(yv[va], p)['log_loss']:.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)

    assert not np.isnan(oof).any(), (
        "some scans received no prediction — a fold column does not partition all rows. "
        "train_cnn2 guards this the same way; an unfilled row would silently score as ~13.8 "
        "log loss and produce a catastrophically wrong arm that still looks like a number."
    )
    mean = oof.mean(axis=1)
    out = pd.DataFrame({"uid": folds.uid, "y": folds.y, "oof_mean": mean})
    for si, seed in enumerate(args.fold_seeds):
        out[f"oof_s{seed}"] = oof[:, si]
    dest = ROOT / "artifacts" / "oof" / f"{args.tag}.csv"
    tmp = dest.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, dest)          # atomic: a truncated OOF must never look complete

    print()
    for si, seed in enumerate(args.fold_seeds):
        print(f"  seed {seed}: {score(yv, oof[:, si])['log_loss']:.4f}")
    print(report(folds.y.to_numpy(), mean))
    print("  ⚠ FOR THIS ARM THE QUANTITY THAT MATTERS IS THE BLEND, NOT THIS NUMBER.")
    print("     bn_m25 was WORSE standalone (0.2754 vs 0.2694) yet improved the 008a blend by")
    print("     +0.0021 with all three seeds agreeing, because it is more decorrelated from the")
    print("     pool (error Jaccard 0.669 -> 0.525). Score it on the shipped TTA basis with scripts/eval_tta_n25.py.")
    print("  These ARE comparable to stage-1 numbers: identical partitions, independent training\n"
          "  randomness. Compare arm-vs-control within stage 2, and also stage-2-control against\n"
          "  stage-1-control, which measures the training noise directly.")


if __name__ == "__main__":
    main()
