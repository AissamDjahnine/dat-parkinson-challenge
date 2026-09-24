"""Cross-validation scheme.

Prevalence is confounded with acquisition (EDA.md: chi2 p=2.3e-5 on slice
thickness, p=1.4e-7 on the combined key), and a metadata-only model scores
AUROC 0.726 under random k-fold versus 0.533 grouped. Random folds are therefore
not an option -- they would reward centre recognition.

The grouping key is `slice thickness x intensity population`, which resolves the
2.46 mm group into what are almost certainly two distinct centres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# p99.5 counts below this put a scan in the "low-count" reconstruction family.
# The cohort distribution is strongly bimodal with a wide empty gap, so the exact
# threshold is not delicate.
INTENSITY_SPLIT = 300.0

N_SPLITS = 5
SEEDS = (0, 1, 2)


def acquisition_group(slice_mm: float, p995: float) -> str:
    """Centre proxy for one scan. Computable from a single scan in isolation."""
    return f"{round(float(slice_mm), 2)}/{'lo' if p995 < INTENSITY_SPLIT else 'hi'}"


def add_group(df: pd.DataFrame, slice_col="slice_mm", p995_col="p995") -> pd.DataFrame:
    out = df.copy()
    out["group"] = [
        acquisition_group(s, p) for s, p in zip(out[slice_col], out[p995_col])
    ]
    return out


def partition_groups(
    sizes: dict[str, int],
    positives: dict[str, int],
    n_splits: int,
    rng: np.random.Generator,
    alpha: float = 2.0,
) -> dict[str, int]:
    """Randomly partition whole groups into folds, balancing size and prevalence.

    sklearn's StratifiedGroupKFold is effectively deterministic here: with only
    13 indivisible groups its greedy assignment returns the same partition for
    every seed, so repeated CV would report a variance of exactly zero. This
    shuffles the insertion order and greedily places each group in the fold that
    keeps both fold size and fold prevalence closest to target, which gives
    genuinely different partitions per seed.
    """
    total = sum(sizes.values())
    target = total / n_splits
    global_prev = sum(positives.values()) / max(total, 1)

    fold_n = np.zeros(n_splits)
    fold_pos = np.zeros(n_splits)
    assign: dict[str, int] = {}

    # largest groups first (they constrain the packing most), order shuffled
    # within equal sizes so the partition still varies with the seed
    names = list(sizes)
    rng.shuffle(names)
    names.sort(key=lambda g: -sizes[g])

    for g in names:
        n_g, p_g = sizes[g], positives[g]
        cost = np.empty(n_splits)
        for f in range(n_splits):
            new_prev = (fold_pos[f] + p_g) / (fold_n[f] + n_g)
            # rank on the fold's CURRENT load, not on |new_size - target|:
            # the latter makes a half-full fold cheaper than an empty one and
            # can starve the last fold entirely
            cost[f] = fold_n[f] / target + alpha * abs(new_prev - global_prev)
        best = int(np.argmin(cost + rng.normal(0, 0.05, n_splits)))
        assign[g] = best
        fold_n[best] += n_g
        fold_pos[best] += p_g
    return assign


def make_folds(
    df: pd.DataFrame,
    label_col: str = "y",
    group_col: str = "group",
    n_splits: int = N_SPLITS,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """Assign every scan a fold index per seed.

    Written to disk once so that every model track trains on identical folds --
    without that, out-of-fold predictions cannot be honestly ensembled.
    """
    out = df[["uid", label_col, group_col]].copy()
    agg = out.groupby(group_col)[label_col].agg(["size", "sum"])
    sizes = agg["size"].to_dict()
    positives = agg["sum"].to_dict()

    for seed in seeds:
        rng = np.random.default_rng(seed)
        assign = partition_groups(sizes, positives, n_splits, rng)
        col = out[group_col].map(assign).to_numpy()
        assert (col >= 0).all(), "every scan must land in exactly one validation fold"
        assert len(set(assign.values())) == n_splits, "a fold ended up empty"
        out[f"fold_s{seed}"] = col
    return out


def fold_iter(folds: pd.DataFrame, seed: int, n_splits: int = N_SPLITS):
    """Yield (train_idx, valid_idx) positional arrays for one seed."""
    col = folds[f"fold_s{seed}"].to_numpy()
    for k in range(n_splits):
        yield np.nonzero(col != k)[0], np.nonzero(col == k)[0]


def leave_one_group_out(df: pd.DataFrame, group_col="group", min_n: int = 100):
    """Yield (name, train_idx, valid_idx) holding out one whole centre.

    A harsher test than grouped k-fold: it simulates a private set drawn from a
    centre mix we never trained on.
    """
    g = df[group_col].to_numpy()
    for name, n in pd.Series(g).value_counts().items():
        if n < min_n:
            continue
        va = np.nonzero(g == name)[0]
        yield name, np.nonzero(g != name)[0], va
