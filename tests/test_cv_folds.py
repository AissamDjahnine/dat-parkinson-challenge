"""Grouped-CV integrity — the single most important modelling invariant.

Metadata alone scores AUROC 0.726 under random k-fold versus 0.533 grouped
(EDA.md), so a group leaking across the train/valid boundary turns CV into a
centre-recognition score. Every recorded result in the research log assumes
`artifacts/folds.csv` is leak-free and shared by all tracks; if that breaks,
every number in the log becomes incomparable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from datpark.cv import (
    INTENSITY_SPLIT,
    N_SPLITS,
    SEEDS,
    acquisition_group,
    add_group,
    fold_iter,
    make_folds,
)

ROOT = Path(__file__).resolve().parent.parent
FOLDS = ROOT / "artifacts" / "folds.csv"


def _synthetic(n=600, n_groups=12, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n)
    # deliberately confound prevalence with group, as the real data does
    rate = np.linspace(0.25, 0.8, n_groups)
    return pd.DataFrame({
        "uid": [f"u{i:04d}" for i in range(n)],
        "group": [f"c{k}" for k in g],
        "y": (rng.random(n) < rate[g]).astype(int),
    })


# ------------------------------------------------------------- the leak invariant
def test_no_group_spans_two_folds_synthetic():
    df = _synthetic()
    folds = make_folds(df)
    for seed in SEEDS:
        per_group = folds.groupby("group")[f"fold_s{seed}"].nunique()
        bad = per_group[per_group > 1]
        assert bad.empty, f"seed {seed}: groups split across folds: {list(bad.index)}"


def test_every_sample_in_exactly_one_validation_fold():
    df = _synthetic()
    folds = make_folds(df)
    for seed in SEEDS:
        col = folds[f"fold_s{seed}"].to_numpy()
        assert set(np.unique(col)) == set(range(N_SPLITS))
        counts = np.bincount(col, minlength=N_SPLITS)
        assert counts.sum() == len(df)
        assert (counts > 0).all(), "an empty fold makes the OOF incomplete"


def test_fold_iter_partitions_cleanly():
    df = _synthetic()
    folds = make_folds(df)
    for seed in SEEDS:
        seen = []
        for tr, va in fold_iter(folds, seed):
            assert len(np.intersect1d(tr, va)) == 0, "train/valid overlap"
            assert len(tr) + len(va) == len(df)
            seen.append(va)
        allva = np.concatenate(seen)
        assert len(allva) == len(df) and len(np.unique(allva)) == len(df)


def test_seeds_give_different_partitions():
    """Seed-averaging only helps if the partitions actually differ."""
    df = _synthetic()
    folds = make_folds(df)
    cols = [folds[f"fold_s{s}"].to_numpy() for s in SEEDS]
    assert not np.array_equal(cols[0], cols[1])


def test_make_folds_is_deterministic():
    df = _synthetic()
    a, b = make_folds(df), make_folds(df)
    for s in SEEDS:
        assert np.array_equal(a[f"fold_s{s}"], b[f"fold_s{s}"])


# ----------------------------------------------------------- the grouping key
def test_acquisition_group_is_single_scan_computable():
    """Rule 6: the group key must not need any other scan."""
    assert acquisition_group(2.46, 100.0) == "2.46/lo"
    assert acquisition_group(2.46, 1000.0) == "2.46/hi"
    assert acquisition_group(2.4600001, 50.0) == "2.46/lo"


def test_intensity_split_boundary():
    below = acquisition_group(3.0, INTENSITY_SPLIT - 1e-6)
    above = acquisition_group(3.0, INTENSITY_SPLIT + 1e-6)
    assert below.endswith("/lo") and above.endswith("/hi")


def test_add_group_does_not_mutate_input():
    df = pd.DataFrame({"slice_mm": [2.46, 3.9], "p995": [10.0, 900.0]})
    before = df.copy()
    out = add_group(df)
    assert "group" in out.columns
    pd.testing.assert_frame_equal(df, before)


# ------------------------------------------------- the frozen on-disk partition
def test_shipped_folds_csv_is_leak_free():
    """Guards the actual file every recorded result was produced against."""
    if not FOLDS.exists():
        from run_tests import SkipTest
        raise SkipTest("artifacts/folds.csv absent")
    folds = pd.read_csv(FOLDS)
    assert len(folds) == folds.uid.nunique(), "duplicate uids in folds.csv"
    for seed in SEEDS:
        col = f"fold_s{seed}"
        assert col in folds.columns
        per_group = folds.groupby("group")[col].nunique()
        bad = per_group[per_group > 1]
        assert bad.empty, f"{col}: groups leak across folds: {list(bad.index)}"
        assert set(folds[col].unique()) == set(range(N_SPLITS))


def test_shipped_folds_match_every_oof_file():
    """OOF files are only ensemblable if they share one uid order."""
    if not FOLDS.exists():
        from run_tests import SkipTest
        raise SkipTest("artifacts/folds.csv absent")
    folds = pd.read_csv(FOLDS)
    oof_dir = ROOT / "artifacts" / "oof"
    files = sorted(oof_dir.glob("*.csv")) if oof_dir.exists() else []
    if not files:
        from run_tests import SkipTest
        raise SkipTest("no OOF files yet")
    checked = 0
    for f in files:
        d = pd.read_csv(f)
        if len(d) != len(folds):
            continue  # partial/other-shape artefacts are not OOF over the full set
        assert (d.uid.values == folds.uid.values).all(), (
            f"{f.name} uid order differs from folds.csv -- blending it would "
            f"silently scramble labels")
        # ★ Label-noise experiment arms store the FLIPPED training labels in their own `y` column BY
        # DESIGN (the folds table keeps truth). They are
        # therefore legitimate exceptions to this assert, and the exception is narrow and named rather
        # than a pattern, so a genuinely scrambled arm still fails. An earlier experiment records what happens when
        # a scorer trusts these labels: two arm damages were reported ~3x too large.
        if f.name.startswith("ln4_") or f.name in ("ln_d47.csv",):
            n_diff = int((d.y.values != folds.y.values).sum())
            # ⚠ The 041 arms are INCONSISTENT about which y they save, and that inconsistency is the
            # point: `train_med3d.py` and the SBR path write the FOLDS TABLE's y (0 differences), while
            # `train_cnn2.py`/`train_slicenet.py` write the y TENSOR (47 differences). Verified 2026-08-06.
            # Hence the only safe rule, and the reason this exemption exists at all: TRUTH ALWAYS COMES
            # FROM folds.csv, never from an arm's own y column.
            assert n_diff in (0, 47, 75), (
                f"{f.name} is a label-noise arm but differs on {n_diff} labels, "
                f"which is neither 0 (saved truth) nor the frozen 47/75 (saved the flipped tensor)")
            continue
        assert (d.y.values == folds.y.values).all(), f"{f.name} labels differ"
        checked += 1
    assert checked > 0, "no full-length OOF file was actually verified"
