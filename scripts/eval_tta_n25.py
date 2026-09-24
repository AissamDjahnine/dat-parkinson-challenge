"""Score the 011 noise-augmented checkpoints on the TTA basis 008a actually ships.

    python scripts/eval_tta_n25.py --save

Inference only. No training, no weights touched. Additive: `eval_tta.py` and `eval_tta_sm0.py`
are imported unmodified and their FAMILIES dict is extended in memory.

── THE GAP THIS CLOSES ─────────────────────────────────────────────────────────────────────
011's arms were trained by `train_bn.py`, which scores with `--tta-flip`, i.e. the 2-view **rl**
mirror. The shipped sm0 slot in `assets/config.json` declares **si_rl**. So every 011 number in
an earlier experiment — including the headline +0.0026 — is on the `rl` basis, while the member it would
replace is on `si_rl`. The comparison is sound (each arm was compared to its OWN control on the
same basis) but the ARCHIVE cannot be built until the winning checkpoints are scored on si_rl,
because that is what `main.py` will actually run at inference time.

── ★ WHICH SCHEME WE ADOPT IS FIXED BEFORE ANY NUMBER EXISTS ────────────────────────────────
    PRIMARY = si_rl, because that is what the shipped config declares.

All eight schemes are scored because it costs nothing extra, but they are DIAGNOSTICS. Adopting
the best of eight is selection over eight, and an earlier analysis measures selection as a first-order effect
here: best-of-5 under a true zero averages +0.0047. `eval_tta_sm0.py` already set the precedent
of refusing sub-0.002 wins for exactly this reason, and it found si_rl was already best.

── ★ BOTH DRAWS, POOLED — NOT THE BETTER ONE ───────────────────────────────────────────────
`bn_n25` (offset 0) scored 0.2437 in the blend and `s2b_n25` (offset 1000) scored 0.2458. Picking
the first because it is better is picking the luckier of two coin flips; an earlier analysis measured that
pooling returns their average, 0.2447, which is the honest expectation for either. So the archive
ships **30 checkpoints** and this script reports the pooled number as primary.

── HARNESS ANCHORS, ASSERTED NOT ASSUMED ───────────────────────────────────────────────────
Scheme `rl` must reproduce the trainers' own standalone numbers, because that is the basis they
used:

    bn_n25   rl -> 0.2605      s2b_n25  rl -> 0.2682

If either disagrees by more than 0.0005 the checkpoint directory, the cube grid or the
architecture is not what we think it is, and every downstream number is void.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_tta  # noqa: E402  imported unmodified
from datpark.metrics import score  # noqa: E402
from datpark.nets import DatNet  # noqa: E402

# same architecture, same grid, same checkpoint naming as the shipped sm0 family — verified on
# disk: artifacts/ckpt/{bn_n25,s2b_n25}/datnet_s{seed}_f{fold}.pt
BUILDER = lambda: DatNet(stem_stride=3)  # noqa: E731
for fam in ("bn_n25", "s2b_n25"):
    eval_tta.FAMILIES[fam] = (fam, "cubes_hires.npy", BUILDER, "datnet")

PRIMARY = "si_rl"                    # what assets/config.json declares for the sm0 slot
ANCHOR_RL = {"bn_n25": 0.2605, "s2b_n25": 0.2682}
SEL_BAR = 0.002                      # below this, a best-of-8 win is selection, not signal


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sg(z):
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    folds = pd.read_csv(ROOT / "artifacts" / "folds.csv")
    y = folds.y.to_numpy()
    oofdir = ROOT / "artifacts" / "oof"

    def get(tag, col="oof_mean"):
        f = oofdir / f"{tag}.csv"
        if not f.exists():
            return None
        return pd.read_csv(f).set_index("uid").reindex(folds.uid)[col].to_numpy()

    # ---- 1. score every scheme on both draws -------------------------------------------------
    # captured ONCE: rebuilding argv from a mutated argv inside the loop is how a flag silently
    # goes missing on the second family, and --save going missing would leave nothing to read.
    passthrough = sys.argv[1:]
    for fam in ("bn_n25", "s2b_n25"):
        print(f"\n{'=' * 86}\n=== {fam}\n{'=' * 86}", flush=True)
        sys.argv = ["eval_tta.py", "--family", fam] + passthrough
        eval_tta.main()

    # ---- 2. harness anchors ------------------------------------------------------------------
    print(f"\n{'=' * 86}\n=== HARNESS ANCHORS — scheme rl must reproduce the trainers' numbers")
    bad = []
    for fam, want in ANCHOR_RL.items():
        p = get(f"tta_{fam}_rl")
        if p is None:
            print(f"  {fam:<10} rl  MISSING — did you pass --save?"); bad.append(fam); continue
        got = score(y, p)["log_loss"]
        ok = abs(got - want) < 5e-4
        print(f"  {fam:<10} rl  {got:.4f}  expected {want:.4f}  |d|={abs(got-want):.4f}  "
              f"{'OK' if ok else '!!! MISMATCH — everything below is VOID'}")
        if not ok:
            bad.append(fam)
    if bad:
        print(f"\n  ABORT: {bad} failed the anchor. Do not build an archive from these.")
        sys.exit(1)

    # ---- 3. the shipped blend, sm0 slot replaced --------------------------------------------
    sia, sl, md = (get("tta_siamese_si_rot_scale"), get("ckpt_slice128"), get("ckpt_med3d"))
    sbr = get("sbr_lgb", "oof_raw")
    if any(v is None for v in (sia, sl, md, sbr)):
        print("\n  missing a shipped member's OOF — cannot evaluate the blend."); return

    def blend(sm0_p):
        return sg(0.8 * (lg(sm0_p) + lg(sia) + lg(sl) + lg(md)) / 4 + 0.2 * lg(sbr))

    shipped = get("tta_sm0_si_rl")
    base = score(y, blend(shipped))["log_loss"]
    print(f"\n{'=' * 86}\n=== 008a BLEND with the sm0 slot replaced   (shipped reference {base:.4f})")
    print(f"\n  {'scheme':<16}{'draw1':>9}{'draw2':>9}{'POOLED 30ckpt':>16}{'d vs 008a':>11}")
    rows = {}
    for s in eval_tta.SCHEMES:
        a, b = get(f"tta_bn_n25_{s}"), get(f"tta_s2b_n25_{s}")
        if a is None or b is None:
            continue
        ba, bb = score(y, blend(a))["log_loss"], score(y, blend(b))["log_loss"]
        bp = score(y, blend((a + b) / 2))["log_loss"]
        rows[s] = bp
        mark = "  <- PRIMARY" if s == PRIMARY else ""
        print(f"  {s:<16}{ba:>9.4f}{bb:>9.4f}{bp:>16.4f}{base - bp:>+11.4f}{mark}")

    if PRIMARY not in rows:
        print(f"\n  {PRIMARY} not scored — cannot conclude."); return
    d = base - rows[PRIMARY]
    print(f"\n  PRIMARY {PRIMARY}: pooled 30-checkpoint blend {rows[PRIMARY]:.4f}, "
          f"delta vs 008a {d:+.4f}")
    print(f"  an earlier experiment measured {'+0.0026':>7} on the rl basis. A si_rl number close to that is a "
          f"consistency check,\n  not a second piece of evidence — same checkpoints, same folds.")

    best = min(rows, key=rows.get)
    if best != PRIMARY:
        gain = rows[PRIMARY] - rows[best]
        if gain > SEL_BAR:
            print(f"\n  ★ '{best}' beats the shipped '{PRIMARY}' by {gain:+.4f}, above the "
                  f"{SEL_BAR} selection bar.\n    Worth a config change — but confirm it on the "
                  f"OTHER draw before adopting.")
        else:
            print(f"\n  '{best}' leads by only {gain:+.4f} over '{PRIMARY}' across 8 schemes — "
                  f"below the {SEL_BAR}\n    selection bar. KEEP {PRIMARY}. This is selection bias, and "
                  f"eval_tta_sm0.py reached the same verdict.")
    else:
        print(f"\n  '{PRIMARY}' is already best of the eight. Nothing to change, and the shipped "
              f"basis is\n    now MEASURED on these weights rather than transferred.")

    if d <= 0:
        print(f"\n  ⚠ the gain does NOT survive the si_rl basis ({d:+.4f}). Do not build the "
              f"archive.\n    The rl-basis +0.0026 would then be basis-specific, which is a "
              f"finding in itself.")
    else:
        print(f"\n  next: build the archive with 30 checkpoints in assets/cnn/sm0, tta={PRIMARY},"
              f"\n    declared oof {rows[PRIMARY]:.4f}. Then container-verify, smoke-test, archive "
              f"with its CV\n    in the filename, and log it in the archive ledger.")


if __name__ == "__main__":
    main()
