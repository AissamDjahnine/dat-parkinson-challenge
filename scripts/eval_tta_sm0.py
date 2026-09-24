"""Score every TTA scheme on the `sm0` checkpoints — the family 005 actually ships.

    python scripts/eval_tta_sm0.py --save

Inference only over the 15 existing checkpoints in artifacts/ckpt/sm0. No training.

── WHY THIS EXISTS: 005's quoted CV describes a model it does not ship ──────────────────────
`inference/assets/config.json` declares the sm0 family as `"tta": "si_rl"` (4 views)
and records `"oof": {"log_loss": 0.2694}`. But that 0.2694 comes from
`artifacts/oof/cnn2_sm0.0.csv`, produced by `train_cnn2.py --tta-flip`, which is the plain
2-view R-L mirror — scheme `rl`, not `si_rl`.

`si_rl` was never scored on sm0's own weights. `eval_tta.py`'s FAMILIES dict knows only
aug15x / r128 / siamese, so `artifacts/oof/tta_sm0_*` does not exist. The choice was
TRANSFERRED from the aug15x family, where si_rl beat rl by +0.0013 on different weights.

Two things this run settles:
  1. the true CV of the configuration that ships (currently an extrapolation, not a number);
  2. whether `si_rl` is even the right scheme for sm0. The best scheme differs per family with
     no consistent ordering — si_rot_scale for siamese, si_rl for aug15x, rl_rot for r128 —
     so assuming aug15x's winner carries over is exactly the kind of transfer this project has
     been burned by. If another scheme wins, switching costs a one-line config.json edit and a
     repack: no retraining, no new weights.

── WHY A SEPARATE FILE ──────────────────────────────────────────────────────────────────────
`eval_tta.py` produced the OOF files that 003/004/005 were built on. It is left untouched so
those results stay bit-reproducible; this script imports it and registers one extra family.
The checkpoints in artifacts/ckpt/sm0 were verified byte-identical (md5 of the concatenated
15 files) to inference/assets/cnn/sm0, so what is measured here is what runs.

── BUILT-IN VALIDATION ──────────────────────────────────────────────────────────────────────
Scheme `rl` must reproduce the recorded 0.2694, because `rl` IS what --tta-flip did. If it
does not, the harness disagrees with the trainer and every number below is suspect. The same
check passed for the siamese family: tta_siamese_rl matched its --tta-flip OOF to 0.0e+00.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_tta  # noqa: E402  the untouched original
from datpark.nets import DatNet  # noqa: E402

# The shipped family: same architecture and cube grid as aug15x, different checkpoints.
eval_tta.FAMILIES["sm0"] = ("sm0", "cubes_hires.npy", lambda: DatNet(stem_stride=3), "datnet")

RECORDED_RL = 0.2694          # artifacts/oof/cnn2_sm0.0.csv, produced with --tta-flip
SHIPPED_SCHEME = "si_rl"      # what assets/config.json currently declares


def main() -> None:
    argv = sys.argv[1:]
    sys.argv = ["eval_tta.py", "--family", "sm0"] + argv
    eval_tta.main()

    # ---- read back and judge -----------------------------------------------------------
    oof = ROOT / "artifacts" / "oof"
    folds = pd.read_csv(ROOT / "artifacts" / "folds.csv")
    y = folds.y.to_numpy().astype(float)

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())

    def get(name, col="oof_mean"):
        f = oof / f"{name}.csv"
        if not f.exists():
            return None
        return pd.read_csv(f).set_index("uid").reindex(folds.uid)[col].to_numpy()

    scored = {s: get(f"tta_sm0_{s}") for s in eval_tta.SCHEMES}
    scored = {s: v for s, v in scored.items() if v is not None}
    if not scored:
        print("\n(no OOF written — pass --save to persist the per-scheme predictions)")
        return

    print(f"\n{'scheme':<16}{'log loss':>10}   vs shipped si_rl")
    ship = ll(scored[SHIPPED_SCHEME]) if SHIPPED_SCHEME in scored else None
    for s, p in sorted(scored.items(), key=lambda kv: ll(kv[1])):
        mark = "  <- SHIPPED" if s == SHIPPED_SCHEME else ""
        delta = "" if ship is None else f"{ship - ll(p):+9.4f}"
        print(f"{s:<16}{ll(p):>10.4f}{delta}{mark}")

    # validation: `rl` must reproduce the trainer's own number
    if "rl" in scored:
        d = abs(ll(scored["rl"]) - RECORDED_RL)
        verdict = "OK" if d < 0.0005 else "!!! HARNESS DISAGREES WITH THE TRAINER"
        print(f"\nvalidation: scheme 'rl' = {ll(scored['rl']):.4f} vs recorded "
              f"{RECORDED_RL:.4f} (|d| = {d:.4f})  {verdict}")

    # what it means for the blend that actually ships
    sia, sbr = get("tta_siamese_si_rot_scale"), get("sbr_lgb", "oof_raw")
    if sia is not None and sbr is not None:
        lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
        sg = lambda z: 1.0 / (1.0 + np.exp(-z))
        print(f"\n005 blend using each sm0 scheme (0.8*mean(sm0, siamese) + 0.2*SBR):")
        rows = {s: ll(sg(0.8 * (lg(p) + lg(sia)) / 2 + 0.2 * lg(sbr))) for s, p in scored.items()}
        for s, v in sorted(rows.items(), key=lambda kv: kv[1]):
            mark = "  <- what 005 ships today" if s == SHIPPED_SCHEME else ""
            print(f"   {s:<16}{v:.4f}{mark}")
        best = min(rows, key=rows.get)
        if SHIPPED_SCHEME in rows:
            gain = rows[SHIPPED_SCHEME] - rows[best]
            print(f"\n   best scheme is '{best}' — worth {gain:+.4f} over the shipped "
                  f"'{SHIPPED_SCHEME}'")
            if best != SHIPPED_SCHEME and gain > 0.002:
                print("   ★ WORTH ACTING ON: edit assets/config.json cnn_families.sm0.tta and"
                      "\n     repack. No retraining — the weights are unchanged.")
            elif best != SHIPPED_SCHEME:
                print("   too small to act on: 8 schemes were compared, so a sub-0.002 win is"
                      "\n     selection. Keep the shipped scheme.")
            else:
                print("   the shipped scheme is already the best. Nothing to change; we now"
                      "\n     have a MEASURED number for it instead of a transferred guess.")


if __name__ == "__main__":
    main()
