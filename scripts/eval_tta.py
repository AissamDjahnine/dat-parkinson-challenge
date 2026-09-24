"""Evaluate richer test-time augmentation on ALREADY-TRAINED checkpoints.

No retraining: TTA is an inference-time choice, so every scheme is scored by
re-running the saved fold models over their own validation folds. Minutes, not
hours, and the OOF stays directly comparable to every recorded result.

    python scripts/eval_tta.py --family aug15x
    python scripts/eval_tta.py --family siamese
    python scripts/eval_tta.py --family r128        # loads the 5.7 GB cache

★ The idea worth testing first. The shipped mirror TTA flips **R-L**
(`torch.flip(x, dims=[2])`), but an earlier analysis established that training augmented with an
**S-I** flip — the R-L flip was never applied during training at all. So the models
were made invariant to S-I mirroring and were never asked to be invariant to R-L
mirroring. TTA should average over the transform the model was *trained* to be
invariant to. Averaging over an unseen transform is at best noise reduction and at
worst evaluates the model off its training manifold.

That predicts S-I flip TTA > R-L flip TTA, which is free to check and would mean the
shipped submission has been using the wrong mirror all along.

Rule compliance: every view is a per-sample deterministic transform. No dataset
statistics, no cross-sample information, nothing fitted at inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datpark.cv import N_SPLITS  # noqa: E402
from datpark.metrics import score  # noqa: E402
from datpark.nets import DatNet, apply_affine  # noqa: E402
from datpark.nets2 import SiameseDatNet  # noqa: E402
from train_cnn import load_data  # noqa: E402

# theta row -> anatomical axis, under affine_grid's (x, y, z) = (W, H, D) order.
ROW_SI, ROW_AP, ROW_RL = 0, 1, 2

FAMILIES = {
    # name:      (ckpt dir,   cubes,               builder,                   prefix)
    "aug15x":  ("aug15x",  "cubes_hires.npy", lambda: DatNet(stem_stride=3),        "datnet"),
    "r128":    ("r128",    "cubes_r128.npy",  lambda: DatNet(stem_stride=4),        "cnn"),
    "siamese": ("siamese", "cubes_hires.npy", lambda: SiameseDatNet(stem_stride=3), "siamese"),
}


def theta(flip_si=False, flip_rl=False, rot_deg=0.0, scale=1.0, shift=0.0):
    """One deterministic view. Rotation is in the SAGITTAL plane (rows 0,1), which
    is what training actually used — TTA should match the trained manifold."""
    t = torch.zeros(1, 3, 4)
    a = np.deg2rad(rot_deg)
    c, s = float(np.cos(a)), float(np.sin(a))
    t[:, ROW_SI, ROW_SI] = c * scale
    t[:, ROW_SI, ROW_AP] = -s * scale
    t[:, ROW_AP, ROW_SI] = s * scale
    t[:, ROW_AP, ROW_AP] = c * scale
    t[:, ROW_RL, ROW_RL] = scale
    if flip_si:
        t[:, ROW_SI, :] *= -1
    if flip_rl:
        t[:, ROW_RL, :] *= -1
    t[:, ROW_SI, 3] = shift
    return t


def scheme_views(name: str):
    """Return a list of thetas (None = identity, skips grid_sample entirely)."""
    if name == "none":
        return [None]
    if name == "rl":        # the shipped scheme
        return [None, theta(flip_rl=True)]
    if name == "si":        # the transform training actually used
        return [None, theta(flip_si=True)]
    if name == "si_rl":
        return [None, theta(flip_si=True), theta(flip_rl=True),
                theta(flip_si=True, flip_rl=True)]
    if name == "rot":
        return [None] + [theta(rot_deg=r) for r in (-7.0, 7.0)]
    if name == "si_rot":
        return ([None, theta(flip_si=True)]
                + [theta(rot_deg=r) for r in (-7.0, 7.0)]
                + [theta(flip_si=True, rot_deg=r) for r in (-7.0, 7.0)])
    if name == "rl_rot":
        return ([None, theta(flip_rl=True)]
                + [theta(rot_deg=r) for r in (-7.0, 7.0)]
                + [theta(flip_rl=True, rot_deg=r) for r in (-7.0, 7.0)])
    if name == "si_rot_scale":
        v = [None, theta(flip_si=True)]
        for r in (-7.0, 7.0):
            v += [theta(rot_deg=r), theta(flip_si=True, rot_deg=r)]
        for sc in (0.96, 1.04):
            v += [theta(scale=sc), theta(flip_si=True, scale=sc)]
        return v
    raise ValueError(name)


SCHEMES = ["none", "rl", "si", "si_rl", "rot", "rl_rot", "si_rot", "si_rot_scale"]


def predict(net, X, views, device, batch=32):
    """Mean probability over the views. Per-sample throughout."""
    out = np.zeros(len(X), dtype=np.float64)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(X), batch):
            xb = X[s : s + batch].float().to(device)
            acc = torch.zeros(len(xb), dtype=torch.float64, device=device)
            for v in views:
                z = xb if v is None else apply_affine(xb, v.to(device).expand(len(xb), 3, 4))
                acc += torch.sigmoid(net(z.to(memory_format=torch.channels_last_3d)).float()).double()
            out[s : s + len(xb)] = (acc / len(views)).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="aug15x", choices=list(FAMILIES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--save", action="store_true", help="write OOF csv per scheme")
    args = ap.parse_args()

    ckdir, cubes, builder, prefix = FAMILIES[args.family]
    device = "cuda"
    folds, X, y = load_data(cubes)
    yv = y.numpy()
    ck = ROOT / "artifacts" / "ckpt" / ckdir
    print(f"family={args.family}  cubes={cubes}  ckpt={ck}  n={len(X)}")

    oof = {s: np.full(len(X), np.nan) for s in SCHEMES}
    acc = {s: [] for s in SCHEMES}
    for seed in args.seeds:
        fold = folds[f"fold_s{seed}"].to_numpy()
        for k in range(N_SPLITS):
            va = fold == k
            net = builder().to(device)
            net.load_state_dict(torch.load(ck / f"{prefix}_s{seed}_f{k}.pt",
                                           map_location="cpu", weights_only=True))
            net.eval().to(memory_format=torch.channels_last_3d)
            Xv = X[va]
            for s in SCHEMES:
                oof[s][va] = predict(net, Xv, scheme_views(s), device)
        for s in SCHEMES:
            assert not np.isnan(oof[s]).any()
            acc[s].append(oof[s].copy())
        print(f"  seed {seed} done", flush=True)

    print(f"\n{'scheme':>14}{'views':>7}{'logloss':>10}{'AUROC':>9}{'vs rl':>9}")
    base = None
    res = {}
    for s in SCHEMES:
        m = np.mean(acc[s], axis=0)
        sc = score(yv, m)
        res[s] = m
        if s == "rl":
            base = sc["log_loss"]
    for s in SCHEMES:
        sc = score(yv, res[s])
        d = "" if base is None else f"{base - sc['log_loss']:+9.4f}"
        print(f"{s:>14}{len(scheme_views(s)):>7}{sc['log_loss']:>10.4f}{sc['auroc']:>9.4f}{d}")

    if args.save:
        out = ROOT / "artifacts" / "oof"
        for s in SCHEMES:
            pd.DataFrame({"uid": folds.uid, "y": yv, "oof_mean": res[s]}).to_csv(
                out / f"tta_{args.family}_{s}.csv", index=False)
        print(f"\nOOF -> {out}/tta_{args.family}_*.csv")


if __name__ == "__main__":
    main()
