"""011: acquisition-physics augmentation — strong Gaussian blur and correlated noise.

    python scripts/train_bn.py --bn-m 2.5 --bn-p 0.9 --tag bn_m25

Additive: `train_cnn2.py`, `train_aug2.py` and `train_cnn3.py` are untouched. The loop below is
`train_cnn2.train_fold` with ONE hook — `datpark.augment3.blur_noise` applied after the affine and
intensity augmentation, since it is an intensity-domain operation. With `--bn-m 0` it is the
champion configuration, which is what the control arm verifies against the recorded **0.2694**.

Provenance, dose and the reason this is not the augmentation lever we already closed are all in
`datpark/augment3.py`. Short version: Buddenkotte & Buchert, J Nucl Med 2024;65(9):1463, same
modality and task at n=1100, cross-site accuracy 0.960->0.989 and 0.953->0.975 on two independent
OOD sets, where *realistic*-magnitude augmentation was worse than nothing. 009 swept geometric,
mixing and occlusion augmentation; this is acquisition physics, which is the axis our 10 hospitals
actually differ on.

★ PRIMARY HYPOTHESIS, designated before any number exists: **bn_m25** (M=2.5, P=0.9), the paper's
own optimum. The other arms are descriptive — dose-response direction and component attribution —
so that this is not a best-of-five selection.

★ THE BAR, unchanged from every other sweep: beat the control by **> 0.010** with all three seeds
agreeing, i.e. below **0.2594**.

⚠ EXPECT LESS THAN THEIR HEADLINE. Their gain came from genuinely different scanners. All 20 scans
of the competition's demonstration test set fall into acquisition groups that already exist in our
training data, so the real test split is a random draw over the SAME 10 centres. Our grouped CV
holds out whole centres and is therefore HARDER than the real test on this axis — if anything it
over-rewards this method. Visual check passed: at M=2.5 an abnormal
scan still reads abnormal and a normal scan still reads normal, which is the test that killed the
009 erasing arm.
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
from datpark.augment3 import blur_noise  # noqa: E402
from datpark.losses import get_loss  # noqa: E402
from datpark.metrics import report, score  # noqa: E402
from datpark.nets import apply_affine  # noqa: E402
from datpark.nets2 import build  # noqa: E402
from train_cnn import load_data  # noqa: E402  identical data path
from train_cnn2 import make_grid_fn  # noqa: E402  identical affine grid


def train_fold(Xtr, ytr, Xva, args, device, seed, grid_fn):
    """train_cnn2.train_fold plus a gamma passthrough and an optional EMA shadow."""
    loss_fn = get_loss(args.loss)
    loss_kw = {"q": args.loss_q, "beta": args.loss_beta, "drop_frac": args.loss_drop,
               "gamma": args.focal_gamma, "focal_normalise": not args.focal_raw}
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    kw = dict(dropout=args.dropout, stem_stride=args.stem_stride)
    if args.widths:
        kw["widths"] = tuple(int(x) for x in args.widths.split(","))
    model = build(args.arch, **kw).to(device).to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, (len(Xtr) + args.batch - 1) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.25)
    Xtr, ytr, Xva = Xtr.to(device), ytr.to(device), Xva.to(device)
    n = len(Xtr)

    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device, generator=gen)
        for s in range(0, n, args.batch):
            b = perm[s : s + args.batch]
            xb, yb = Xtr[b].float(), ytr[b]
            xb = apply_affine(xb, grid_fn(len(b), device, gen))
            if args.intensity > 0:
                xb = xb * (1 + (torch.rand(len(b), 1, 1, 1, 1, device=device,
                                           generator=gen) * 2 - 1) * args.intensity)
            # ---- acquisition physics: blur + correlated noise, AFTER geometry and intensity ---
            # Intensity-domain, so it belongs after both. With --bn-m 0 this is a no-op and the
            # arm is the champion, which is what the control verifies.
            if args.bn_m > 0:
                xb = blur_noise(xb, M=args.bn_m, P=args.bn_p, voxel_mm=args.voxel_mm,
                                blur=not args.bn_no_blur, noise=not args.bn_no_noise,
                                generator=gen,
                                blur_axes=(0, 1) if args.bn_blur_2d else (0, 1, 2))
            tgt = yb * (1 - args.smooth) + 0.5 * args.smooth
            xb = xb.to(memory_format=torch.channels_last_3d)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(xb).float(), tgt, **loss_kw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(Xva), 64):
            xb = Xva[s : s + 64].float().to(memory_format=torch.channels_last_3d)
            p = torch.sigmoid(model(xb).float())
            if args.tta_flip:
                p = (p + torch.sigmoid(model(torch.flip(xb, dims=[2])).float())) / 2
            preds.append(p.cpu())
    return torch.cat(preds).numpy(), model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal-gamma", type=float, default=1.0,
                    help="only used when --loss focal; 0 recovers bce exactly")
    ap.add_argument("--focal-raw", action="store_true",
                    help="disable the gradient-scale normalisation. Off by default because raw "
                         "focal shrinks the loss and mimics a lower learning rate, which would "
                         "confound the arm")
    ap.add_argument("--bn-m", type=float, default=0.0,
                    help="magnitude multiplier M; 0 disables. The paper's optimum is 2.5")
    ap.add_argument("--bn-p", type=float, default=0.9,
                    help="Bernoulli probability that each component is applied")
    ap.add_argument("--bn-no-blur", action="store_true", help="noise only, for attribution")
    ap.add_argument("--bn-no-noise", action="store_true", help="blur only, for attribution")
    ap.add_argument("--bn-blur-2d", action="store_true",
                    help="blur the AXIAL PLANE ONLY, leaving superior-inferior untouched — what "
                         "Buddenkotte & Buchert actually do on their 2-D slabs. Default "
                         "off = isotropic 3-D, which is what every 011 arm used. RNG consumption is "
                         "identical either way, so the arms are directly comparable.")
    ap.add_argument("--voxel-mm", type=float, default=1.3333)
    # everything below is the champion configuration
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="cnn3")
    ap.add_argument("--save-dir", default=None,
                    help="ALWAYS set this — a candidate found under submission-window pressure "
                         "must not need a retrain before it can be packed and TTA-scored")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds, X, y = load_data(args.cubes)
    grid_fn = make_grid_fn(args)
    yv = y.numpy().astype(np.float64)

    print(f"tag={args.tag}  loss={args.loss} gamma={args.focal_gamma} "
          f"bn_m={args.bn_m} bn_p={args.bn_p}  seeds={args.seeds}", flush=True)
    if args.loss == "bce" and args.bn_m == 0:
        print("  (no blur/noise: this is the CONTROL and must reproduce the champion 0.2694)",
              flush=True)

    oof = np.full((len(folds), len(args.seeds)), np.nan, dtype=np.float64)
    t0 = time.time()
    for si, seed in enumerate(args.seeds):
        col = f"fold_s{seed}"
        for k in range(N_SPLITS):
            va = (folds[col] == k).to_numpy()
            # per-fold seed matches train_cnn2.py:189 — plain `seed` would give all five folds
            # one shared RNG stream and the control would not reproduce the champion
            p, model = train_fold(X[~va], y[~va], X[va], args, device, seed * 100 + k, grid_fn)
            oof[va, si] = p
            if args.save_dir:
                sd = ROOT / args.save_dir
                sd.mkdir(parents=True, exist_ok=True)
                torch.save({k2: v.cpu() for k2, v in model.state_dict().items()},
                           sd / f"{args.arch}_s{seed}_f{k}.pt")
            print(f"  seed {seed} fold {k}: n_va={len(p):4d}  "
                  f"logloss {score(yv[va], p)['log_loss']:.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)

    assert not np.isnan(oof).any(), (
        "some scans received no prediction — a fold column does not partition all rows. "
        "An unfilled row scores as ~13.8 log loss and would produce a catastrophically wrong "
        "arm that still looks like a number."
    )
    mean = oof.mean(axis=1)
    out = pd.DataFrame({"uid": folds.uid, "y": folds.y, "oof_mean": mean})
    for si, seed in enumerate(args.seeds):
        out[f"oof_s{seed}"] = oof[:, si]
    dest = ROOT / "artifacts" / "oof" / f"{args.tag}.csv"
    tmp = dest.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, dest)              # atomic: a truncated OOF must never look complete

    print()
    for si, seed in enumerate(args.seeds):
        print(f"  seed {seed}: {score(yv, oof[:, si])['log_loss']:.4f}")
    print(report(folds.y.to_numpy(), mean))
    print("  reference: control = 0.2694. Bar to matter: < 0.2594, all three seeds agreeing.")


if __name__ == "__main__":
    main()
