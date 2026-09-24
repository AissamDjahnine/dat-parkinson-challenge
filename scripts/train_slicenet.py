"""Train the 2.5D SliceNet family on the frozen grouped folds.

`scripts/train_cnn.py` and `train_cnn2.py` are NOT modified. This imports their
`load_data` verbatim, so data loading, fold ordering, float16 handling and SCALE are
provably identical to every recorded result -- only the architecture differs.

    python scripts/train_slicenet.py --cubes cubes_hires.npy --tag slice_128_pos
    python scripts/train_slicenet.py --cubes cubes_tight.npy --no-pos-embed --tag slice_96_nopos

Memory note: SliceNet folds the slice axis into the batch, so the effective 2D batch is
(batch x slices). At 96^3 with batch 32 that is 3072 images of 96x96 -- fine, but the
default batch is lowered to 16 to keep activations comfortable on a 16 GB card. Batch
size is therefore NOT matched to the 3D families; that is a deliberate, recorded
difference (matching it would OOM at 96 slices).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datpark.cv import N_SPLITS  # noqa: E402
from datpark.metrics import report, score  # noqa: E402
from datpark.augment3 import blur_noise  # noqa: E402
from datpark.nets import apply_affine, count_params, random_affine_grid  # noqa: E402
from datpark.nets2 import random_affine_grid_anat  # noqa: E402
from datpark.nets3 import SliceNet  # noqa: E402
from train_cnn import load_data  # noqa: E402  identical data path


def make_grid_fn(args):
    if args.aug == "orig":
        return lambda n, dev, gen: random_affine_grid(
            n, dev, rot_deg=args.rot, trans=args.trans, scale=args.scale,
            flip_lr=not args.no_flip, generator=gen)
    return lambda n, dev, gen: random_affine_grid_anat(
        n, dev, rot_deg=args.rot, trans=args.trans, scale=args.scale,
        flip_lr=not args.no_flip, generator=gen, rot_plane=args.rot_plane)


def build(args):
    return SliceNet(dropout=args.dropout, stem_stride=args.stem_stride,
                    pos_embed=not args.no_pos_embed)


def train_fold(Xtr, ytr, Xva, args, device, seed: int, grid_fn):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = build(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, (len(Xtr) + args.batch - 1) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.25)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xva = Xva.to(device)
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
            # ---- 011 acquisition-physics augmentation: blur + correlated noise ----------
            # Same insertion point as scripts/train_bn.py: after geometry and intensity, since it
            # is an intensity-domain operation. `--bn-m 0` is an EXACT no-op (asserted in
            # tests/test_augment3.py), so the default reproduces this trainer's prior behaviour
            # bit for bit and the shipped artifacts remain valid controls.
            if args.bn_m > 0:
                xb = blur_noise(xb, M=args.bn_m, P=args.bn_p, voxel_mm=args.bn_voxel_mm,
                                blur=not args.bn_no_blur, noise=not args.bn_no_noise,
                                generator=gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xb)
                tgt = yb * (1 - args.smooth) + 0.5 * args.smooth
                loss = F.binary_cross_entropy_with_logits(logit.float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(Xva), args.batch):
            xb = Xva[s : s + args.batch].float()
            p = torch.sigmoid(model(xb).float())
            if args.tta_flip:  # per-sample R-L mirror; rule-compliant
                p = (p + torch.sigmoid(model(torch.flip(xb, dims=[2])).float())) / 2
            preds.append(p.cpu())
    return torch.cat(preds).numpy(), model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", default="cubes_hires.npy")
    ap.add_argument("--no-pos-embed", action="store_true",
                    help="train the S-I-invariant variant (attention pooling is a "
                         "weighted sum, so without positional encoding the net is "
                         "exactly invariant to an S-I flip)")
    ap.add_argument("--aug", default="orig", choices=["orig", "fixed"])
    ap.add_argument("--rot-plane", default="axial", choices=["axial", "sagittal"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--stem-stride", type=int, default=2)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--rot", type=float, default=15.0)
    ap.add_argument("--trans", type=float, default=0.090)
    ap.add_argument("--scale", type=float, default=0.075)
    ap.add_argument("--bn-m", type=float, default=0.0,
                    help="011 blur+noise magnitude multiplier; 0 disables (exact no-op). "
                         "The J Nucl Med 2024 optimum is 2.5")
    ap.add_argument("--bn-p", type=float, default=0.9)
    ap.add_argument("--bn-no-blur", action="store_true")
    ap.add_argument("--bn-no-noise", action="store_true")
    ap.add_argument("--bn-voxel-mm", type=float, default=1.3333)
    ap.add_argument("--intensity", type=float, default=0.1)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="slicenet")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    device = "cuda"
    folds, X, y = load_data(args.cubes)
    print(f"arch=slicenet  pos_embed={not args.no_pos_embed}  n={len(X)}  "
          f"cube={tuple(X.shape[2:])}  slices={X.shape[-1]}  "
          f"params={count_params(build(args)):,}")
    print(f"aug={args.aug}  smooth={args.smooth}  batch={args.batch}  "
          f"epochs={args.epochs}  seeds={args.seeds}  tta-flip={args.tta_flip}")

    grid_fn = make_grid_fn(args)
    oof_cols = {}
    for seed in args.seeds:
        oof = np.full(len(X), np.nan)
        fold = folds[f"fold_s{seed}"].to_numpy()
        t0 = time.time()
        for k in range(N_SPLITS):
            va = fold == k
            p, model = train_fold(X[~va], y[~va], X[va], args, device,
                                  seed * 100 + k, grid_fn)
            oof[va] = p
            if args.save_dir:
                sd = Path(args.save_dir); sd.mkdir(parents=True, exist_ok=True)
                torch.save({k2: v.cpu() for k2, v in model.state_dict().items()},
                           sd / f"slicenet_s{seed}_f{k}.pt")
            print(f"  seed {seed} fold {k}: n_va={va.sum():>4}  "
                  f"logloss {score(y.numpy()[va], p)['log_loss']:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        assert not np.isnan(oof).any()
        s = score(y.numpy(), oof)
        oof_cols[f"oof_s{seed}"] = oof
        print(f"seed {seed}: log loss {s['log_loss']:.4f}  AUROC {s['auroc']:.4f}",
              flush=True)

    mean_oof = np.mean(list(oof_cols.values()), axis=0)
    report(y.numpy(), mean_oof, folds.group.to_numpy(),
           title=f"{args.tag}: seed-averaged OOF")
    out = ROOT / "artifacts" / "oof" / f"{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"uid": folds.uid, "y": y.numpy(), **oof_cols,
                  "oof_mean": mean_oof}).to_csv(out, index=False)
    print(f"\nOOF -> {out}")


if __name__ == "__main__":
    main()
