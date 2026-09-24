"""Architecture-diversity and augmentation-fix training.

**`scripts/train_cnn.py` is not modified.** This script imports its `load_data`
and `epoch_metrics` verbatim, so data loading, fold ordering, float16 handling and
SCALE are provably identical to every result already recorded -- only the
architecture and the augmentation axis mapping are new.

    python scripts/train_cnn2.py --arch datnet  --aug fixed --cubes cubes_hires.npy --stem-stride 3 --epochs 30 --seeds 0 1 2 --tta-flip --tag cnn2_datnet_fixaug
    python scripts/train_cnn2.py --arch siamese --aug fixed --cubes cubes_hires.npy --stem-stride 3 --epochs 30 --seeds 0 1 2 --tta-flip --tag cnn2_siamese_fixaug

`--aug orig` reproduces the historical (buggy) S-I-mirror / sagittal-rotation
behaviour so an arm can be compared against the existing results without the fix
as a confound. `--aug fixed` mirrors R-L and rotates axially, which is what the
original docstring always claimed. See `scripts/verify_aug_axes.py` for proof of
which axes each one touches.
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
from datpark.nets import apply_affine, count_params, random_affine_grid  # noqa: E402
from datpark.losses import get_loss  # noqa: E402
from datpark.nets2 import build, random_affine_grid_anat  # noqa: E402
from train_cnn import epoch_metrics, load_data  # noqa: E402  identical data path


def make_grid_fn(args):
    """Return the affine-grid builder this run should use."""
    if args.aug == "orig":
        return lambda n, dev, gen: random_affine_grid(
            n, dev, rot_deg=args.rot, trans=args.trans, scale=args.scale,
            flip_lr=not args.no_flip, generator=gen)
    return lambda n, dev, gen: random_affine_grid_anat(
        n, dev, rot_deg=args.rot, trans=args.trans, scale=args.scale,
        flip_lr=not args.no_flip, generator=gen, rot_plane=args.rot_plane)


def train_fold(Xtr, ytr, Xva, args, device, seed: int, grid_fn):
    """Same loop as train_cnn.train_fold, with the arch, grid builder and loss injected."""
    loss_fn = get_loss(args.loss)
    loss_kw = {"q": args.loss_q, "beta": args.loss_beta, "drop_frac": args.loss_drop}
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    kw = dict(dropout=args.dropout, stem_stride=args.stem_stride)
    if args.widths:                      # None => build() called exactly as before
        kw["widths"] = tuple(int(x) for x in args.widths.split(","))
    model = (build(args.arch, **kw)
             .to(device).to(memory_format=torch.channels_last_3d))
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
            xb = xb.to(memory_format=torch.channels_last_3d)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xb)
                tgt = yb * (1 - args.smooth) + 0.5 * args.smooth
                loss = loss_fn(logit.float(), tgt, **loss_kw)
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
            if args.tta_flip:  # per-sample, so rule-compliant; a no-op for siamese
                p = (p + torch.sigmoid(model(torch.flip(xb, dims=[2])).float())) / 2
            preds.append(p.cpu())
    return torch.cat(preds).numpy(), model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="datnet", choices=["datnet", "siamese"])
    ap.add_argument("--aug", default="fixed", choices=["fixed", "orig"],
                    help="fixed = mirror R-L + axial rotation (what the original "
                         "docstring claimed); orig = the historical S-I mirror + "
                         "sagittal rotation, for confound-free comparison")
    ap.add_argument("--rot-plane", default="axial", choices=["axial", "sagittal"])
    ap.add_argument("--cubes", default="cubes_hires.npy")
    # Alternative fold table, for the more-data-per-model experiment (folds10.csv =
    # leave-one-centre-out, ~90% training fraction). Default None keeps folds.csv, so
    # every prior run reproduces bit-for-bit.
    ap.add_argument("--folds", default=None,
                    help="e.g. folds10.csv; must cover exactly the same uids as folds.csv")
    ap.add_argument("--n-splits", type=int, default=None,
                    help="number of folds; defaults to datpark.cv.N_SPLITS (5)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--stem-stride", type=int, default=3)
    ap.add_argument("--widths", default=None,
                    help="comma-separated channel widths, e.g. 32,64,128,224. Default "
                         "None keeps DatNet's own (24,48,96,160) so prior runs reproduce.")
    ap.add_argument("--smooth", type=float, default=0.05)
    # Noise-robust objectives (datpark/losses.py, ). Defaults are the
    # neutral settings, so --loss bce reproduces every result recorded before this flag.
    ap.add_argument("--loss", default="bce",
                    choices=["bce", "gce", "sce", "bootstrap", "trimmed"])
    ap.add_argument("--loss-q", type=float, default=0.7, help="gce only")
    ap.add_argument("--loss-beta", type=float, default=0.5,
                    help="sce reverse-CE weight, or bootstrap target-mixing weight")
    ap.add_argument("--loss-drop", type=float, default=0.05, help="trimmed only")
    ap.add_argument("--rot", type=float, default=10.0)
    ap.add_argument("--trans", type=float, default=0.06)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--intensity", type=float, default=0.1)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="cnn2")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    device = "cuda"
    folds, X, y = load_data(args.cubes)
    n_splits = args.n_splits or N_SPLITS
    if args.folds:
        # X order is fixed by load_data (folds.csv order); swap ONLY the fold assignment,
        # reindexed by uid, so the cube tensor and labels stay exactly as loaded.
        alt = pd.read_csv(ROOT / "artifacts" / args.folds).set_index("uid")
        assert set(alt.index) == set(folds.uid), f"{args.folds} uids differ from folds.csv"
        alt = alt.reindex(folds.uid)
        assert (alt.y.to_numpy() == folds.y.to_numpy()).all(), "labels differ"
        for c in [c for c in alt.columns if c.startswith("fold_s")]:
            folds[c] = alt[c].to_numpy()
        folds["group"] = alt["group"].to_numpy()
        print(f"fold table: {args.folds}  n_splits={n_splits}  "
              f"groups={folds.group.nunique()}")
    _kw = dict(stem_stride=args.stem_stride)
    if args.widths:
        _kw["widths"] = tuple(int(x) for x in args.widths.split(","))
    m = build(args.arch, **_kw)
    print(f"arch={args.arch}  aug={args.aug}"
          f"{'/' + args.rot_plane if args.aug == 'fixed' else ''}  "
          f"n={len(X)}  cube={tuple(X.shape[2:])}  stem_stride={args.stem_stride}  "
          f"params={count_params(m):,}")
    print(f"flip-aug={'off' if args.no_flip else 'on'}  tta-flip={args.tta_flip}  "
          f"epochs={args.epochs}  seeds={args.seeds}")
    print(f"loss={args.loss}  q={args.loss_q}  beta={args.loss_beta}  drop={args.loss_drop}  "
          f"smooth={args.smooth}")
    if args.arch == "siamese":
        print("NOTE siamese is exactly L<->R invariant: flip aug and mirror TTA are "
              "no-ops on this arm (verify_aug_axes.py check 4).")

    grid_fn = make_grid_fn(args)
    per_seed, oof_cols = [], {}
    for seed in args.seeds:
        oof = np.full(len(X), np.nan)
        fold = folds[f"fold_s{seed}"].to_numpy()
        t0 = time.time()
        for k in range(n_splits):
            va = fold == k
            p, model = train_fold(X[~va], y[~va], X[va], args, device,
                                  seed * 100 + k, grid_fn)
            oof[va] = p
            if args.save_dir:
                sd = Path(args.save_dir); sd.mkdir(parents=True, exist_ok=True)
                torch.save({k2: v.cpu() for k2, v in model.state_dict().items()},
                           sd / f"{args.arch}_s{seed}_f{k}.pt")
            print(f"  seed {seed} fold {k}: n_va={va.sum():>4}  "
                  f"logloss {score(y.numpy()[va], p)['log_loss']:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        assert not np.isnan(oof).any()
        s = score(y.numpy(), oof)
        per_seed.append(s)
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
