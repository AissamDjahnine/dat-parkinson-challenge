"""Train the 3D CNN track on the frozen grouped folds.

Uses the same folds as every other track so out-of-fold predictions stay
directly comparable and ensemblable.

    python scripts/train_cnn.py --epochs 60
    python scripts/train_cnn.py --epochs 60 --no-flip   # test LR flip
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

from datpark.cv import N_SPLITS  # noqa: E402
from datpark.metrics import report, score  # noqa: E402
from datpark.nets import DatNet, apply_affine, count_params, random_affine_grid  # noqa: E402

CACHE = ROOT / "artifacts" / "cache"
SCALE = 6.0  # cubes are reference-multiples ~0-12; this puts them near unit range


def load_data(cubes_file: str):
    """Load a cube cache, reordered to match artifacts/folds.csv.

    Kept in float16: at 128^3 a float32 copy is 11 GB, which does not fit
    alongside activations on a 16 GB card. Values are reference-multiples in
    [0, 12/SCALE], so float16 is far more precision than the data carries, and
    every batch is cast to float32 before augmentation anyway.

    The uid order comes from the *matching* meta file, not always meta_v1: both
    are built from the same sorted glob so they agree today, but a silent
    mismatch would scramble labels invisibly, hence the assert.
    """
    folds = pd.read_csv(ROOT / "artifacts" / "folds.csv")
    tag = Path(cubes_file).stem.removeprefix("cubes_")
    meta = CACHE / f"meta_{tag}.csv"
    order = pd.read_csv(meta if meta.exists() else CACHE / "meta_v1.csv").uid.tolist()
    assert set(order) == set(folds.uid), f"{meta.name} uids do not match folds.csv"
    pos = {u: i for i, u in enumerate(order)}
    cubes = np.load(CACHE / cubes_file, mmap_mode="r")
    idx = [pos[u] for u in folds.uid]
    X = (torch.from_numpy(np.asarray(cubes[idx], dtype=np.float16)) / SCALE).unsqueeze(1)
    y = torch.from_numpy(folds.y.to_numpy().astype(np.float32))
    return folds, X, y


def epoch_metrics(model, X, y, device, batch: int = 64) -> dict:
    """Full classification metric set on one fold, at threshold 0.5.

    NB: threshold-based metrics (accuracy / precision / sensitivity /
    specificity / F1) are diagnostics only. The competition scores log loss, so
    0.5 is an arbitrary operating point -- a model can improve log loss while
    accuracy is flat, and vice versa.
    """
    model.eval()
    ps = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(X), batch):
            xb = X[s : s + batch].to(device).float().to(memory_format=torch.channels_last_3d)
            ps.append(torch.sigmoid(model(xb).float()).cpu())
    p = torch.cat(ps).numpy().astype(np.float64)
    yy = y.numpy().astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    yhat = (pc >= 0.5).astype(int)
    tp = int(((yhat == 1) & (yy == 1)).sum()); tn = int(((yhat == 0) & (yy == 0)).sum())
    fp = int(((yhat == 1) & (yy == 0)).sum()); fn = int(((yhat == 0) & (yy == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    from sklearn.metrics import roc_auc_score
    return {
        "loss": float(-(yy * np.log(pc) + (1 - yy) * np.log1p(-pc)).mean()),
        "auroc": float(roc_auc_score(yy, pc)) if len(np.unique(yy)) > 1 else float("nan"),
        "accuracy": (tp + tn) / len(yy), "precision": prec,
        "sensitivity": sens, "specificity": spec, "f1": f1,
    }


def train_fold(Xtr, ytr, Xva, args, device, seed: int, yva=None, history=None,
               hist_key=None):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = (DatNet(dropout=args.dropout, stem_stride=args.stem_stride)
             .to(device).to(memory_format=torch.channels_last_3d))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, (len(Xtr) + args.batch - 1) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.25)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xva = Xva.to(device)
    n = len(Xtr)

    for ep in range(args.epochs):
        model.train()
        ep_loss, ep_batches = 0.0, 0
        perm = torch.randperm(n, device=device, generator=gen)
        for s in range(0, n, args.batch):
            b = perm[s : s + args.batch]
            xb, yb = Xtr[b].float(), ytr[b]
            theta = random_affine_grid(len(b), device, rot_deg=args.rot,
                                       trans=args.trans, scale=args.scale,
                                       flip_lr=not args.no_flip, generator=gen)
            xb = apply_affine(xb, theta)
            if args.intensity > 0:
                xb = xb * (1 + (torch.rand(len(b), 1, 1, 1, 1, device=device,
                                           generator=gen) * 2 - 1) * args.intensity)
            xb = xb.to(memory_format=torch.channels_last_3d)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xb)
                # label smoothing: ~1 in 5 exams are genuinely ambiguous, so
                # driving the model to hard 0/1 targets is fitting label noise
                tgt = yb * (1 - args.smooth) + 0.5 * args.smooth
                loss = F.binary_cross_entropy_with_logits(logit.float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += float(loss.detach()); ep_batches += 1

        if history is not None and yva is not None:
            m = epoch_metrics(model, Xva, yva, device)
            history.append({"key": hist_key, "epoch": ep + 1,
                            "train_loss": ep_loss / max(ep_batches, 1),
                            "lr": sched.get_last_lr()[0], **{f"val_{k}": v for k, v in m.items()}})
            model.train()

    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(Xva), 64):
            xb = Xva[s : s + 64].float().to(memory_format=torch.channels_last_3d)
            p = torch.sigmoid(model(xb).float())
            if args.tta_flip:  # mirror TTA is per-sample, so rule-compliant
                p = (p + torch.sigmoid(model(torch.flip(xb, dims=[2])).float())) / 2
            preds.append(p.cpu())
    return torch.cat(preds).numpy(), model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", default="cubes_v1.npy")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--stem-stride", type=int, default=2,
                    help="stem downsampling; set to voxels-per-4mm to keep the "
                         "post-stem feature map at a fixed physical scale "
                         "(2 at 2.0mm, 3 at 1.33mm, 4 at 1.0mm)")
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--rot", type=float, default=10.0)
    ap.add_argument("--trans", type=float, default=0.06)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--intensity", type=float, default=0.1)
    ap.add_argument("--no-flip", action="store_true", help="disable LR-flip augmentation")
    ap.add_argument("--tta-flip", action="store_true", help="average with mirrored input")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="cnn")
    ap.add_argument("--history", action="store_true",
                    help="record per-epoch validation metrics and write a CSV")
    ap.add_argument("--save-dir", default=None,
                    help="if set, write each fold model state_dict here for submission")
    args = ap.parse_args()

    device = "cuda"
    folds, X, y = load_data(args.cubes)
    print(f"n={len(X)}  cube={tuple(X.shape[2:])}  stem_stride={args.stem_stride}  "
          f"params={count_params(DatNet(stem_stride=args.stem_stride)):,}")
    print(f"flip-aug={'off' if args.no_flip else 'on'}  tta-flip={args.tta_flip}  "
          f"epochs={args.epochs}")

    per_seed, oof_cols = [], {}
    history: list[dict] = [] if args.history else None
    for seed in args.seeds:
        oof = np.full(len(X), np.nan)
        fold = folds[f"fold_s{seed}"].to_numpy()
        t0 = time.time()
        for k in range(N_SPLITS):
            va = fold == k
            p, model = train_fold(X[~va], y[~va], X[va], args, device, seed * 100 + k,
                                  yva=y[va], history=history, hist_key=f's{seed}f{k}')
            oof[va] = p
            if args.save_dir:
                sd = Path(args.save_dir); sd.mkdir(parents=True, exist_ok=True)
                # cpu tensors so the checkpoint loads with or without a GPU
                torch.save({k2: v.cpu() for k2, v in model.state_dict().items()},
                           sd / f'cnn_s{seed}_f{k}.pt')
            print(f"  seed {seed} fold {k}: n_va={va.sum():>4}  "
                  f"logloss {score(y.numpy()[va], p)['log_loss']:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        assert not np.isnan(oof).any()
        s = score(y.numpy(), oof)
        per_seed.append(s)
        oof_cols[f"oof_s{seed}"] = oof
        print(f"seed {seed}: log loss {s['log_loss']:.4f}  AUROC {s['auroc']:.4f}")

    mean_oof = np.mean(list(oof_cols.values()), axis=0)
    report(y.numpy(), mean_oof, folds.group.to_numpy(), title=f"{args.tag}: seed-averaged OOF")

    out = ROOT / "artifacts" / "oof" / f"{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"uid": folds.uid, "y": y.numpy(), **oof_cols, "oof_mean": mean_oof}).to_csv(
        out, index=False)
    print(f"\nOOF -> {out}")

    if history:
        hp = ROOT / "artifacts" / f"history_{args.tag}.csv"
        pd.DataFrame(history).to_csv(hp, index=False)
        print(f"per-epoch metrics -> {hp}")


if __name__ == "__main__":
    main()
