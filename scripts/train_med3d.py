"""Fine-tune a MedicalNet-pretrained 3D ResNet as a REPLACEMENT for sm0.

    python scripts/train_med3d.py --tag m3d_pre  --weights artifacts/pretrained/resnet_10_23dataset.pth
    python scripts/train_med3d.py --tag m3d_scr  --scratch      # the control

Additive: nothing here is imported by any existing experiment, and `load_data` /
`epoch_metrics` come from `train_cnn.py` verbatim so the data path is bit-identical to every
recorded result. Augmentation, loss, TTA and the fold loop mirror `train_cnn2.py`.

★ WHAT IS BEING TESTED, precisely: does a pretrained INITIALISATION beat a random one, with
architecture, schedule, augmentation, seeds and folds all held constant? Because the control
shares the architecture, a null here means "pretraining does not transfer", not "the model is
too big" -- the mistake an earlier analysis taught us to design out.

THE BAR, fixed before any number exists: beat `sm0`'s **0.2694** on `folds.csv` by **> 0.010**
with all three seeds agreeing. Below that it is selection bias (measured at +0.0030 to +0.0068
per composition decision on these same 1362 OOF predictions), and it must clear a real margin
because the public LB has SE 0.040 on ~145 scans.

⚠ SIZE, decided before running so it cannot become a sunk cost. One r10 checkpoint is 57 MB
against DatNet's ~6.5 MB, so 15 folds is ~860 MB versus 005's whole 200 MB ZIP. The container
has no network, so weights must ship inside it. If this wins, packaging needs a decision:
fewer seeds, fp16 weights, or r10 only. It is not a blocker, but do not discover it later.
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
from datpark.losses import get_loss  # noqa: E402
from datpark.augment3 import blur_noise  # noqa: E402
from datpark.med3d import build_med3d, set_train_mode, standardise_per_scan  # noqa: E402
from datpark.metrics import report, score  # noqa: E402
from datpark.nets import apply_affine, random_affine_grid  # noqa: E402
from datpark.nets2 import random_affine_grid_anat  # noqa: E402
from train_cnn import epoch_metrics, load_data  # noqa: E402  identical data path


def make_grid_fn(args):
    if args.aug == "orig":
        return lambda n, dev, gen: random_affine_grid(
            n, dev, rot_deg=args.rot, trans=args.trans, scale=args.scale,
            flip_lr=not args.no_flip, generator=gen)
    return lambda n, dev, gen: random_affine_grid_anat(
        n, dev, rot_deg=args.rot, trans=args.trans, scale=args.scale,
        flip_lr=not args.no_flip, generator=gen, rot_plane=args.rot_plane)


def train_fold(Xtr, ytr, Xva, args, device, seed, grid_fn, first=False):
    loss_fn = get_loss(args.loss)
    loss_kw = {"q": args.loss_q, "beta": args.loss_beta, "drop_frac": args.loss_drop}
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    model, info = build_med3d(depth=args.depth,
                             weights=None if args.scratch else args.weights,
                             dropout=args.dropout, maxpool=not args.no_maxpool,
                             freeze_until=args.freeze_until, verbose=first)
    model = model.to(device).to(memory_format=torch.channels_last_3d)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = max(1, (len(Xtr) + args.batch - 1) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.25)
    Xtr, ytr, Xva = Xtr.to(device), ytr.to(device), Xva.to(device)
    n = len(Xtr)

    for _ in range(args.epochs):
        set_train_mode(model)   # frozen submodules stay in eval — see med3d.set_train_mode
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
            xb = xb.to(memory_format=torch.channels_last_3d)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(xb)
                tgt = yb * (1 - args.smooth) + 0.5 * args.smooth
                loss = loss_fn(logit.float(), tgt, **loss_kw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

    model.eval()                     # BatchNorm in eval: required by rule 7 (determinism)
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(Xva), 32):
            xb = Xva[s : s + 32].float().to(memory_format=torch.channels_last_3d)
            p = torch.sigmoid(model(xb).float())
            if args.tta_flip:        # per-sample, so test-sample independence holds
                p = (p + torch.sigmoid(model(torch.flip(xb, dims=[2])).float())) / 2
            preds.append(p.cpu())
    # Reported per fold so an overnight log shows how close each arm ran to the 16 GB ceiling.
    if device == "cuda":
        info["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
    return torch.cat(preds).numpy(), model, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=10, choices=[10, 18, 34])
    ap.add_argument("--weights", default=None,
                    help="defaults to artifacts/pretrained/resnet_<depth>_23dataset.pth")
    ap.add_argument("--scratch", action="store_true",
                    help="random init, same architecture — THE CONTROL")
    ap.add_argument("--freeze-until", default="none",
                    choices=["none", "stem", "layer1", "layer2", "layer3", "layer4"])
    ap.add_argument("--no-maxpool", action="store_true")
    ap.add_argument("--deterministic", action="store_true",
                    help="bit-reproducible training. Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                         "in the environment, and FAILS LOUDLY on --maxpool because PyTorch "
                         "has no deterministic max_pool3d_with_indices_backward_cuda — that "
                         "op is the sole source of med3d's 0.0021 run-to-run drift (isolated "
                         "by experiment: without the maxpool, two runs are bit-identical).")
    ap.add_argument("--standardise", action="store_true",
                    help="z-score each scan (MedicalNet's own input form) — per-scan stats "
                         "only, so rule 6 still holds")
    # everything below matches the champion recipe (train_cnn2.py defaults for sm0)
    ap.add_argument("--cubes", default="cubes_hires.npy")
    ap.add_argument("--aug", default="orig", choices=["fixed", "orig"])
    ap.add_argument("--rot-plane", default="axial", choices=["axial", "sagittal"])
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
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)   # 14M params at 96^3 needs a smaller batch
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--loss", default="bce")
    ap.add_argument("--loss-q", type=float, default=0.7)
    ap.add_argument("--loss-beta", type=float, default=0.5)
    ap.add_argument("--loss-drop", type=float, default=0.05)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--n-splits", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="med3d")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--max-folds", type=int, default=None,
                    help="stop after N folds of the first seed — timing probe only, no OOF")
    args = ap.parse_args()

    # Each depth has its own checkpoint AND its own shortcut type (r10 uses B, r18/r34 use A);
    # datpark.med3d picks the shortcut from the depth, so only the path is resolved here.
    if args.weights is None:
        args.weights = f"artifacts/pretrained/resnet_{args.depth}_23dataset.pth"

    if not args.scratch and not (ROOT / args.weights).exists():
        sys.exit(f"missing weights: {args.weights}\n"
                 f"  fetch with: curl -sSL -o {args.weights} "
                 f"https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet"
                 f"{args.depth}/resolve/main/resnet_{args.depth}_23dataset.pth")

    if args.deterministic:
        # Order matters: these must be set before any kernel is selected or any model built.
        if not args.no_maxpool:
            sys.exit("--deterministic requires --no-maxpool: PyTorch has no deterministic "
                     "max_pool3d_with_indices_backward_cuda, so a maxpool run cannot be "
                     "bit-reproducible however the flags are set.")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            sys.exit("--deterministic requires CUBLAS_WORKSPACE_CONFIG=:4096:8 in the "
                     "environment (cuBLAS reduction order is otherwise unspecified).")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        print("  determinism: STRICT (cudnn.deterministic, use_deterministic_algorithms)",
              flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds, X, y = load_data(args.cubes)
    if args.standardise:
        X = standardise_per_scan(X)
        print(f"  input z-scored per scan: mean {X.float().mean():+.4f} "
              f"std {X.float().std():.4f}  (was reference-multiples, mean ~1)", flush=True)
    grid_fn = make_grid_fn(args)

    n_splits = args.n_splits or N_SPLITS
    if args.folds:
        alt = pd.read_csv(ROOT / "artifacts" / args.folds).set_index("uid")
        assert set(alt.index) == set(folds.uid), f"{args.folds} uids differ from folds.csv"
        alt = alt.reindex(folds.uid)
        assert (alt.y.to_numpy() == folds.y.to_numpy()).all(), "labels differ"
        for c in [c for c in alt.columns if c.startswith("fold_s")]:
            folds[c] = alt[c].to_numpy()
        folds["group"] = alt["group"].to_numpy()

    print(f"tag={args.tag}  depth=r{args.depth}  "
          f"{'SCRATCH (control)' if args.scratch else 'PRETRAINED ' + args.weights}  "
          f"lr={args.lr}  batch={args.batch}  epochs={args.epochs}  seeds={args.seeds}",
          flush=True)

    save = None
    if args.save_dir:
        save = ROOT / args.save_dir
        save.mkdir(parents=True, exist_ok=True)

    oof = np.zeros((len(folds), len(args.seeds)), dtype=np.float64)
    t_all = time.time()
    for si, seed in enumerate(args.seeds):
        col = f"fold_s{seed}"
        t0 = time.time()
        for k in range(n_splits):
            va = (folds[col] == k).to_numpy()
            if not va.any():
                sys.exit(f"fold {k} of {col} is empty")
            p, model, info = train_fold(X[~va], y[~va], X[va], args, device, seed, grid_fn,
                                        first=(si == 0 and k == 0))
            oof[va, si] = p
            print(f"  seed {seed} fold {k}: n_va={len(p):4d}  "
                  f"logloss {score(y.numpy()[va], p)['log_loss']:.4f}  "
                  f"({time.time() - t0:.0f}s"
                  f"{', peak %.1f GB' % info['peak_vram_gb'] if 'peak_vram_gb' in info else ''})",
                  flush=True)
            if save is not None:
                torch.save(model.state_dict(), save / f"med3d_s{seed}_f{k}.pt")
            if args.max_folds and k + 1 >= args.max_folds:
                print(f"\nTIMING PROBE: {args.max_folds} fold(s) in "
                      f"{time.time() - t0:.0f}s -> a 15-fold arm is about "
                      f"{(time.time() - t0) / args.max_folds * 15 / 60:.0f} min. "
                      f"No OOF written.", flush=True)
                return

    mean = oof.mean(axis=1)
    out = pd.DataFrame({"uid": folds.uid, "y": folds.y, "oof_mean": mean})
    for si, seed in enumerate(args.seeds):
        out[f"oof_s{seed}"] = oof[:, si]
    # Write via a temp file + atomic rename. The suite's resume logic treats an existing
    # artifacts/oof/<tag>.csv as "this arm is done", so a process killed midway through
    # to_csv() would leave a TRUNCATED file that the next run would silently accept and never
    # recompute. os.replace() is atomic on the same filesystem: the file either does not exist
    # or is complete.
    dest = ROOT / "artifacts" / "oof" / f"{args.tag}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, dest)

    print(f"\nwrote {dest.relative_to(ROOT)}   total {(time.time() - t_all) / 60:.1f} min")
    for si, seed in enumerate(args.seeds):
        print(f"  seed {seed}: {score(folds.y.to_numpy(), oof[:, si])['log_loss']:.4f}")
    print(report(folds.y.to_numpy(), mean))
    print(f"  reference: sm0 = 0.2694 on folds.csv; the bar is < 0.2594 with all seeds agreeing")
    epoch_metrics  # noqa: B018  imported to keep the data/metric path identical to train_cnn


if __name__ == "__main__":
    main()
