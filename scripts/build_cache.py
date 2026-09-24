"""Run datpark.preprocess over the training set and cache the result.

Writes a single float16 array (n, G, G, G) plus a uid index, so training can
memory-map it instead of decoding 1362 gzipped NIfTIs every epoch.

    python scripts/build_cache.py             # default config
    python scripts/build_cache.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datpark.preprocess import PreprocessConfig, preprocess  # noqa: E402

NIFTI = ROOT / "data" / "niftis"
OUT = ROOT / "artifacts" / "cache"


def _one(args):
    uid, cfg = args
    cube, meta = preprocess(NIFTI / f"{uid}.nii.gz", cfg)
    return uid, cube.astype(np.float16), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (len(__import__("os").sched_getaffinity(0)) - 2)))
    ap.add_argument("--voxel-mm", type=float, default=PreprocessConfig.voxel_mm)
    ap.add_argument("--crop-mm", type=float, default=PreprocessConfig.crop_mm)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    cfg = PreprocessConfig(voxel_mm=args.voxel_mm, crop_mm=args.crop_mm)
    g = cfg.grid
    uids = sorted(p.stem.removesuffix(".nii") for p in NIFTI.glob("*.nii.gz"))
    print(f"{len(uids)} scans -> {g}^3 @ {cfg.voxel_mm} mm ({cfg.crop_mm} mm box), "
          f"{args.workers} workers")

    OUT.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((len(uids), g, g, g), dtype=np.float16)
    metas = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (uid, cube, meta) in enumerate(
            ex.map(_one, [(u, cfg) for u in uids], chunksize=8), start=0
        ):
            arr[i] = cube
            metas.append({"uid": uid, **meta})
            if (i + 1) % 200 == 0 or i + 1 == len(uids):
                print(f"  {i + 1}/{len(uids)}", flush=True)

    npy = OUT / f"cubes_{args.tag}.npy"
    np.save(npy, arr)
    pd.DataFrame(metas).to_csv(OUT / f"meta_{args.tag}.csv", index=False)
    (OUT / f"config_{args.tag}.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    print(f"\nwrote {npy}  ({npy.stat().st_size / 1e6:.0f} MB)")
    print(f"      {OUT / f'meta_{args.tag}.csv'}")

    a = arr.astype(np.float32)
    print("\nsanity (values are multiples of the non-specific reference level):")
    print(f"  overall mean {a.mean():.2f}   p50 {np.percentile(a, 50):.2f}   "
          f"p99 {np.percentile(a, 99):.2f}   max {a.max():.2f}")
    centre = a[:, g // 4 : 3 * g // 4, g // 4 : 3 * g // 4, g // 4 : 3 * g // 4]
    print(f"  central half mean {centre.mean():.2f}  (should exceed the overall mean:"
          f" the striatum is centred)")
    peak = a.reshape(len(uids), -1).max(axis=1)
    print(f"  per-scan peak: p5 {np.percentile(peak, 5):.2f}  median {np.median(peak):.2f}"
          f"  p95 {np.percentile(peak, 95):.2f}")
    dead = int((peak < 1.5).sum())
    print(f"  scans with peak < 1.5x reference: {dead} "
          f"({'ok' if dead < 20 else 'INVESTIGATE - localisation may be failing'})")


if __name__ == "__main__":
    main()
