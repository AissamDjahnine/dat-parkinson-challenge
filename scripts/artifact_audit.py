"""Cohort audit for scans whose intensity scale is set by something other than the brain.

Two failure modes matter for both display and training:
  * edge/streak artefacts — the global maximum sits on the volume border;
  * high background — the brain is a weak signal over a bright floor.

Both break any normalisation that keys off whole-volume percentiles.
Aggregate output only.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BORDER = 3  # voxels from a face that counts as "on the edge"


def main() -> None:
    labels = pd.read_csv(ROOT / "data" / "train_labels.csv").set_index("uid").is_pathologic.to_dict()
    rows = []
    paths = sorted((ROOT / "data" / "niftis").glob("*.nii.gz"))
    for n, p in enumerate(paths, 1):
        uid = p.stem.removesuffix(".nii")
        d = np.asanyarray(nib.load(p).dataobj).astype(np.float32)
        shape = np.array(d.shape)
        mx = np.array(np.unravel_index(int(np.argmax(d)), d.shape))
        on_edge = bool((mx < BORDER).any() or (mx >= shape - BORDER).any())

        p50, p995, mxv = (float(np.percentile(d, 50)), float(np.percentile(d, 99.5)), float(d.max()))
        # how far the true max overshoots the 99.5th percentile
        overshoot = mxv / p995 if p995 > 0 else np.inf
        rows.append(
            {
                "uid": uid,
                "label": labels.get(uid),
                "max_on_edge": on_edge,
                "bg_ratio": p50 / p995 if p995 > 0 else np.nan,
                "overshoot": overshoot,
                "p50": p50,
                "p995": p995,
                "vmax": mxv,
                "nonzero_frac": float((d > 0).mean()),
            }
        )
        if n % 200 == 0:
            print(f"  …{n}/{len(paths)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "artifacts" / "artifact_audit.csv", index=False)

    print(f"\nscans audited: {len(df)}")
    print(f"global max sits on the volume border : {df.max_on_edge.sum()} ({df.max_on_edge.mean():.1%})")
    print(f"  of those, abnormal                 : {df[df.max_on_edge].label.mean():.1%} "
          f"(cohort base rate {df.label.mean():.1%})")

    print("\nbackground ratio  p50 / p99.5:")
    for q in (0.5, 0.9, 0.95, 0.99, 1.0):
        print(f"  q{q:<5} {df.bg_ratio.quantile(q):.3f}")
    print(f"  scans with bg_ratio > 0.20 : {(df.bg_ratio > 0.20).sum()} ({(df.bg_ratio > 0.20).mean():.1%})")

    print("\nmax / p99.5 overshoot (how far the peak exceeds the clip point):")
    for q in (0.5, 0.9, 0.99, 1.0):
        print(f"  q{q:<5} {df.overshoot.quantile(q):.2f}")
    print(f"  scans with overshoot > 2.0 : {(df.overshoot > 2.0).sum()} ({(df.overshoot > 2.0).mean():.1%})")

    flagged = df[df.max_on_edge | (df.bg_ratio > 0.20)]
    print(f"\nflagged either way: {len(flagged)} ({len(flagged)/len(df):.1%}) "
          f"-> artifacts/artifact_audit.csv")


if __name__ == "__main__":
    main()
