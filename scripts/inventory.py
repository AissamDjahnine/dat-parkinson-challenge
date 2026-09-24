"""Header-only inventory of the DaT training set.

Reads NIfTI headers (never voxel data) and prints cohort-level aggregates only.
No per-examination values are emitted -- see the project data-handling rule.
"""

import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def scan_headers(nifti_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(nifti_dir.glob("*.nii.gz")):
        img = nib.load(path)  # lazy: header only, no voxel access
        hdr = img.header
        shape = tuple(int(d) for d in img.shape)
        zooms = tuple(round(float(z), 4) for z in hdr.get_zooms()[:3])
        rows.append(
            {
                "uid": path.stem.removesuffix(".nii"),
                "shape": shape,
                "zooms": zooms,
                "dtype": str(hdr.get_data_dtype()),
                "axcodes": "".join(nib.aff2axcodes(img.affine)),
                "qform_code": int(hdr["qform_code"]),
                "sform_code": int(hdr["sform_code"]),
                "scl_slope": float(hdr["scl_slope"]) if np.isfinite(hdr["scl_slope"]) else np.nan,
                "scl_inter": float(hdr["scl_inter"]) if np.isfinite(hdr["scl_inter"]) else np.nan,
                "fov_mm": tuple(round(s * z, 1) for s, z in zip(shape, zooms)),
                "n_voxels": int(np.prod(shape)),
                "file_mb": round(path.stat().st_size / 1e6, 2),
            }
        )
    return pd.DataFrame(rows)


def show(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def counter_table(series: pd.Series, label: str, top: int | None = None) -> None:
    counts = Counter(series)
    items = counts.most_common(top)
    total = len(series)
    print(f"{label:<34} {'count':>7} {'share':>8}")
    print("-" * 52)
    for value, n in items:
        print(f"{str(value):<34} {n:>7} {n / total:>7.1%}")
    if top is not None and len(counts) > top:
        print(f"{'... (' + str(len(counts) - top) + ' more)':<34}")
    print(f"{'TOTAL distinct':<34} {len(counts):>7}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--niftis", type=Path, default=Path("data/niftis"))
    ap.add_argument("--labels", type=Path, default=Path("data/train_labels.csv"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/header_inventory.csv"))
    args = ap.parse_args()

    df = scan_headers(args.niftis)
    labels = pd.read_csv(args.labels)

    show("COHORT")
    print(f"NIfTI files            : {len(df)}")
    print(f"Label rows             : {len(labels)}")
    print(f"uids in both           : {len(set(df.uid) & set(labels.uid))}")
    print(f"files without a label  : {len(set(df.uid) - set(labels.uid))}")
    print(f"labels without a file  : {len(set(labels.uid) - set(df.uid))}")
    print(f"duplicate label uids   : {int(labels.uid.duplicated().sum())}")
    print(f"total on disk          : {df.file_mb.sum() / 1000:.2f} GB")

    show("CLASS BALANCE")
    vc = labels.is_pathologic.value_counts(dropna=False).sort_index()
    for value, n in vc.items():
        print(f"  is_pathologic = {value!s:<6} {n:>6}  {n / len(labels):>6.1%}")
    print(f"  distinct label values: {sorted(labels.is_pathologic.dropna().unique())}")
    print(f"  NaN labels           : {int(labels.is_pathologic.isna().sum())}")

    show("VOLUME SHAPE")
    counter_table(df["shape"], "shape (i,j,k)")

    show("VOXEL SPACING (mm)")
    counter_table(df["zooms"], "zooms (mm)")

    show("PHYSICAL FIELD OF VIEW (mm)")
    counter_table(df["fov_mm"], "shape x spacing", top=15)

    show("VOXEL DTYPE")
    counter_table(df["dtype"], "dtype")

    show("ORIENTATION (affine axis codes)")
    counter_table(df["axcodes"], "axcodes")

    show("AFFINE CODES / INTENSITY SCALING")
    counter_table(df["qform_code"].astype(str) + "/" + df["sform_code"].astype(str), "qform/sform")
    print()
    print(f"scl_slope distinct : {sorted(df.scl_slope.dropna().unique())[:10]}")
    print(f"scl_inter distinct : {sorted(df.scl_inter.dropna().unique())[:10]}")

    show("ACQUISITION SIGNATURE  (shape + spacing + dtype)  -- proxy for centre/scanner")
    sig = df["shape"].astype(str) + " @ " + df["zooms"].astype(str) + " " + df["dtype"]
    counter_table(sig, "signature", top=20)

    show("CLASS BALANCE PER ACQUISITION SIGNATURE")
    merged = df.assign(sig=sig).merge(labels, on="uid", how="inner")
    per_sig = (
        merged.groupby("sig")
        .agg(n=("uid", "size"), pathologic_rate=("is_pathologic", "mean"))
        .sort_values("n", ascending=False)
    )
    print(f"{'signature':<44} {'n':>5} {'path.rate':>10}")
    print("-" * 62)
    for s, row in per_sig.head(20).iterrows():
        print(f"{s:<44} {int(row.n):>5} {row.pathologic_rate:>9.1%}")
    if len(per_sig) > 20:
        print(f"... ({len(per_sig) - 20} more signatures)")
    print(f"\ndistinct signatures: {len(per_sig)}")
    print(f"signatures with n < 10: {int((per_sig.n < 10).sum())}")
    print(f"pathologic rate spread: {per_sig.pathologic_rate.min():.1%} .. {per_sig.pathologic_rate.max():.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.assign(sig=sig).to_csv(args.out, index=False)
    print(f"\nper-file header table written to {args.out} (stays on disk)")


if __name__ == "__main__":
    main()
