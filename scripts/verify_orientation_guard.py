"""012 — prove the orientation guard leaves valid current inputs BIT-IDENTICAL. CPU, read-only.

    python scripts/verify_orientation_guard.py [--n 80]

The guard added to `submission_src_v8/datpark/preprocess.py::load_volume` is only admissible if it
changes nothing on data like ours. That is an empirical claim about 1362 real files, not a property to be
asserted from the nibabel docs, so this measures it three ways:

  A. every training scan is RAS (from the stored header inventory — no file reads)
  B. on a sample of REAL scans, load_volume with and without the guard returns bit-identical arrays
     AND bit-identical zooms
  C. on a SYNTHETIC non-RAS volume the guard actually fires and corrects the left-right axis —
     otherwise it is dead code and buys nothing

Test C matters as much as B: a guard that is a no-op on everything is not protection, it is decoration.

Memory: scans are read ONE AT A TIME and released. Two system-RAM OOMs on 2026-08-04 came from CPU-side
probes holding whole caches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NIFTI = ROOT / "data" / "niftis"


def load_plain(path):
    """load_volume WITHOUT the guard — byte-for-byte the shipped 011 logic."""
    import nibabel as nib
    img = nib.load(str(path))
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    zooms = np.where(np.isfinite(zooms) & (zooms > 0), zooms, 1.0)
    zooms = np.clip(zooms, 0.5, 20.0)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3])
    return data, zooms


def load_guarded(path):
    """load_volume WITH the guard — the 012 logic."""
    import nibabel as nib
    img = nib.load(str(path))
    try:
        img = nib.as_closest_canonical(img)
    except Exception:  # noqa: BLE001
        pass
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    zooms = np.where(np.isfinite(zooms) & (zooms > 0), zooms, 1.0)
    zooms = np.clip(zooms, 0.5, 20.0)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3])
    return data, zooms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="real scans to compare (0 = all 1362)")
    args = ap.parse_args()
    ok = True

    # ---- A ----------------------------------------------------------------------------------
    print("A. orientation of the training set (from artifacts/eda_table.csv, no file reads)")
    eda = pd.read_csv(ROOT / "artifacts" / "eda_table.csv")
    vc = eda.axcodes.value_counts()
    for k, v in vc.items():
        print(f"     {k}: {v}")
    if not (len(vc) == 1 and vc.index[0] == "RAS"):
        print("   !! not uniformly RAS — the guard is NOT a no-op on this dataset; STOP")
        ok = False
    else:
        print(f"   -> {vc.iloc[0]}/{len(eda)} RAS. The guard must therefore be an exact identity here.")

    # ---- B ----------------------------------------------------------------------------------
    folds = pd.read_csv(ROOT / "artifacts" / "folds.csv")
    uids = folds.uid.tolist()
    if args.n and args.n < len(uids):
        step = len(uids) // args.n
        uids = uids[::step][:args.n]        # deterministic spread across the sorted uid order
    print(f"\nB. bit-equality of load_volume with vs without the guard, on {len(uids)} REAL scans")
    bad = []
    for i, uid in enumerate(uids):
        p = NIFTI / f"{uid}.nii.gz"
        if not p.exists():
            print(f"   !! {p} missing"); ok = False; break
        a, za = load_plain(p)
        b, zb = load_guarded(p)
        same = a.shape == b.shape and np.array_equal(a, b) and np.array_equal(za, zb)
        if not same:
            bad.append(uid)
        del a, b
        if (i + 1) % 20 == 0:
            print(f"     {i + 1}/{len(uids)} checked", flush=True)
    if bad:
        print(f"   !! {len(bad)} scans CHANGED, e.g. {bad[:5]} — the guard is not a no-op; DO NOT SHIP")
        ok = False
    else:
        print(f"   -> {len(uids)}/{len(uids)} bit-identical, arrays AND zooms. Guard is an exact no-op.")

    # ---- C ----------------------------------------------------------------------------------
    print("\nC. does the guard actually FIRE on non-RAS input? (a dead guard is decoration)")
    import tempfile

    import nibabel as nib

    # ⚠ AN EARLIER VERSION OF THIS TEST WAS WRONG and reported the guard as broken. It built the LAS
    # file as the SAME array with a negated x-affine — which describes DIFFERENT anatomy (the marker
    # ends up on the subject's left), and then compared it against the RAS file's marker index. To
    # represent the SAME anatomy in LAS the array must ALSO be mirrored and the translation adjusted:
    #
    #     B[j] = A[n-1-j],  and  world_B(j) must equal world_A(n-1-j)
    #     -s*j + t' = s*(n-1-j) + t   =>   t' = s*(n-1) + t
    #
    # Both images then describe identical world content, and `as_closest_canonical` must recover A.
    # The construction is verified below by sampling the same WORLD point through both affines.
    n, s = 24, 2.0
    A = np.zeros((n, n, n), dtype=np.float32)
    A[18:22, :, :] = 5.0                             # marker at HIGH index = subject's RIGHT in RAS
    ras = np.diag([s, s, s, 1.0])
    B = A[::-1].copy()
    las = np.diag([-s, s, s, 1.0])
    las[0, 3] = s * (n - 1)

    with tempfile.TemporaryDirectory() as td:
        pr, pl = Path(td) / "ras.nii.gz", Path(td) / "las.nii.gz"
        nib.save(nib.Nifti1Image(A, ras), str(pr))
        nib.save(nib.Nifti1Image(B, las), str(pl))
        print(f"     axcodes ras.nii.gz = {nib.aff2axcodes(nib.load(str(pr)).affine)}")
        print(f"     axcodes las.nii.gz = {nib.aff2axcodes(nib.load(str(pl)).affine)}")

        # the two files must describe the SAME anatomy, or test C proves nothing
        inv_r = np.linalg.inv(ras)
        inv_l = np.linalg.inv(las)
        agree = True
        for wx in (2.0, 12.0, 24.0, 38.0, 44.0):
            w = np.array([wx, 10.0, 10.0, 1.0])
            ir = np.rint(inv_r @ w).astype(int)[:3]
            il = np.rint(inv_l @ w).astype(int)[:3]
            if not (0 <= ir[0] < n and 0 <= il[0] < n):
                continue
            if A[tuple(ir)] != B[tuple(il)]:
                agree = False
        if not agree:
            print("   !! the two fixtures do NOT describe the same anatomy — test C is invalid")
            ok = False
        else:
            print("     fixtures verified: both files describe the same world content")

        r_plain, _ = load_plain(pr)
        r_guard, _ = load_guarded(pr)
        if not np.array_equal(r_plain, r_guard):
            print("   !! guard altered an already-RAS file"); ok = False
        else:
            print("     RAS file: guard is an identity (as required)")

        l_plain, _ = load_plain(pl)
        l_guard, _ = load_guarded(pl)
        print(f"     LAS file: marker at x={int(l_plain.mean(axis=(1, 2)).argmax())} unguarded, "
              f"x={int(l_guard.mean(axis=(1, 2)).argmax())} guarded; RAS truth x="
              f"{int(r_guard.mean(axis=(1, 2)).argmax())}")
        if np.array_equal(l_plain, l_guard):
            print("   !! guard did NOT fire on an LAS volume — it is dead code"); ok = False
        elif not np.array_equal(l_guard, r_guard):
            print("   !! guard fired but did not recover the RAS array"); ok = False
        else:
            print("     -> guard RECOVERED the RAS array exactly from an LAS file. Not dead code.")

    print()
    print("=" * 78)
    print(f"  ORIENTATION GUARD {'ADMISSIBLE' if ok else 'NOT ADMISSIBLE'} for 012")
    print("=" * 78)
    if ok:
        print("  no-op on every valid current input, and corrective on input unlike ours.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
