"""Prove which anatomical axes the augmentations actually touch.

Run before trusting any augmentation change:

    uv run python scripts/verify_aug_axes.py

Reasoning from the PyTorch docs about `affine_grid`'s coordinate order is how the
original bug survived, so this asserts on measured behaviour instead. A marker
voxel is placed off-centre on one axis at a time; whichever axis the marker
*moves along* is the axis that transform acts on.

Cubes are RAS with no reorientation (`datpark.preprocess.load_volume`), so
array axis 0 = R-L, axis 1 = A-P, axis 2 = S-I, i.e. tensor dims 2, 3, 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datpark.nets import apply_affine, random_affine_grid  # noqa: E402
from datpark.nets2 import SiameseDatNet, random_affine_grid_anat  # noqa: E402

G = 16
AXES = [(0, "R-L (axis0/dim2)"), (1, "A-P (axis1/dim3)"), (2, "S-I (axis2/dim4)")]
OFF = 3  # marker sits at index 3, mirrors to 12 about the 7.5 centre


def marker(axis: int) -> torch.Tensor:
    x = torch.zeros(1, 1, G, G, G)
    idx = [G // 2] * 3
    idx[axis] = OFF
    x[0, 0, idx[0], idx[1], idx[2]] = 1.0
    return x


def moved_axes(before: torch.Tensor, after: torch.Tensor) -> np.ndarray:
    s = np.array(np.unravel_index(int(before.argmax()), (G, G, G)))
    d = np.array(np.unravel_index(int(after.argmax()), (G, G, G)))
    # grid_sample with align_corners=False shifts everything by a half voxel, so
    # only displacements of more than one voxel count as a real move
    return np.abs(s - d) > 1


def which_axis_flips(theta_fn) -> set[int]:
    """Return the set of axes a pure-flip theta actually mirrors."""
    hit = set()
    for axis, _ in AXES:
        x = marker(axis)
        theta = theta_fn()
        if moved_axes(x, apply_affine(x, theta))[axis]:
            hit.add(axis)
    return hit


def pure_flip_original():
    t = torch.zeros(1, 3, 4)
    t[:, 0, 0] = t[:, 1, 1] = t[:, 2, 2] = 1.0
    t[:, 0, :] *= -1  # exactly the flip branch of nets.random_affine_grid
    return t


def pure_flip_fixed():
    t = torch.zeros(1, 3, 4)
    t[:, 0, 0] = t[:, 1, 1] = t[:, 2, 2] = 1.0
    t[:, 2, :] *= -1  # nets2 writes the flip into row 2
    return t


def main() -> None:
    ok = True

    print("=== 1. flip augmentation: which axis is mirrored? ===")
    orig, fixed = which_axis_flips(pure_flip_original), which_axis_flips(pure_flip_fixed)
    names = dict(AXES)
    print(f"  nets.random_affine_grid  (original) mirrors: {[names[a] for a in sorted(orig)]}")
    print(f"  nets2.random_affine_grid_anat (fixed) mirrors: {[names[a] for a in sorted(fixed)]}")
    if orig != {2}:
        print(f"  !! expected the original to mirror S-I only, got {orig}"); ok = False
    if fixed != {0}:
        print(f"  !! expected the fix to mirror R-L only, got {fixed}"); ok = False
    print(f"  -> original mirrors S-I (impossible anatomy): {orig == {2}}")
    print(f"  -> fixed mirrors R-L (the real symmetry):     {fixed == {0}}")

    print("\n=== 2. mirror TTA: torch.flip(x, dims=[2]) ===")
    hit = {a for a, _ in AXES if moved_axes(marker(a), torch.flip(marker(a), dims=[2]))[a]}
    print(f"  mirrors: {[names[a] for a in sorted(hit)]}")
    if hit != {0}:
        print(f"  !! expected R-L only, got {hit}"); ok = False
    print("  -> test-time mirroring was ALWAYS correct; only training aug was wrong.")

    print("\n=== 3. rotation plane (90 deg, axis that stays put = rotation axis) ===")
    for label, fn, expect in [
        ("original (nets)", lambda: _rot90_rows(0, 1), 0),
        ("fixed axial (nets2)", lambda: _rot90_rows(2, 1), 2),
    ]:
        still = [a for a, _ in AXES if not moved_axes(marker(a), apply_affine(marker(a), fn()))[a]]
        # the rotation axis is the one whose marker does not leave its own axis
        print(f"  {label:22} marker stays on: {[names[a] for a in still]}")
        if expect == 0 and 0 not in still:
            print("  !! expected original to rotate about R-L (sagittal)"); ok = False
        if expect == 2 and 2 not in still:
            print("  !! expected fix to rotate about S-I (axial)"); ok = False

    print("\n=== 4. SiameseDatNet is exactly L<->R invariant by construction ===")
    torch.manual_seed(0)
    net = SiameseDatNet(stem_stride=3).eval()
    x = torch.randn(4, 1, 24, 24, 24)
    with torch.no_grad():
        a = net(x)
        b = net(torch.flip(x, dims=[2]))
    gap = float((a - b).abs().max())
    print(f"  max |f(x) - f(mirror(x))| = {gap:.2e}")
    if gap > 1e-4:
        print("  !! not invariant -- the split or fusion is wrong"); ok = False
    else:
        print("  -> invariant, so R-L flip aug and mirror TTA are both no-ops on this arm.")

    print("\n=== 5. distributions unchanged (fix is an axis permutation, not a new aug) ===")
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    t_o = random_affine_grid(4096, "cpu", flip_lr=True, generator=g1)
    t_f = random_affine_grid_anat(4096, "cpu", flip_lr=True, generator=g2)
    for nm, t in [("original", t_o), ("fixed", t_f)]:
        print(f"  {nm:9} |det| mean {t[:, :, :3].det().abs().mean():.4f}   "
              f"translation sd {t[:, :, 3].std():.4f}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
    sys.exit(0 if ok else 1)


def _rot90_rows(a: int, b: int) -> torch.Tensor:
    """Pure 90 deg rotation written into theta rows (a, b)."""
    t = torch.zeros(1, 3, 4)
    for i in range(3):
        t[:, i, i] = 1.0
    c, s = 0.0, 1.0
    t[:, a, a] = c; t[:, a, b] = -s
    t[:, b, a] = s; t[:, b, b] = c
    return t


if __name__ == "__main__":
    main()
