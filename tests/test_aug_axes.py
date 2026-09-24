"""Regression guard for the augmentation axis bug found 2026-07-28.

`nets.random_affine_grid` documented "flip_lr mirrors the Right-Left axis" and
"rotation in the axial plane" but did neither: `affine_grid` on 5D input orders
grid coords (x, y, z) with x indexing the LAST spatial dim, so theta row 0 drives
S-I, not R-L. The original therefore mirrored the brain upside-down and rotated
sagittally. `nets2.random_affine_grid_anat` fixes the mapping.

These tests assert on *measured* voxel displacement, never on doc-derived
reasoning -- reasoning from the docs is exactly how the bug survived.

Cubes are RAS with no reorientation (`preprocess.load_volume`), so
array axis 0 = R-L, axis 1 = A-P, axis 2 = S-I == tensor dims 2, 3, 4.
"""

from __future__ import annotations

import numpy as np
import torch

from datpark.nets import apply_affine, random_affine_grid
from datpark.nets2 import ROW_RL, SiameseDatNet, random_affine_grid_anat

G = 16
OFF = 3          # mirrors to 12 about the 7.5 centre
RL, AP, SI = 0, 1, 2


def _marker(axis: int) -> torch.Tensor:
    x = torch.zeros(1, 1, G, G, G)
    idx = [G // 2] * 3
    idx[axis] = OFF
    x[0, 0, idx[0], idx[1], idx[2]] = 1.0
    return x


def _peak(t: torch.Tensor) -> np.ndarray:
    return np.array(np.unravel_index(int(t.argmax()), (G, G, G)))


def _moved(before: torch.Tensor, after: torch.Tensor) -> np.ndarray:
    """Per-axis displacement > 1 voxel. align_corners=False costs a half voxel."""
    return np.abs(_peak(before) - _peak(after)) > 1


def _identity_theta() -> torch.Tensor:
    t = torch.zeros(1, 3, 4)
    t[:, 0, 0] = t[:, 1, 1] = t[:, 2, 2] = 1.0
    return t


def _mirrored_axes(row: int) -> set[int]:
    """Which anatomical axes a pure flip written into `theta[row]` mirrors."""
    hit = set()
    for axis in (RL, AP, SI):
        t = _identity_theta()
        t[:, row, :] *= -1
        x = _marker(axis)
        if _moved(x, apply_affine(x, t))[axis]:
            hit.add(axis)
    return hit


# --------------------------------------------------------------------- the bug
def test_original_flip_mirrors_si_not_rl():
    """Documents the defect so nobody 'fixes' nets2 back to the broken mapping."""
    assert _mirrored_axes(0) == {SI}, "original flip should mirror S-I"


def test_fixed_flip_mirrors_rl_only():
    assert _mirrored_axes(ROW_RL) == {RL}, "corrected flip must mirror R-L alone"


def test_flip_lr_flag_actually_mirrors_rl():
    """End-to-end through the public helper, not a hand-built theta."""
    torch.manual_seed(0)
    for axis in (RL, AP, SI):
        x = _marker(axis)
        # force the mirror branch: sample until we get a negative R-L scale
        for seed in range(50):
            g = torch.Generator().manual_seed(seed)
            t = random_affine_grid_anat(1, "cpu", rot_deg=0.0, trans=0.0, scale=0.0,
                                        flip_lr=True, generator=g)
            if t[0, ROW_RL, ROW_RL] < 0:
                break
        else:
            raise AssertionError("never drew a mirrored sample in 50 tries")
        moved = _moved(x, apply_affine(x, t))
        assert moved[axis] == (axis == RL), (
            f"axis {axis} moved={moved[axis]} under an R-L mirror")


def test_mirror_tta_axis_was_always_correct():
    """torch.flip(x, dims=[2]) is the shipped TTA; it must hit R-L."""
    for axis in (RL, AP, SI):
        x = _marker(axis)
        assert _moved(x, torch.flip(x, dims=[2]))[axis] == (axis == RL)


def test_rotation_plane():
    """The axis whose marker stays on its own axis is the rotation axis."""
    def rot90(a: int, b: int) -> torch.Tensor:
        t = _identity_theta()
        t[:, a, a] = 0.0; t[:, a, b] = -1.0
        t[:, b, a] = 1.0; t[:, b, b] = 0.0
        return t

    # original wrote rotation into rows 0,1 -> rotates about R-L (sagittal)
    assert not _moved(_marker(RL), apply_affine(_marker(RL), rot90(0, 1)))[RL]
    # corrected axial rotation is about S-I
    assert not _moved(_marker(SI), apply_affine(_marker(SI), rot90(ROW_RL, 1)))[SI]


# ------------------------------------------------- the fix changes only the axes
def test_fix_is_a_pure_axis_permutation():
    """Same augmentation strength, different axes -- so an A/B is not confounded."""
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = random_affine_grid(4096, "cpu", flip_lr=True, generator=g1)
    b = random_affine_grid_anat(4096, "cpu", flip_lr=True, generator=g2)
    det_a = a[:, :, :3].det().abs()
    det_b = b[:, :, :3].det().abs()
    assert abs(float(det_a.mean()) - float(det_b.mean())) < 1e-5
    assert abs(float(a[:, :, 3].std()) - float(b[:, :, 3].std())) < 1e-5


def test_no_flip_leaves_orientation_alone():
    g = torch.Generator().manual_seed(0)
    t = random_affine_grid_anat(512, "cpu", flip_lr=False, generator=g)
    assert (t[:, ROW_RL, ROW_RL] > 0).all(), "flip_lr=False must never mirror"


def test_flip_is_roughly_balanced():
    g = torch.Generator().manual_seed(0)
    t = random_affine_grid_anat(4000, "cpu", flip_lr=True, generator=g)
    frac = float((t[:, ROW_RL, ROW_RL] < 0).float().mean())
    assert 0.45 < frac < 0.55, f"mirror fraction {frac:.3f} should be ~0.5"


def test_shapes_and_finiteness():
    for fn in (random_affine_grid, random_affine_grid_anat):
        t = fn(8, "cpu", flip_lr=True)
        assert t.shape == (8, 3, 4)
        assert torch.isfinite(t).all()


# --------------------------------------------------------- siamese invariance
def test_siamese_is_exactly_lr_invariant():
    """Invariance is structural, so it must hold to floating-point exactness."""
    torch.manual_seed(0)
    net = SiameseDatNet(stem_stride=3).eval()
    x = torch.randn(4, 1, 24, 24, 24)
    with torch.no_grad():
        gap = float((net(x) - net(torch.flip(x, dims=[2]))).abs().max())
    assert gap < 1e-4, f"siamese not L<->R invariant: max gap {gap:.2e}"


def test_siamese_is_not_invariant_to_other_axes():
    """A net invariant to everything would just be ignoring its input."""
    torch.manual_seed(0)
    net = SiameseDatNet(stem_stride=3).eval()
    x = torch.randn(4, 1, 24, 24, 24)
    with torch.no_grad():
        base = net(x)
        ap = float((base - net(torch.flip(x, dims=[3]))).abs().max())
        si = float((base - net(torch.flip(x, dims=[4]))).abs().max())
    assert ap > 1e-4 and si > 1e-4, "siamese should only be invariant on R-L"
