"""Contracts for the 011 acquisition-physics augmentation (blur + correlated noise).

The 009 erasing arm was designed, unit tested and still wrong — its tests asserted the cuboid
avoided a sphere but never that the sphere contained the striatum. So these tests check the
properties the EXPERIMENT'S INTERPRETATION depends on, not just that the functions run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datpark.augment3 import FWHM_TO_SIGMA, blur3d, blur_noise  # noqa: E402


def test_blur_achieves_the_requested_sigma_and_conserves_mass():
    """A blurred delta must have the sigma we asked for, and blur must not change total signal.

    If the sigma were wrong the whole M grid would be mislabelled and the dose figures in the
    runner header would be fiction.
    """
    d = torch.zeros(1, 1, 64, 64, 64)
    d[0, 0, 32, 32, 32] = 1.0
    idx = np.arange(64) - 32
    for s in (1.0, 2.0, 4.78):
        b = blur3d(d, torch.tensor([s]))
        prof = b[0, 0, :, 32, 32].numpy()
        measured = float(np.sqrt((prof * idx**2).sum() / prof.sum()))
        assert abs(measured - s) < 0.05 * s, f"asked sigma {s}, measured {measured}"
        assert abs(float(b.sum()) - 1.0) < 1e-4, "blur must conserve mass"


def test_blur_applies_a_different_sigma_per_sample():
    """Sigmas are drawn per sample; the grouped-conv trick must not collapse them to one.

    A silent collapse would still produce plausible images, so this is asserted rather than eyeballed.
    """
    x = torch.zeros(2, 1, 32, 32, 32)
    x[:, 0, 16, 16, 16] = 1.0
    b = blur3d(x, torch.tensor([0.5, 4.0]))
    peak0, peak1 = float(b[0, 0, 16, 16, 16]), float(b[1, 0, 16, 16, 16])
    assert peak0 > 5 * peak1, (
        f"sample 0 (sigma 0.5) should stay far more peaked than sample 1 (sigma 4.0); "
        f"got {peak0:.4f} vs {peak1:.4f} — the per-sample sigma was probably collapsed"
    )


def test_fwhm_to_sigma_conversion():
    assert abs(FWHM_TO_SIGMA - 1.0 / 2.3548200450309493) < 1e-12
    # the runner header quotes sigma 4.78 voxels for FWHM 15 mm at 1.3333 mm — keep that honest
    assert abs(15.0 * FWHM_TO_SIGMA / 1.3333 - 4.78) < 0.01


def test_disabling_switches_are_exact_no_ops():
    """M=0, P=0, or both components off must return the input untouched, bit for bit.

    This is what makes the control arm the champion rather than an approximation of it.
    """
    x = torch.rand(3, 1, 16, 16, 16)
    for kw in ({"M": 0.0, "P": 0.9}, {"M": 2.5, "P": 0.0},
               {"M": 2.5, "P": 0.9, "blur": False, "noise": False}):
        out = blur_noise(x, generator=torch.Generator().manual_seed(0), **kw)
        assert torch.equal(out, x), f"{kw} was not an exact no-op"


def test_components_can_be_isolated():
    """blur-only and noise-only must differ from each other and from both-on.

    Otherwise the attribution arms bn_b25 / bn_n25 measure nothing.
    """
    x = torch.rand(2, 1, 24, 24, 24)
    def run(**kw):
        return blur_noise(x, M=2.5, P=1.0, generator=torch.Generator().manual_seed(1), **kw)
    both, only_b, only_n = run(), run(noise=False), run(blur=False)
    assert not torch.allclose(both, only_b)
    assert not torch.allclose(both, only_n)
    assert not torch.allclose(only_b, only_n)
    # blur alone cannot increase the maximum; noise alone can
    assert float(only_b.max()) <= float(x.max()) + 1e-5


def test_noise_dose_is_relative_to_each_sample_and_scales_with_M():
    """The noise SD is a FRACTION of the sample's own intensity SD, so the dose must scale.

    An absolute dose would hit the 16,000x between-centre intensity range in this dataset very
    unevenly — a scan from a low-count site would be obliterated while a bright one barely moved.
    """
    x = torch.rand(2, 1, 24, 24, 24)
    x[1] *= 10.0                                  # one sample 10x brighter
    out = blur_noise(x, M=2.5, P=1.0, blur=False,
                     generator=torch.Generator().manual_seed(2))
    added = (out - x).flatten(1).std(dim=1) / x.flatten(1).std(dim=1)
    assert abs(float(added[0]) - float(added[1])) < 0.35 * float(added.mean()), (
        f"relative noise dose differs across samples of different brightness: {added.tolist()}"
    )
    lo = (blur_noise(x, M=1.0, P=1.0, blur=False,
                     generator=torch.Generator().manual_seed(2)) - x).std()
    hi = (blur_noise(x, M=4.0, P=1.0, blur=False,
                     generator=torch.Generator().manual_seed(2)) - x).std()
    assert hi > 2.0 * lo, f"noise dose should scale with M; got {float(lo):.4f} -> {float(hi):.4f}"


def test_augmentation_preserves_the_striatal_peak_enough_to_learn_from():
    """The point is to corrupt IRRELEVANT features while leaving the label signal learnable.

    Measured on real scans, the retained striatal peak is 98 % at M=1.0, 91 % at M=2.5 and 86 % at
    M=4.0. This synthetic version guards the property rather than the exact number: a blob on a
    background must remain clearly the brightest structure after augmentation, or the arm is
    training on images whose label is no longer visible — the failure that killed 009's erasing.
    """
    x = torch.rand(1, 1, 48, 48, 48) * 0.05
    zz, yy, xx = torch.meshgrid(*[torch.arange(48)] * 3, indexing="ij")
    blob = torch.exp(-(((zz - 24.0) ** 2 + (yy - 24.0) ** 2 + (xx - 24.0) ** 2) / 18.0))
    x[0, 0] += blob
    out = blur_noise(x, M=2.5, P=1.0, generator=torch.Generator().manual_seed(3))
    centre = float(out[0, 0, 20:28, 20:28, 20:28].mean())
    edge = float(out[0, 0, :6, :6, :6].mean())
    assert centre > 3.0 * edge, (
        f"the blob is no longer clearly brightest after augmentation (centre {centre:.4f} vs "
        f"edge {edge:.4f}) — the label signal would not survive"
    )

