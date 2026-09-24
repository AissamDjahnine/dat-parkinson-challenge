"""Architecture contracts: shapes, determinism, capacity parity, no NaNs.

The shipped checkpoints load against `DatNet(stem_stride=2)`, so the frozen-default
test below is a guard on submission reproducibility, not a style check.
"""

from __future__ import annotations

import torch

from datpark.nets import DatNet, count_params
from datpark.nets2 import SiameseDatNet, build


def _fwd(net, n=2, g=24):
    net = net.eval()
    with torch.no_grad():
        return net(torch.randn(n, 1, g, g, g))


def test_datnet_default_is_frozen():
    """15 shipped checkpoints load against stride 2 / kernel 5. Do not change."""
    net = DatNet()
    assert net.stem[0].stride == (2, 2, 2)
    assert net.stem[0].kernel_size == (5, 5, 5)
    assert count_params(net) == 1_629_833, (
        "DatNet param count changed -- shipped checkpoints will not load")


def test_stem_kernel_tracks_stride():
    """k = 2s+1 avoids aliasing a stride-4 subsample through a 5^3 window."""
    for s in (2, 3, 4):
        assert DatNet(stem_stride=s).stem[0].kernel_size == (2 * s + 1,) * 3
        assert SiameseDatNet(stem_stride=s).stem[0].kernel_size == (2 * s + 1,) * 3


def test_output_shape_is_one_logit_per_sample():
    for net in (DatNet(stem_stride=3), SiameseDatNet(stem_stride=3)):
        out = _fwd(net, n=3)
        assert out.shape == (3,), f"{type(net).__name__} gave {tuple(out.shape)}"


def test_finite_outputs_on_extreme_inputs():
    """Cubes are reference-multiples clipped at 12, then scaled by 6."""
    for net in (DatNet(stem_stride=3), SiameseDatNet(stem_stride=3)):
        net = net.eval()
        for fill in (0.0, 2.0, 12.0 / 6.0):
            with torch.no_grad():
                out = net(torch.full((2, 1, 24, 24, 24), fill))
            assert torch.isfinite(out).all(), f"{type(net).__name__} NaN at fill={fill}"


def test_capacity_parity_so_arch_is_the_variable():
    """Diversity experiment is invalid if the two arms differ in size."""
    a = count_params(DatNet(stem_stride=3))
    b = count_params(SiameseDatNet(stem_stride=3))
    assert abs(b - a) / a < 0.10, f"param gap {a:,} vs {b:,} exceeds 10%"


def test_eval_is_deterministic():
    """Dropout off and BatchNorm frozen in eval, so repeat calls must agree."""
    torch.manual_seed(0)
    for net in (DatNet(stem_stride=3), SiameseDatNet(stem_stride=3)):
        net = net.eval()
        x = torch.randn(2, 1, 24, 24, 24)
        with torch.no_grad():
            assert torch.equal(net(x), net(x)), f"{type(net).__name__} non-deterministic"


def test_train_mode_batchnorm_does_not_leak_across_samples_at_eval():
    """Rule 6: no cross-sample statistics at inference.

    In eval mode BatchNorm uses running stats, so a sample's prediction must not
    depend on what else is in the batch. Verified numerically because getting this
    wrong is a disqualification, not a bug.
    """
    torch.manual_seed(0)
    for net in (DatNet(stem_stride=3), SiameseDatNet(stem_stride=3)):
        net = net.eval()
        x = torch.randn(4, 1, 24, 24, 24)
        with torch.no_grad():
            batched = net(x)
            alone = torch.cat([net(x[i : i + 1]) for i in range(4)])
        gap = float((batched - alone).abs().max())
        assert gap < 1e-4, f"{type(net).__name__} batch-dependent at eval: {gap:.2e}"


def test_build_dispatch():
    assert isinstance(build("datnet", stem_stride=3), DatNet)
    assert isinstance(build("siamese", stem_stride=3), SiameseDatNet)
    try:
        build("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown arch should raise ValueError")


def test_siamese_splits_at_the_midline():
    """Odd and even grids must both split without dropping or double-counting."""
    for g in (24, 25):
        out = _fwd(SiameseDatNet(stem_stride=3), n=2, g=g)
        assert out.shape == (2,), f"grid {g} failed"


def test_grids_used_in_practice_all_forward():
    """64/96/128 are the three cached resolutions."""
    for g, s in ((64, 2), (96, 3), (128, 4)):
        for net in (DatNet(stem_stride=s), SiameseDatNet(stem_stride=s)):
            out = _fwd(net, n=1, g=g)
            assert out.shape == (1,), f"{type(net).__name__} failed at grid {g}"
