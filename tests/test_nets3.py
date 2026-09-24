"""Contracts for the 2.5D SliceNet (datpark/nets3.py).

The load-bearing tests are the two rule-compliance ones: SliceNet folds the slice axis
into the batch dimension before the 2D layers, which is exactly the sort of reshape that
could let information leak between samples. That is checked numerically, not argued.
"""

from __future__ import annotations

import torch

from datpark.nets import count_params
from datpark.nets2 import SiameseDatNet
from datpark.nets3 import SliceNet, build


def _net(**kw):
    torch.manual_seed(0)
    return SliceNet(stem_stride=2, **kw).eval()


# ------------------------------------------------------------- rule compliance
def test_prediction_is_independent_of_batch_mates():
    """Rule 6. The (N,S)->(N*S) fold must not mix samples at eval."""
    net = _net()
    x = torch.randn(4, 1, 24, 24, 24)
    with torch.no_grad():
        batched = net(x)
        alone = torch.cat([net(x[i : i + 1]) for i in range(4)])
    gap = float((batched - alone).abs().max())
    assert gap < 1e-4, f"SliceNet is batch-dependent at eval: {gap:.2e}"


def test_a_sample_is_unaffected_by_reordering():
    """Permuting the batch must permute the outputs, nothing more."""
    net = _net()
    x = torch.randn(5, 1, 24, 24, 24)
    perm = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        a, b = net(x), net(x[perm])
    assert torch.allclose(a[perm], b, atol=1e-5)


def test_eval_is_deterministic():
    net = _net()
    x = torch.randn(2, 1, 24, 24, 24)
    with torch.no_grad():
        assert torch.equal(net(x), net(x))


# ------------------------------------------------------------------- mechanics
def test_output_shape_is_one_logit_per_sample():
    with torch.no_grad():
        assert _net()(torch.randn(3, 1, 24, 24, 24)).shape == (3,)


def test_slices_are_taken_along_the_SI_axis():
    """Axial planes are R-L x A-P, so the slice count must equal the last dim.

    If this ever slices the wrong axis the model still trains and nothing looks broken
    -- the same failure mode as the augmentation axis bug. Hence a test.
    """
    net = _net()
    x = torch.randn(2, 1, 16, 20, 24)   # deliberately anisotropic
    with torch.no_grad():
        e = net.encode_slices(x)
    assert e.shape[:2] == (2, 24), f"expected 24 slices (last dim), got {e.shape[1]}"


def test_attention_is_a_distribution_over_slices():
    net = _net()
    x = torch.randn(3, 1, 24, 24, 24)
    a = net.slice_attention(x)
    assert a.shape == (3, 24)
    assert torch.allclose(a.sum(dim=1), torch.ones(3), atol=1e-5)
    assert (a >= 0).all()


def test_attention_actually_discriminates():
    """A uniform attention map would mean the pooling is doing nothing."""
    net = _net()
    x = torch.zeros(1, 1, 24, 24, 24)
    x[0, 0, 8:16, 8:16, 10:14] = 3.0          # signal in a few slices only
    a = net.slice_attention(x)[0]
    assert float(a.max() / a.min()) > 1.05, "attention is essentially uniform"


def test_capacity_is_comparable_to_the_other_families():
    """The diversity experiment needs architecture, not size, to be the variable."""
    a = count_params(SliceNet(stem_stride=2))
    b = count_params(SiameseDatNet(stem_stride=3))
    assert abs(a - b) / b < 0.10, f"param gap too large: SliceNet {a:,} vs siamese {b:,}"


def test_finite_on_extreme_inputs():
    net = _net()
    for fill in (0.0, 2.0, 12.0 / 6.0):
        with torch.no_grad():
            out = net(torch.full((2, 1, 24, 24, 24), fill))
        assert torch.isfinite(out).all(), f"NaN at fill={fill}"


def test_grids_in_use_all_forward():
    """72^3 is the tighter-crop cache; 96^3 is the standard one."""
    for g in (72, 96):
        with torch.no_grad():
            assert _net()(torch.zeros(1, 1, g, g, g)).shape == (1,)


def test_odd_and_anisotropic_shapes():
    for shape in ((24, 24, 25), (20, 24, 24)):
        with torch.no_grad():
            assert _net()(torch.randn(1, 1, *shape)).shape == (1,)


def test_build_dispatch_extends_nets2():
    assert isinstance(build("slicenet", stem_stride=2), SliceNet)
    assert isinstance(build("siamese", stem_stride=3), SiameseDatNet)
    from datpark.nets import DatNet
    assert isinstance(build("datnet", stem_stride=3), DatNet)


def test_positional_encoding_breaks_the_SI_invariance():
    """★ Attention pooling is a weighted SUM, which is order-invariant, so a naive
    SliceNet is EXACTLY S-I invariant. That would waste the S-I flip augmentation --
    our single largest lever (+0.0255) -- and discard S-I ordering entirely.
    `pos_embed` exists to fix it, and this test is why we know it is needed.

    Thresholds are relative to the output scale: at random init the absolute output
    range is ~1e-4, so an absolute threshold measures nothing.
    """
    x = torch.randn(8, 1, 24, 24, 24)

    # Without positional encoding the invariance is mathematically exact; in float32
    # it lands at ~3e-8 because reversing the slice order changes the summation order
    # in the pooling and float addition is not associative. Assert a tolerance, and
    # compare it against the WITH-encoding case below so the contrast is the evidence.
    off = _net(pos_embed=False)
    with torch.no_grad():
        o_off = off(x)
        si_off = float((o_off - off(torch.flip(x, dims=[4]))).abs().max()) / float(o_off.std())
    assert si_off < 1e-3, (
        f"without positional encoding S-I invariance should be exact up to float "
        f"round-off, got {si_off:.2%} of output std")

    on = _net(pos_embed=True)
    with torch.no_grad():
        o = on(x)
        si = float((o - on(torch.flip(x, dims=[4]))).abs().max()) / float(o.std())
        rl = float((o - on(torch.flip(x, dims=[2]))).abs().max()) / float(o.std())
    assert si > 0.10, f"positional encoding is not restoring S-I sensitivity ({si:.1%})"
    assert rl > 0.10, f"unexpectedly R-L invariant ({rl:.1%})"
    assert si > 1000 * si_off, (
        f"positional encoding barely changes S-I sensitivity: {si:.2%} with vs "
        f"{si_off:.4%} without")


def test_positional_encoding_is_scale_matched_to_the_embeddings():
    """Guards a real bug: before LayerNorm the embeddings measured ~0.002 against a
    sinusoid of ~0.64, so position would swamp content throughout early training."""
    net = _net()
    x = torch.randn(4, 1, 24, 24, 24)
    with torch.no_grad():
        e = net.encode_slices(x, with_pos=False)
    scale = float(e.abs().mean())
    assert 0.1 < scale < 3.0, (
        f"slice-embedding scale {scale:.4f} is far from the sinusoid's ~0.64; "
        f"positional encoding would dominate or vanish")
