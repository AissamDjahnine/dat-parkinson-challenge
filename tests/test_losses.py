"""Contracts for the noise-robust losses.

The load-bearing property is that **every loss reduces to plain BCE at its neutral
parameter**, so the sweep is a clean interpolation from current behaviour rather than a
jump. If that breaks, an arm's result cannot be attributed to the loss shape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from datpark.losses import LOSSES, bce, bootstrap_soft, gce, get_loss, sce, trimmed_bce


def _lt(n=64, seed=0):
    torch.manual_seed(seed)
    logit = torch.randn(n) * 2
    target = (torch.rand(n) < 0.55).float()
    return logit, target


# ------------------------------------------------- reduces to BCE at neutral setting
def test_sce_reduces_to_bce_at_beta_zero():
    l, t = _lt()
    assert torch.allclose(sce(l, t, alpha=1.0, beta=0.0), bce(l, t), atol=1e-6)


def test_bootstrap_reduces_to_bce_at_beta_one():
    l, t = _lt()
    assert torch.allclose(bootstrap_soft(l, t, beta=1.0), bce(l, t), atol=1e-6)


def test_trimmed_reduces_to_bce_at_zero_drop():
    l, t = _lt()
    assert torch.allclose(trimmed_bce(l, t, drop_frac=0.0), bce(l, t), atol=1e-6)


def test_gce_approaches_bce_as_q_goes_to_zero():
    """GCE -> CE as q -> 0. Check the trend rather than exact equality."""
    l, t = _lt()
    ref = float(bce(l, t))
    errs = [abs(float(gce(l, t, q=q)) - ref) for q in (0.5, 0.1, 0.01)]
    assert errs[0] > errs[1] > errs[2], f"GCE not converging to BCE: {errs}"


# ------------------------------------------------------------- the robustness property
def test_gce_downweights_confidently_wrong_samples_relative_to_bce():
    """★ The mechanism that distinguishes these from label smoothing.

    A confidently-mislabelled sample should contribute a SMALLER share of the total
    gradient under GCE than under BCE. Smoothing cannot do this -- it caps confidence on
    every sample equally, which is why an earlier analysis found it taxes the easy 90 %.
    """
    logit = torch.tensor([5.0, 0.5, -0.5, 1.0], requires_grad=True)
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])          # first one is confidently wrong

    def share(fn):
        g = torch.autograd.grad(fn(logit, target), logit, retain_graph=False)[0].abs()
        return float(g[0] / g.sum())

    s_bce = share(lambda a, b: F.binary_cross_entropy_with_logits(a, b))
    s_gce = share(lambda a, b: gce(a, b, q=0.7))
    assert s_gce < s_bce, (
        f"GCE gives the confidently-wrong sample {s_gce:.3f} of the gradient vs "
        f"BCE's {s_bce:.3f} — expected less")


def test_trimmed_actually_drops_the_worst():
    l, t = _lt(n=100)
    assert float(trimmed_bce(l, t, drop_frac=0.1)) < float(bce(l, t))


def test_bootstrap_pulls_targets_toward_prediction():
    """A sample the model believes strongly should see a softened target."""
    logit = torch.tensor([4.0])
    target = torch.tensor([0.0])
    hard = float(F.binary_cross_entropy_with_logits(logit, target))
    soft = float(bootstrap_soft(logit, target, beta=0.7))
    assert soft < hard, "bootstrapping should reduce the loss on a disputed sample"


# --------------------------------------------------------------------------- hygiene
def test_all_losses_are_finite_and_scalar():
    l, t = _lt()
    for name, fn in LOSSES.items():
        v = fn(l, t)
        assert v.shape == (), f"{name} did not reduce to a scalar"
        assert torch.isfinite(v), f"{name} produced {v}"


def test_all_losses_survive_saturated_logits():
    """Cubes are clipped and the model can saturate; no loss may produce NaN/inf."""
    for mag in (30.0, -30.0):
        l = torch.full((16,), mag)
        for tv in (0.0, 1.0):
            t = torch.full((16,), tv)
            for name, fn in LOSSES.items():
                v = fn(l, t)
                assert torch.isfinite(v), f"{name} not finite at logit={mag} target={tv}"


def test_all_losses_are_differentiable():
    l, t = _lt()
    for name, fn in LOSSES.items():
        x = l.clone().requires_grad_(True)
        g = torch.autograd.grad(fn(x, t), x)[0]
        assert torch.isfinite(g).all(), f"{name} produced non-finite gradients"
        assert g.abs().sum() > 0, f"{name} produced zero gradient everywhere"


def test_losses_accept_smoothed_targets():
    """The trainer may pass label-smoothed targets, so targets are floats in [0,1]."""
    l, _ = _lt()
    t = torch.full_like(l, 0.525)
    for name, fn in LOSSES.items():
        assert torch.isfinite(fn(l, t)), f"{name} failed on a soft target"


def test_get_loss_dispatch_and_error():
    assert get_loss("gce") is gce
    try:
        get_loss("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown loss should raise ValueError")
