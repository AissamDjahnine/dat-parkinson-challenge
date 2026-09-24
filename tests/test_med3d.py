"""Contracts for datpark/med3d.py — the MedicalNet-pretrained 3D ResNet.

The whole point of this experiment is to answer "does a pretrained initialisation help?", and
there is exactly one way to get a *fake* answer: load the weights badly, train a randomly
initialised network, and report the resulting null as evidence about pretraining.
`torch.load_state_dict(strict=False)` makes that failure silent — rename one layer and it
loads nothing while raising nothing.

So the load path is tested harder than the architecture:

  * `test_every_encoder_tensor_is_loaded` pins the exact tensor and parameter counts.
  * `test_a_key_mismatch_raises_instead_of_silently_skipping` corrupts a checkpoint on purpose
    and requires an exception — this is the test that makes a null trustworthy.
  * `test_pretrained_and_scratch_actually_differ` compares outputs under an identical seed, so
    "pretrained" cannot silently mean "random".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from datpark.med3d import MedResNet3d, build_med3d, load_medicalnet

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "artifacts" / "pretrained" / "resnet_10_23dataset.pth"

# Read off the published checkpoint, not from the paper: 72 tensors, no classifier.
N_TENSORS = 72
N_ENCODER_PARAMS = 14_361_292


def _need_ckpt():
    if not CKPT.exists():
        from run_tests import SkipTest
        raise SkipTest("MedicalNet r10 checkpoint not downloaded")


def test_architecture_shapes_and_output_contract():
    """Logits must be shape (N,) — the training loop and losses assume DatNet's contract."""
    m = MedResNet3d(depth=10, dropout=0.0)
    x = torch.zeros(2, 1, 32, 32, 32)
    m.eval()
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2,), out.shape
    assert torch.isfinite(out).all()


def test_depth_10_and_18_have_the_expected_capacity():
    """r10 is 14.36 M — 9x our 1.6 M champion, which is the overfitting risk to keep in mind.

    Note the two counts that differ by 5,772 and must not be conflated: the checkpoint holds
    14,361,292 *state_dict* numbers, while `.parameters()` reports 14,355,520 because it
    excludes BatchNorm buffers — 2 x 2,880 running stats plus 12 `num_batches_tracked`
    scalars. The first version of this test asserted the checkpoint figure against
    `.parameters()` and failed for exactly that reason.
    """
    p10 = sum(p.numel() for p in MedResNet3d(depth=10).parameters())
    p18 = sum(p.numel() for p in MedResNet3d(depth=18).parameters())
    assert p18 > p10, (p10, p18)
    assert p10 == 14_355_520 + 513, p10                      # encoder + 512->1 head

    m = MedResNet3d(depth=10)
    buffers = sum(b.numel() for b in m.buffers())
    assert buffers == 5_772, buffers
    assert p10 - 513 + buffers == N_ENCODER_PARAMS, (p10, buffers)


def test_every_encoder_tensor_is_loaded():
    """No tensor may be quietly left at its random initialisation."""
    _need_ckpt()
    m = MedResNet3d(depth=10)
    info = load_medicalnet(m, CKPT)
    assert info["loaded_tensors"] == N_TENSORS, info["loaded_tensors"]
    assert info["loaded_params"] == N_ENCODER_PARAMS, info["loaded_params"]
    # only the classification head is fresh; the checkpoint is a segmentation encoder
    assert all(k.startswith("head.") for k in info["fresh_tensors"]), info["fresh_tensors"]


def test_loaded_weights_match_the_file_bit_for_bit():
    """Guards against a transposed / reshaped load that would keep the count but move data."""
    _need_ckpt()
    m = MedResNet3d(depth=10)
    load_medicalnet(m, CKPT)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    for key in ("module.conv1.weight", "module.layer4.0.conv2.weight",
                "module.layer2.0.downsample.0.weight", "module.bn1.running_var"):
        mine = m.state_dict()[key.removeprefix("module.")]
        assert torch.equal(mine, raw[key]), f"{key} differs after load"


def test_a_key_mismatch_raises_instead_of_silently_skipping(tmp=None):
    """THE TEST THAT MAKES A NULL MEANINGFUL.

    A checkpoint whose keys do not match must fail loudly. Without this, a refactor that
    renames `layer1` would produce a randomly-initialised "pretrained" model, and the
    experiment would conclude "pretraining does not help" while never having used it.
    """
    _need_ckpt()
    import tempfile

    raw = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    broken = {k.replace("layer1", "stage1"): v for k, v in raw.items()}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "broken.pth"
        torch.save({"state_dict": broken}, p)
        m = MedResNet3d(depth=10)
        try:
            load_medicalnet(m, p)
        except RuntimeError:
            return
        raise AssertionError("a renamed encoder key loaded silently — a null would be fake")


def test_pretrained_and_scratch_actually_differ():
    """Same architecture, same seed, different init => different outputs."""
    _need_ckpt()
    x = torch.randn(2, 1, 32, 32, 32)
    torch.manual_seed(0)
    pre, ipre = build_med3d(depth=10, weights=CKPT, dropout=0.0, verbose=False)
    torch.manual_seed(0)
    scr, iscr = build_med3d(depth=10, weights=None, dropout=0.0, verbose=False)
    assert ipre["pretrained"] and not iscr["pretrained"]
    pre.eval()
    scr.eval()
    with torch.no_grad():
        a, b = pre.features(x), scr.features(x)
    assert not torch.allclose(a, b, atol=1e-4), "pretrained features equal random ones"


def test_eval_mode_is_deterministic():
    """Rule 7: identical input must give an identical prediction, so BN must be in eval."""
    _need_ckpt()
    m, _ = build_med3d(depth=10, weights=CKPT, dropout=0.5, verbose=False)
    m.eval()
    x = torch.randn(3, 1, 32, 32, 32)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b), "eval-mode forward is not deterministic"


def test_prediction_does_not_depend_on_batch_composition():
    """Rule 6 (test-sample independence): a scan's logit must not move with its neighbours."""
    _need_ckpt()
    m, _ = build_med3d(depth=10, weights=CKPT, dropout=0.0, verbose=False)
    m.eval()
    torch.manual_seed(1)
    x = torch.randn(4, 1, 32, 32, 32)
    with torch.no_grad():
        alone = m(x[:1])
        together = m(x)
    assert torch.allclose(alone, together[:1], atol=1e-5), \
        "logit changed when other scans shared the batch — BatchNorm is not in eval"


def test_freeze_until_freezes_a_growing_prefix():
    _need_ckpt()
    counts = []
    for stage in ["none", "stem", "layer1", "layer2", "layer3"]:
        _, info = build_med3d(depth=10, weights=CKPT, freeze_until=stage, verbose=False)
        counts.append(info["frozen_params"])
        assert info["trainable_params"] + info["frozen_params"] == info["params"]
    assert counts == sorted(counts) and counts[0] == 0, counts
    assert counts[-1] > counts[1], counts


def test_r18_uses_type_A_shortcuts_and_loads_completely():
    """r18's checkpoint has NO downsample tensors — its shortcuts are parameter-free.

    MedicalNet trained r10 with `--resnet_shortcut B` and r18/r34 with `A`. Building r18 with
    B would add downsample tensors the checkpoint cannot fill, so this is the test that keeps
    a second depth honest rather than silently half-initialised.
    """
    ck18 = ROOT / "artifacts" / "pretrained" / "resnet_18_23dataset.pth"
    if not ck18.exists():
        from run_tests import SkipTest
        raise SkipTest("MedicalNet r18 checkpoint not downloaded")
    m, info = build_med3d(depth=18, weights=ck18, dropout=0.0, verbose=False)
    assert info["shortcut"] == "A", info["shortcut"]
    assert info["loaded_tensors"] == 102, info["loaded_tensors"]
    assert all(k.startswith("head.") for k in info["fresh_tensors"]), info["fresh_tensors"]
    assert not any("downsample" in k for k in m.state_dict()), "type A must carry no parameters"
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 1, 32, 32, 32))
    assert out.shape == (2,) and torch.isfinite(out).all()


def test_wrong_shortcut_type_is_rejected_not_half_loaded():
    """Forcing B on the r18 checkpoint must raise, not leave the projections random."""
    ck18 = ROOT / "artifacts" / "pretrained" / "resnet_18_23dataset.pth"
    if not ck18.exists():
        from run_tests import SkipTest
        raise SkipTest("MedicalNet r18 checkpoint not downloaded")
    try:
        build_med3d(depth=18, weights=ck18, shortcut="B", verbose=False)
    except RuntimeError:
        return
    raise AssertionError("shortcut B loaded an r18 type-A checkpoint without complaint")


def test_type_A_shortcut_preserves_the_signal_it_pads():
    """The zero-pad shortcut must keep the pooled channels, not blank them."""
    from datpark.med3d import BasicBlock3d
    blk = BasicBlock3d(4, 8, stride=2, shortcut="A")
    x = torch.ones(1, 4, 8, 8, 8)
    idt = blk._identity(x)
    assert idt.shape == (1, 8, 4, 4, 4), idt.shape
    assert torch.equal(idt[:, :4], torch.ones(1, 4, 4, 4, 4)), "pooled channels were altered"
    assert torch.equal(idt[:, 4:], torch.zeros(1, 4, 4, 4, 4)), "padding is not zero"


def test_licence_and_provenance_are_recorded_in_the_module():
    """Prize eligibility depends on the licence, so it must be documented where the code is."""
    text = (ROOT / "datpark" / "med3d.py").read_text()
    assert "MIT" in text
    assert "no network" in text.lower()          # weights must ship inside the ZIP
    assert "isclose" in text or "Disclose" in text or "disclose" in text


def test_standardise_is_per_scan_and_leaves_no_scan_unnormalised():
    """Per-scan z-scoring: every scan ends at ~0 mean / ~1 std, computed from itself alone.

    This closes a false-null risk rather than a crash: MedicalNet was pretrained on z-scored
    volumes, while our cubes are reference-multiples with mean ~1 and a range of ~[0, 12].
    Feeding the pretrained stem the wrong scale would sink the arm for a reason unrelated to
    whether medical pretraining transfers.
    """
    from datpark.med3d import standardise_per_scan
    torch.manual_seed(0)
    # three scans with deliberately different scales and offsets
    X = torch.stack([torch.randn(1, 8, 8, 8) * 3 + 10,
                     torch.randn(1, 8, 8, 8) * 0.1 - 5,
                     torch.randn(1, 8, 8, 8) * 1.0]).float()
    Z = standardise_per_scan(X)
    for i in range(len(Z)):
        assert abs(float(Z[i].mean())) < 1e-4, (i, float(Z[i].mean()))
        assert abs(float(Z[i].std()) - 1.0) < 1e-2, (i, float(Z[i].std()))


def test_standardise_does_not_leak_across_scans():
    """Rule 6: a scan's normalisation must not change when other scans are present."""
    from datpark.med3d import standardise_per_scan
    torch.manual_seed(1)
    a = torch.randn(1, 1, 8, 8, 8) * 2 + 4
    others = torch.randn(3, 1, 8, 8, 8) * 50 - 100      # wildly different neighbours
    alone = standardise_per_scan(a)
    together = standardise_per_scan(torch.cat([a, others]))[:1]
    assert torch.allclose(alone, together, atol=1e-5), \
        "standardisation depended on the other scans in the batch"


def test_frozen_batchnorm_does_not_drift_in_train_mode():
    """FROZEN MUST MEAN TWO THINGS: no gradient AND no running-stat update.

    `requires_grad_(False)` only does the first. A frozen BatchNorm left in train() mode
    overwrites running_mean/var from every batch, so the pretrained statistics get replaced by
    SPECT statistics epoch by epoch — the "frozen encoder" arm would silently not be frozen and
    the freeze curve would measure nothing clean. This was a live bug, caught by writing the
    test, and `set_train_mode` is the fix.
    """
    from datpark.med3d import set_train_mode
    _need_ckpt()
    m, _ = build_med3d(depth=10, weights=CKPT, freeze_until="layer4", verbose=False)
    set_train_mode(m)

    assert not m.bn1.training, "frozen stem BN is still in train mode"
    assert m.head.training, "the head must remain trainable"

    before = m.bn1.running_mean.clone()
    m(torch.randn(4, 1, 32, 32, 32) * 5 + 20)     # deliberately off-distribution input
    assert torch.equal(m.bn1.running_mean, before), "frozen BN running stats drifted"


def test_unfrozen_batchnorm_still_adapts():
    """The complement: a normally fine-tuned model MUST update its BN stats.

    Without this, a `set_train_mode` that froze everything would pass the test above while
    quietly breaking every full fine-tuning arm.
    """
    from datpark.med3d import set_train_mode
    _need_ckpt()
    m, _ = build_med3d(depth=10, weights=CKPT, freeze_until="none", verbose=False)
    set_train_mode(m)
    assert m.bn1.training
    before = m.bn1.running_mean.clone()
    m(torch.randn(4, 1, 32, 32, 32) * 5 + 20)
    assert not torch.equal(m.bn1.running_mean, before), "BN stats did not adapt when unfrozen"


def test_the_freeze_curve_covers_every_stage():
    """All five points of the curve must be constructible, from full fine-tune to linear probe."""
    _need_ckpt()
    seen = {}
    for stage in ["none", "layer1", "layer2", "layer3", "layer4"]:
        _, info = build_med3d(depth=10, weights=CKPT, freeze_until=stage, verbose=False)
        seen[stage] = info["trainable_params"]
    assert seen["none"] > seen["layer1"] > seen["layer2"] > seen["layer3"] > seen["layer4"], seen
    assert seen["layer4"] == 513, seen["layer4"]      # head only: 512 weights + 1 bias


def test_standardise_is_chunk_invariant_and_memory_bounded():
    """Chunking is a memory fix, so it must not change the numbers it produces.

    The one-shot version peaked near 11 GB on the real cache against ~16 GB free — a realistic
    overnight OOM. Any chunk size must give the identical result.
    """
    from datpark.med3d import standardise_per_scan
    torch.manual_seed(3)
    X = (torch.randn(11, 1, 6, 6, 6) * 4 + 7).half()
    ref = standardise_per_scan(X, chunk=1)
    for c in (2, 4, 11, 64):
        assert torch.equal(standardise_per_scan(X, chunk=c), ref), f"chunk={c} differs"


def test_standardise_survives_a_constant_scan():
    """A dead/blank scan has std 0 — eps must keep it finite instead of producing NaNs."""
    from datpark.med3d import standardise_per_scan
    X = torch.cat([torch.full((1, 1, 4, 4, 4), 3.0), torch.randn(1, 1, 4, 4, 4)])
    Z = standardise_per_scan(X)
    assert torch.isfinite(Z).all(), "a constant scan produced non-finite output"
