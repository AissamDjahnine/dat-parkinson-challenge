"""Contracts for solutions/1_008a — the 008a tree (005 + SliceNet + MedicalNet — four CNN families).

The dangerous part of this tree is not the architecture, it is the INPUT NORMALISATION.
measured a 2x2: per-scan z-scoring is worth +0.0227 to the pretrained model and
-0.0298 to a random-init one, and on un-z-scored input the frozen encoder collapses to a
near-constant function (cosine similarity 0.9998 between different scans). So a normalisation
path that silently disagrees with training would not crash — it would quietly destroy the
family while every other check stayed green. `test_zscore_inference_path_matches_training` is
the guard, and it compares against the real training function rather than a reimplementation.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "solutions" / "1_008a"
CUBE_SCALE = 6.0


def _skip_if_absent():
    if not TREE.exists():
        from run_tests import SkipTest
        raise SkipTest("solutions/1_008a not present")


def test_v5_datpark_matches_the_training_code():
    _skip_if_absent()
    for f in sorted((TREE / "datpark").glob("*.py")):
        root = ROOT / "datpark" / f.name
        assert root.exists(), f"{f.name} is in the tree but not at the root"
        assert f.read_bytes() == root.read_bytes(), f"{f.name} drifted from the root copy"


def test_med3d_is_present_and_wired():
    _skip_if_absent()
    assert (TREE / "datpark" / "med3d.py").exists()
    src = (TREE / "main.py").read_text()
    assert "from datpark.med3d import MedResNet3d" in src
    assert '"MedResNet3d"' in src, "no builder registered"
    assert 'norm == "zscore"' in src, "the z-score branch is missing"
    assert 's.get("norm", "scale")' in src, "families do not carry a norm setting"


def test_zscore_inference_path_matches_training():
    """THE CRITICAL TEST: the container's normalisation must equal training's, on real numbers.

    Training path: load_data divides by SCALE and returns float16, then
    datpark.med3d.standardise_per_scan z-scores per scan with eps 1e-5.
    Inference path: main.py divides by CUBE_SCALE then z-scores per scan with eps 1e-5.
    """
    _skip_if_absent()
    from datpark.med3d import standardise_per_scan

    rng = np.random.default_rng(0)
    # values shaped like real cubes: reference-multiples, mean ~0.2 after /SCALE, long tail
    cubes = (rng.gamma(2.0, 0.6, size=(5, 24, 24, 24)) * 2.0).astype(np.float32)

    # --- training ---
    train_in = torch.from_numpy(cubes / CUBE_SCALE).unsqueeze(1).half()
    train_out = standardise_per_scan(train_in).float()

    # --- inference, exactly as main.py does it ---
    x = torch.from_numpy(cubes / CUBE_SCALE).unsqueeze(1)
    flat = x.reshape(x.shape[0], -1).float()
    mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
    sd = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
    infer_out = (x.float() - mu) / (sd + 1e-5)

    # float16 storage in training is the only legitimate difference
    diff = (train_out - infer_out).abs().max().item()
    assert diff < 5e-3, f"inference normalisation differs from training by {diff:.2e}"
    for i in range(len(cubes)):
        assert abs(float(infer_out[i].mean())) < 1e-4
        assert abs(float(infer_out[i].std()) - 1.0) < 1e-2


def test_a_scale_family_is_untouched_by_the_zscore_branch():
    """sm0 and siamese must keep the plain /CUBE_SCALE path — 005's behaviour is unchanged."""
    _skip_if_absent()
    cfg = json.loads((TREE / "assets" / "config.json").read_text())
    for name in ("sm0", "siamese"):
        assert cfg["cnn_families"][name].get("norm", "scale") == "scale", \
            f"{name} must not be z-scored"
    assert cfg["cnn_families"]["med3d"]["norm"] == "zscore"


def test_every_config_family_has_a_builder_and_a_tta_scheme():
    _skip_if_absent()
    cfg = json.loads((TREE / "assets" / "config.json").read_text())
    src = (TREE / "main.py").read_text()
    tree = ast.parse(src)
    builders: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "BUILDERS" for t in node.targets):
            builders = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    consts = {n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for name, spec in cfg["cnn_families"].items():
        assert spec["arch"] in builders, f"{name}: no builder for {spec['arch']}"
        assert spec["tta"] in consts, f"{name}: views() cannot build {spec['tta']}"
        assert spec["grid"] in cfg["grids"], f"{name}: unknown grid {spec['grid']}"


def test_licence_disclosure_is_recorded_in_the_shipped_config():
    """Rule 5 requires disclosing pretrained models. The obligation must travel with the zip."""
    _skip_if_absent()
    cfg = json.loads((TREE / "assets" / "config.json").read_text())
    ext = cfg["cnn_families"]["med3d"]["external_weights"]
    assert "MIT" in ext["licence"]
    assert "DISCLOSURE" in ext and "organizers" in ext["DISCLOSURE"]


def test_batchnorm_guard_covers_2d_and_3d():
    _skip_if_absent()
    src = (TREE / "main.py").read_text()
    assert "BatchNorm2d" in src and "BatchNorm3d" in src


def test_all_four_cnn_families_are_present_and_wired():
    """008a is 005 + BOTH new members. Missing one silently ships a different model."""
    _skip_if_absent()
    cfg = json.loads((TREE / "assets" / "config.json").read_text())
    assert set(cfg["cnn_families"]) == {"sm0", "siamese", "slicenet", "med3d"}, \
        sorted(cfg["cnn_families"])
    src = (TREE / "main.py").read_text()
    for imp in ("from datpark.nets3 import SliceNet", "from datpark.med3d import MedResNet3d"):
        assert imp in src, f"missing import: {imp}"
    assert (TREE / "datpark" / "nets3.py").exists()
    assert (TREE / "datpark" / "med3d.py").exists()


def test_only_med3d_is_zscored():
    """The z-score branch must apply to MedicalNet alone — it HURTS from-scratch models.

    the earlier 2x2: z-scoring is worth +0.0227 to the pretrained net and -0.0298 to a random-init
    one. Applying it to sm0, siamese or slicenet would quietly damage three families.
    """
    _skip_if_absent()
    cfg = json.loads((TREE / "assets" / "config.json").read_text())
    for name in ("sm0", "siamese", "slicenet"):
        assert cfg["cnn_families"][name].get("norm", "scale") == "scale", \
            f"{name} must NOT be z-scored"
    assert cfg["cnn_families"]["med3d"]["norm"] == "zscore"


def test_the_honest_cv_caveats_travel_with_the_package():
    """The shipped config must state the SHIPPED numbers, with their caveats attached.

    This test previously required a `shippable_log_loss` field, which existed only because
    `log_loss` held the DEVELOPMENT figure 0.2468 and something had to flag that the package
    actually scores 0.2472. That was a workaround for a defect, and the test enshrined it.
    `log_loss` now holds the shipped number directly, so the field is gone and the contract is
    stronger: the headline figure must BE the shipped one.
    """
    _skip_if_absent()
    cfg = json.loads((TREE / "assets" / "config.json").read_text())
    cv = cfg["cv"]
    assert cv["log_loss"] == 0.2472, f"headline CV must be the shipped 0.2472, got {cv['log_loss']}"
    assert cv["auroc"] == 0.9607, f"headline AUROC must be the shipped 0.9607, got {cv['auroc']}"
    assert "basis" in cv, "the config must say WHICH basis its numbers are on"
    assert "+0.0002" in cv["honest_note"], "the shrunk floor must be stated"
    assert "+0.0078" in cv["vs_005"], "the advantage over 005 must be the shipped +0.0078"

    txt = json.dumps(cfg)
    assert "NOT bit-reproducible" in txt, "the med3d reproducibility caveat must ship"
    assert cfg["cnn_families"]["med3d"]["oof"]["log_loss"] == 0.2833, (
        "med3d must report the checkpoints IN this package (0.2833), not the dev run (0.2812)"
    )


def test_the_retracted_claim_is_not_presented_as_a_live_claim():
    """Regression: the shipped archive asserted something the report retracts as an error.

    `config.json` stated '+0.0082 (P=0.985, 95% CI [+0.0009, +0.0152]) - the ONLY candidate whose
    interval excludes zero'. On the shipped basis the interval does NOT exclude zero
    ([-0.0000, +0.0149], P=0.975); the figures came from a development OOF. A juror unzipping the
    package would have found us claiming more than our own paper supports.

    The phrase may still appear inside the correction note that records the error — that is the
    point of the note — but it must not appear in any field a reader would take as a live claim.
    """
    _skip_if_absent()
    cv = json.loads((TREE / "assets" / "config.json").read_text())["cv"]
    correction_keys = [k for k in cv if k.startswith("correction")]
    assert correction_keys, "the config must record the correction rather than silently drop it"

    live = " ".join(str(v) for k, v in cv.items() if not k.startswith("correction"))
    for banned in ("excludes zero", "+0.0082", "0.2468", "0.9608"):
        assert banned not in live, (
            f"retracted/development figure {banned!r} is presented as a live claim"
        )


def test_cnn_predict_actually_accepts_and_receives_norm():
    """STRUCTURAL, not textual — this is the bug that got through.

    An earlier patch added the z-score BRANCH and the config field but silently failed to
    widen `cnn_predict`'s signature, because the edit used a string replace that did not
    match and was not asserted. Every text-based check still passed: the branch was present,
    the config carried `norm`, the call site passed it. The container then died with
    `TypeError: cnn_predict() takes 4 positional arguments but 5 were given` — after packing
    a gigabyte. So this test parses the AST and checks the function really takes the
    parameter and every call really supplies it.
    """
    _skip_if_absent()
    tree = ast.parse((TREE / "main.py").read_text())

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "cnn_predict"), None)
    assert fn is not None, "cnn_predict not found"
    params = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "norm" in params, f"cnn_predict does not accept `norm`: {params}"

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "cnn_predict"]
    assert calls, "cnn_predict is never called"
    for c in calls:
        supplied = len(c.args) + len(c.keywords)
        assert supplied >= 5, f"a cnn_predict call passes only {supplied} arguments — norm lost"
