"""Inference entrypoint for the DaT Parkinson's Challenge - model 012.

This is the program that was submitted. Only this docstring differs from the shipped
archive; every executed line is unchanged.

011's 75 CNN checkpoints, byte-identical. Two changes: the SBR head is refitted on a
50-feature multiscale set (datpark/features_v2.py), and every volume is canonicalised to
RAS on load, so a scan stored in another orientation cannot have its laterality transposed.

Pipeline - each scan is decoded ONCE, then resampled to two grids:

    NIfTI -> RAS guard, brain mask, striatal centre, reference level   (computed once)
          -> 64^3 @2.0mm    -> 50 SBR features -> 15 LightGBM boosters     w 0.20
          -> 96^3 @1.333mm  -> sm0      DatNet stem3        x30, TTA si_rl        \\
          -> 96^3 @1.333mm  -> siamese  SiameseDatNet stem3 x15, TTA si_rot_scale  \\
          -> 96^3 @1.333mm  -> slicenet SliceNet            x15, TTA rl            > w 0.80
          -> 96^3 @1.333mm  -> med3d    MedResNet3d (r10)   x15, TTA rl, z-scored  /
          -> equal-weight logit mean of the four CNN families
          -> 0.8*CNN + 0.2*SBR in logit space -> clip -> submission.csv

Grouped CV (5 folds x 3 seeds, n=1362): log loss 0.2426, AUROC 0.9622.
Public leaderboard: log loss 0.2702.

Environment:
  * DATPARK_DATA_DIR   folder holding niftis/ and submission_format.csv
                       (the competition container mounts /code_execution/data instead)
  * assets/            config.json, models/lgb_*.txt, cnn/<family>/*.pt - built by
                       scripts/pack.sh 012; weights are not in the repository
Mask/centre/reference do not depend on `voxel_mm`, so they are computed once per scan
and only the crop+resample runs per grid. Preprocessing is ~97% of runtime.

No post-hoc calibration: cross-fitted temperature scaling measured worse.

Rule compliance:
  * every scan processed independently; no statistic crosses test samples
  * BatchNorm asserted eval() on every net, so a prediction cannot depend on its
    batch-mates
  * all TTA views are per-sample deterministic transforms
  * nothing fitted at inference; all parameters ship frozen in assets/
  * the log contains only submission-side facts -- never test-row counts, per-scan
    values, progress counts, rates or failure tallies
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

SRC_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(SRC_ROOT))

import lightgbm as lgb  # noqa: E402
import torch  # noqa: E402

# Multiscale semi-quantitative features. `datpark.features` holds the earlier twenty-feature set; it is
# retained as the reference implementation but is not part of this inference path.
#
# `robust_midline=False` places the left/right dividing plane at the cube midpoint, which is the geometry
# the boosters were fitted under. The alternative landmark-based midline is implemented in
# `datpark.midline` and is deliberately disabled: it was worth -0.0003 against this ensemble, because the
# plane is quantised to one 2 mm voxel while the displacement it corrects averages 1.3 mm, and the
# regional features are top-k means whose hottest voxels lie ~12 mm from the plane.
from datpark.features_v2 import FEATURE_NAMES_V2, feature_matrix_v2  # noqa: E402
from datpark.nets import DatNet, apply_affine  # noqa: E402
from datpark.nets2 import SiameseDatNet
from datpark.med3d import MedResNet3d, standardise_per_scan
from datpark.nets3 import SliceNet  # noqa: E402
from datpark.preprocess import (  # noqa: E402
    PreprocessConfig,
    _crop_resample,
    brain_mask,
    load_volume,
    reference_level,
    striatum_centre,
)

_CONTAINER_DATA = Path("/code_execution/data")
DATA_DIR = (
    _CONTAINER_DATA if _CONTAINER_DATA.is_dir()
    else Path(os.environ.get("DATPARK_DATA_DIR", str(_CONTAINER_DATA)))
)
NIFTI_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")
ASSETS = SRC_ROOT / "assets"

CUBE_SCALE = 6.0  # must match the normalisation constant used when the models were fitted
CNN_BATCH = 16

# theta row -> anatomical axis, under affine_grid's (x, y, z) = (W, H, D) ordering.
# Cubes are RAS with no reorientation, so axis0=R-L, axis1=A-P, axis2=S-I.
ROW_SI, ROW_AP, ROW_RL = 0, 1, 2

BUILDERS = {
    "DatNet": lambda s: DatNet(stem_stride=s),
    "SiameseDatNet": lambda s: SiameseDatNet(stem_stride=s),
    # MedicalNet r10. No 3D stem stride of its own (conv1 is 7^3 s2 + maxpool), so the
    # argument is ignored. Shortcut type follows the depth, exactly as the checkpoint needs.
    "MedResNet3d": lambda s: MedResNet3d(depth=10, dropout=0.3),
    # SliceNet is 2.5D — a shared 2D encoder over axial slices with attention pooling — so it
    # has no 3D stem and the stem_stride argument is intentionally ignored.
    "SliceNet": lambda s: SliceNet(),
}


# --------------------------------------------------------------------- assets
def load_config() -> dict:
    return json.loads((ASSETS / "config.json").read_text())


def make_cfg(d: dict) -> PreprocessConfig:
    return PreprocessConfig(
        voxel_mm=d["voxel_mm"], crop_mm=d["crop_mm"], brain_frac=d["brain_frac"],
        brain_pct=d["brain_pct"], hot_pct=d["hot_pct"], search_mm=d["search_mm"],
        ref_exclude_pct=d["ref_exclude_pct"], clip=d["clip"],
    )


def _assert_feature_manifest(raw: dict) -> None:
    """The boosters were trained on FEATURE_NAMES_V2 in this exact order. A reordering or a length
    change would be silent and catastrophic, so it is checked at startup against the shipped config."""
    declared = list(raw.get("feature_names", []))
    actual = list(FEATURE_NAMES_V2)
    if declared != actual:
        raise RuntimeError(
            f"feature manifest mismatch: config declares {len(declared)} names, "
            f"datpark.features_v2 produces {len(actual)}; first difference at "
            f"{next((i for i, (a, b) in enumerate(zip(declared, actual)) if a != b), 'length')}")


def load_boosters() -> list[lgb.Booster]:
    paths = sorted((ASSETS / "models").glob("lgb_*.txt"))
    if not paths:
        raise FileNotFoundError("no LightGBM boosters in assets/models")
    return [lgb.Booster(model_file=str(p)) for p in paths]


def load_family(name: str, arch: str, stem_stride: int,
                device: torch.device) -> list[torch.nn.Module]:
    paths = sorted((ASSETS / "cnn" / name).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints in assets/cnn/{name}")
    nets = []
    for p in paths:
        net = BUILDERS[arch](stem_stride)
        net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        net.eval().to(device)
        for mod in net.modules():
            if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                assert not mod.training, f"BatchNorm must be in eval mode ({type(mod).__name__})"
        nets.append(net)
    return nets


# ---------------------------------------------------------------- TTA schemes
def _theta(flip_si=False, flip_rl=False, rot_deg=0.0, scale=1.0):
    """One deterministic view. Rotation is in the sagittal plane, matching training."""
    t = torch.zeros(1, 3, 4)
    a = np.deg2rad(rot_deg)
    c, s = float(np.cos(a)), float(np.sin(a))
    t[:, ROW_SI, ROW_SI] = c * scale
    t[:, ROW_SI, ROW_AP] = -s * scale
    t[:, ROW_AP, ROW_SI] = s * scale
    t[:, ROW_AP, ROW_AP] = c * scale
    t[:, ROW_RL, ROW_RL] = scale
    if flip_si:
        t[:, ROW_SI, :] *= -1
    if flip_rl:
        t[:, ROW_RL, :] *= -1
    return t


def views(scheme: str) -> list:
    """None means identity, which skips grid_sample entirely."""
    if scheme == "rl":
        # Exactly what `--tta-flip` applied during training: identity + one R-L mirror.
        # Verified equivalent to torch.flip(x, dims=[2]): on the siamese family the trainer's
        # own OOF and this theta-based path agreed to 0.0e+00.
        return [None, _theta(flip_rl=True)]
    if scheme == "si_rl":
        return [None, _theta(flip_si=True), _theta(flip_rl=True),
                _theta(flip_si=True, flip_rl=True)]
    if scheme == "si_rot_scale":
        v = [None, _theta(flip_si=True)]
        for r in (-7.0, 7.0):
            v += [_theta(rot_deg=r), _theta(flip_si=True, rot_deg=r)]
        for sc in (0.96, 1.04):
            v += [_theta(scale=sc), _theta(flip_si=True, scale=sc)]
        return v
    raise ValueError(f"unknown TTA scheme {scheme!r}")


# ------------------------------------------------------- per-scan preprocessing
_CFGS: dict[str, PreprocessConfig] | None = None


def _init_worker(cfgs: dict) -> None:
    global _CFGS
    _CFGS = cfgs


def _all_grids(uid: str):
    try:
        data, zooms = load_volume(NIFTI_DIR / f"{uid}.nii.gz")
        base = _CFGS["g64"]
        mask = brain_mask(data, base)
        centre = striatum_centre(data, mask, zooms, base)
        ref = reference_level(data, mask, base)
        out = {}
        for key, cfg in _CFGS.items():
            cube = _crop_resample(data, centre, zooms, cfg) / ref
            np.clip(cube, 0.0, cfg.clip, out=cube)
            out[key] = cube.astype(np.float32)
        out["feats"] = feature_matrix_v2(out["g64"][None], voxel_mm=base.voxel_mm,
                                         robust_midline=False)[0][0]
        # A scan can parse cleanly and still be garbage: one NaN voxel makes the
        # reference level fall back to 1.0, so the cube becomes raw counts and
        # saturates — measured to flip a normal scan from p=0.076 to p=0.674,
        # CONFIDENTLY WRONG, which costs more than the prevalence fallback would.
        # All-zero and constant volumes do the same. Detect and fall back instead.
        g = out.get("g96", next(iter(out.values())))
        if (not np.isfinite(g).all()) or float(g.std()) < 1e-3:
            return None
        return out
    except Exception:
        return None


# ------------------------------------------------------------------ prediction
def cnn_predict(cubes: np.ndarray, nets, scheme: str, device: torch.device,
                norm: str = "scale") -> np.ndarray:
    """Mean probability over (checkpoints x TTA views). Per-sample throughout."""
    vs = views(scheme)
    out = np.zeros(len(cubes), dtype=np.float64)
    x_all = torch.from_numpy(cubes / CUBE_SCALE).unsqueeze(1)
    if norm == "zscore":
        # ★ DOMAIN MATCHING, NOT PREPROCESSING. MedicalNet was pretrained on zero-mean /
        # unit-variance volumes; our cubes are mean +0.21 / std 0.12. measured the
        # 2x2: z-scoring HELPS the pretrained model (+0.0227) and HURTS a random-init one
        # (-0.0298), and on raw input the frozen encoder collapses to a near-constant function
        # (cosine similarity 0.9998 between different scans, 205/512 dead features). Getting
        # this wrong at inference silently destroys the family.
        # Statistics are PER SCAN, so rule 6 (test-sample independence) holds exactly.
        # Division by CUBE_SCALE above is mathematically redundant here (z-scoring is
        # scale-invariant) but kept so the order matches training byte for byte.
        # Use the vendored, CHUNKED implementation rather than re-deriving it here. The naive
        # one-liner (`(x_all.float() - mu) / (sd + 1e-5)` over the whole stack) was measured at
        # 3.45 GiB peak for 300 scans, extrapolating to ~16 GiB at n=1400 — on top of the g64
        # and g96 stacks already resident. datpark/med3d.py chunks precisely to avoid that and
        # its docstring records the same 11 GB near-miss during training. Verified bit-exact:
        # torch.equal(standardise_per_scan(x), naive(x)) is True.
        x_all = standardise_per_scan(x_all.float())
    elif norm != "scale":
        raise ValueError(f"unknown normalisation {norm!r}")
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for s in range(0, len(x_all), CNN_BATCH):
            xb = x_all[s : s + CNN_BATCH].to(device)
            acc = torch.zeros(len(xb), dtype=torch.float64)
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
                   else torch.autocast("cpu", enabled=False))
            with ctx:
                for v in vs:
                    z = xb if v is None else apply_affine(
                        xb, v.to(device).expand(len(xb), 3, 4))
                    if use_amp:
                        z = z.to(memory_format=torch.channels_last_3d)
                    for net in nets:
                        acc += torch.sigmoid(net(z).float()).double().cpu()
            out[s : s + len(xb)] = (acc / (len(vs) * len(nets))).numpy()
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def main() -> None:
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH)
    n = len(submission_format)
    logger.info("Loaded submission format.")

    raw = load_config()
    _assert_feature_manifest(raw)          # fail fast rather than predict on misaligned columns
    cfgs = {k: make_cfg(v) for k, v in raw["grids"].items()}
    w_cnn = float(raw["blend"]["w_cnn"])
    lo, hi = raw["clip"]
    fallback = float(raw["train_prevalence"])
    fam_spec = raw["cnn_families"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boosters = load_boosters()
    families = [
        (name, load_family(name, s["arch"], s["stem_stride"], device), s["grid"], s["tta"],
         s.get("norm", "scale"))
        for name, s in fam_spec.items()
    ]
    logger.info(
        "Loaded " + ", ".join(f"{len(nets)}x{name}[{tta}]" for name, nets, _, tta, _n in families)
        + f" and {len(boosters)} boosters on {device.type}; grids "
        + ", ".join(f"{k}={c.grid}^3@{c.voxel_mm}mm" for k, c in cfgs.items())
        + f"; blend w_cnn={w_cnn}")

    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    logger.info(f"Featurising with {workers} worker processes.")

    uids = list(submission_format["uid"])
    results: list[dict | None] = [None] * n
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(cfgs,)) as ex:
        try:
            for i, r in enumerate(ex.map(_all_grids, uids, chunksize=4)):
                results[i] = r
        except Exception as exc:
            # A worker killed by the OS (OOM, segfault) raises BrokenProcessPool out of the
            # map loop, NOT into the per-scan except below. Uncaught, that means no
            # submission.csv is written at all and the entire run scores nothing. Any scan not
            # yet filled stays None and takes the prevalence fallback — a far better outcome
            # than losing every prediction.
            # Deliberately reports only the exception TYPE, which is a property of this code. No
            # count, rate or identifier is emitted: those would be quantities derived from the
            # evaluation set, and nothing about the evaluation set belongs in a log.
            logger.warning(f"preprocessing pool ended early ({type(exc).__name__}); "
                           f"affected scans fall back to the prior")
    logger.info("Preprocessing complete.")

    preds = np.full(n, fallback, dtype=np.float64)
    ok = np.array([r is not None for r in results])
    if ok.any():
        good = [r for r in results if r is not None]
        feats = np.stack([r["feats"] for r in good])
        cubes = {k: np.stack([r[k] for r in good]) for k in cfgs}

        cnn_logits = [logit(cnn_predict(cubes[grid], nets, tta, device, norm))
                      for _, nets, grid, tta, norm in families]
        z_cnn = np.mean(cnn_logits, axis=0)

        p_sbr = np.mean([b.predict(feats) for b in boosters], axis=0)
        blended = w_cnn * z_cnn + (1.0 - w_cnn) * logit(p_sbr)
        preds[ok] = 1.0 / (1.0 + np.exp(-blended))
        logger.info("Inference complete.")

    submission_format["is_pathologic"] = np.clip(preds, lo, hi)
    submission_format.to_csv(WRITE_SUBMISSION_PATH, index=False)

    # No completion statistics are logged. A count of unprocessable scans, a fallback rate, or an
    # elapsed time are all quantities derived from the evaluation set — the elapsed time because it
    # scales with how many scans there are. Correctness does not depend on reporting them: every
    # failure path already resolves to the prior-probability fallback above, and the written CSV is
    # complete by construction.
    logger.success(f"Wrote predictions to {WRITE_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
