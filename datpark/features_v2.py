"""022: SBR v2 — multiscale regional features measured against a robust anatomical midline.

ADDITIVE — `datpark/features.py` is untouched and still serves the shipped SBR member. This module is a
parallel implementation; `sbr_v1_anchor` in the 022 sweep re-runs the ORIGINAL v1 features through the new
harness precisely so the harness can be proven equivalent before any v2 number is read.

── WHAT CHANGES FROM v1, AND WHY EACH PART ─────────────────────────────────────────────────
1. **The left/right plane.** v1 splits at the cube midpoint, which is the intensity-weighted centroid and
   is dragged toward the healthy side in asymmetric disease (corr +0.3124, p < 0.0001). v2
   splits at `midline.estimate_midline`, which uses fixed-count landmarks and an unweighted median so
   intensity cannot move the plane.

2. **Fixed VOLUMES instead of one scale.** v1 uses a single ~0.5 ml top-k mean. v2 measures at 0.5, 2, 5
   and 10 ml. A fixed volume is the right unit for a semi-quantitative binding ratio: 0.5 ml is close to
   the peak putaminal voxel and 10 ml approaches the whole striatum, so the *profile across scales* is
   itself a shape cue — a scan with normal peak uptake but a shrunken high-binding volume looks different
   from a uniformly reduced one, and a single scale cannot tell them apart.

3. **Three geometric features.** `abs_shift_mm`, `landmark_sep_mm`, `reliable`. Label-free and per-scan.
   ⚠ the pre-registration pre-specifies how to read these: a gain driven mainly by them means the model is using
   shift as a DISEASE PROXY rather than the corrected geometry as a better measurement. That is a
   different finding and must be reported as such, not as "the midline fix worked".

── WHAT DELIBERATELY DOES NOT CHANGE ───────────────────────────────────────────────────────
The anterior/posterior boundary stays at the core's own mid-j, exactly as v1. Introducing an AP
registration method at the same time would confound two geometry changes in one arm, and the whole point
of the descriptive arms is to keep effects attributable.

The feature ORDER is fixed by `FEATURE_NAMES_V2` and asserted at train and inference time. A silent
reordering between the two would be invisible and catastrophic.

Orientation (all 1362 scans verified RAS): axis 0 = R(+), axis 1 = A(+), axis 2 = S(+).
"""
from __future__ import annotations

import numpy as np

from datpark.midline import Midline, core_of, estimate_midline, k_for_volume, split_masks

VOLUMES_ML = (0.5, 2.0, 5.0, 10.0)          # frozen in the pre-registration
EPS = 1e-6

_GLOBAL = ["sbr_max", "sbr_mean", "vol_gt2", "vol_gt3", "vol_gt4", "extent_z", "spread"]
_PER_VOL = ["sbr_l", "sbr_r", "sbr_min_side", "asym_index",
            "sbr_ant_l", "sbr_post_l", "sbr_ant_r", "sbr_post_r",
            "pc_ratio_min", "pc_ratio_asym"]
_GEOM = ["abs_shift_mm", "landmark_sep_mm", "reliable"]


def _vol_tag(v: float) -> str:
    return f"{v:g}".replace(".", "p")


FEATURE_NAMES_V2 = (
    _GLOBAL
    + [f"{n}_v{_vol_tag(v)}" for v in VOLUMES_ML for n in _PER_VOL]
    + _GEOM
)


def _topk_mean(x: np.ndarray, k: int) -> float:
    """Mean of the k largest values. Identical to features.py:28 so v1 and v2 agree where they overlap."""
    if x.size == 0:
        return 0.0
    k = min(k, x.size)
    return float(np.partition(x.ravel(), -k)[-k:].mean())


def _regional(core: np.ndarray, left: np.ndarray, right: np.ndarray,
              k: int) -> list[float]:
    """The ten per-volume features at one scale, on a boolean-mask split.

    Masking rather than slicing is what lets the plane sit at a CONTINUOUS position: a slice index would
    force the midline back onto an integer voxel boundary and throw away most of the correction.
    """
    lv, rv = core[left], core[right]
    kk = max(4, k // 2)                     # per side, mirroring v1's k // 2
    sbr_l = _topk_mean(lv, kk) - 1.0
    sbr_r = _topk_mean(rv, kk) - 1.0
    denom = sbr_l + sbr_r
    asym = abs(sbr_l - sbr_r) / denom if denom > EPS else 0.0

    # anterior half ~ caudate, posterior half ~ putamen. Putaminal loss with relative caudate sparing is
    # the classic degenerative pattern, so the posterior/anterior ratio is the shape cue a bare SBR misses.
    mid_j = core.shape[1] // 2
    ant_mask = np.zeros(core.shape, dtype=bool)
    ant_mask[:, mid_j:, :] = True
    ka = max(2, k // 4)

    def ant_post(side: np.ndarray) -> tuple[float, float]:
        ant = _topk_mean(core[side & ant_mask], ka) - 1.0
        post = _topk_mean(core[side & ~ant_mask], ka) - 1.0
        return ant, post

    ant_l, post_l = ant_post(left)
    ant_r, post_r = ant_post(right)
    pc_l = post_l / ant_l if ant_l > EPS else 0.0
    pc_r = post_r / ant_r if ant_r > EPS else 0.0
    pc_denom = pc_l + pc_r

    return [sbr_l, sbr_r, min(sbr_l, sbr_r), asym,
            ant_l, post_l, ant_r, post_r,
            min(pc_l, pc_r),
            abs(pc_l - pc_r) / pc_denom if pc_denom > EPS else 0.0]


def striatal_features_v2(cube: np.ndarray, voxel_mm: float = 2.0,
                         robust_midline: bool = True) -> tuple[np.ndarray, Midline]:
    """Feature vector for one cube, plus the midline diagnostics used for reporting.

    `robust_midline=False` reproduces v1's plane (the cube midpoint) with the v2 feature set — the
    `sbr_v2_feat` attribution arm.
    """
    core = core_of(cube, voxel_mm)
    hg = core.shape[0]

    if robust_midline:
        ml = estimate_midline(core, voxel_mm)
    else:
        ml = Midline((hg - 1) / 2.0, 0.0, float("nan"), False, "midline disabled (control arm)")

    # v1 slices `core[:hg//2]` / `core[hg//2:]` — 15 voxels each side for hg=30 — so its dividing
    # BOUNDARY sits at (hg-1)/2 = 14.5, which is also the array's centre of symmetry. As a threshold in
    # `x < plane`, 14.5 and 15.0 select the identical voxels (integer voxel centres quantise the plane,
    # see test_split_is_QUANTISED_to_one_voxel...), so either value reproduces v1's split. 14.5 is used
    # for consistency with `midline.estimate_midline`'s zero-shift reference.
    plane = ml.x if robust_midline else (hg - 1) / 2.0
    left, right = split_masks(core.shape, plane)

    k_half = k_for_volume(0.5, voxel_mm)
    sbr_max = float(core.max()) - 1.0
    sbr_mean = _topk_mean(core, k_half) - 1.0
    vols = [float((core > t).mean()) for t in (2.0, 3.0, 4.0)]
    hot = core > 2.0
    if hot.any():
        idx = np.array(np.nonzero(hot), dtype=np.float32)
        extent_z = float(idx[2].max() - idx[2].min()) * voxel_mm
        spread = float(idx.std(axis=1).mean()) * voxel_mm
    else:
        extent_z, spread = 0.0, 0.0

    feats: list[float] = [sbr_max, sbr_mean, *vols, extent_z, spread]
    for v in VOLUMES_ML:
        feats.extend(_regional(core, left, right, k_for_volume(v, voxel_mm)))
    feats.extend([abs(ml.shift_mm),
                  ml.separation_mm if np.isfinite(ml.separation_mm) else 0.0,
                  float(ml.reliable)])

    out = np.asarray(feats, dtype=np.float32)
    if out.size != len(FEATURE_NAMES_V2):
        raise AssertionError(f"produced {out.size} features, manifest declares "
                             f"{len(FEATURE_NAMES_V2)} — order/length drift")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), ml


def feature_matrix_v2(cubes: np.ndarray, voxel_mm: float = 2.0,
                      robust_midline: bool = True):
    """(N, F) features and the per-scan midline records. One scan never depends on another."""
    rows, mls = [], []
    for c in cubes:
        f, ml = striatal_features_v2(c, voxel_mm, robust_midline)
        rows.append(f)
        mls.append(ml)
    return np.stack(rows), mls


# ---------------------------------------------------------------------------------------------------
# v1's TWENTY features, computable at an ARBITRARY plane.
#
# Needed for the two attribution arms in the pre-registration: `sbr_v2_mid` (robust plane, v1 features) isolates
# the MIDLINE, and `sbr_v1_anchor` (v1 plane, v1 features) proves the harness is equivalent to
# `features.py` before any v2 number is read. Without the anchor, a v2 result could be a harness
# artefact and be indistinguishable from the hypothesis.
#
# Reproducing v1 EXACTLY is possible with masks because every v1 quantity is a top-k mean over a SET of
# voxels: `core[:hg//2]` and `core[mask]` present the same multiset to `_topk_mean`, and slicing
# `side[:, :mid]` is the same set as `mask & (j < mid)`. Asserted bit-for-bit in test_midline.py.
# ---------------------------------------------------------------------------------------------------

FEATURE_NAMES_V1 = [
    "sbr_max", "sbr_mean", "sbr_l", "sbr_r", "sbr_min_side", "sbr_max_side",
    "asym_index", "sbr_ant_l", "sbr_post_l", "sbr_ant_r", "sbr_post_r",
    "pc_ratio_l", "pc_ratio_r", "pc_ratio_min", "pc_ratio_asym",
    "vol_gt2", "vol_gt3", "vol_gt4", "extent_z", "spread",
]


def striatal_features_v1_at_plane(cube: np.ndarray, voxel_mm: float = 2.0,
                                  robust_midline: bool = False) -> tuple[np.ndarray, Midline]:
    """v1's exact 20 features, but split at a chosen plane.

    `robust_midline=False` must reproduce `datpark.features.striatal_features` bit-for-bit.
    """
    core = core_of(cube, voxel_mm)
    hg = core.shape[0]

    if robust_midline:
        ml = estimate_midline(core, voxel_mm)
        plane = ml.x
    else:
        ml = Midline((hg - 1) / 2.0, 0.0, float("nan"), False, "v1 plane (control)")
        plane = (hg - 1) / 2.0          # `x < 14.5` selects 0..14 == v1's core[:hg//2]

    left, right = split_masks(core.shape, plane)
    k = k_for_volume(0.5, voxel_mm)     # v1's single scale

    sbr_max = float(core.max()) - 1.0
    sbr_mean = _topk_mean(core, k) - 1.0
    sbr_l = _topk_mean(core[left], k // 2) - 1.0
    sbr_r = _topk_mean(core[right], k // 2) - 1.0
    denom = sbr_l + sbr_r
    asym = abs(sbr_l - sbr_r) / denom if denom > EPS else 0.0

    mid_j = core.shape[1] // 2
    ant_mask = np.zeros(core.shape, dtype=bool)
    ant_mask[:, mid_j:, :] = True

    def ant_post(side: np.ndarray) -> tuple[float, float]:
        post = _topk_mean(core[side & ~ant_mask], k // 4) - 1.0
        ant = _topk_mean(core[side & ant_mask], k // 4) - 1.0
        return ant, post

    ant_l, post_l = ant_post(left)
    ant_r, post_r = ant_post(right)
    pc_l = post_l / ant_l if ant_l > EPS else 0.0
    pc_r = post_r / ant_r if ant_r > EPS else 0.0
    pc_denom = pc_l + pc_r

    vols = [float((core > t).mean()) for t in (2.0, 3.0, 4.0)]
    hot = core > 2.0
    if hot.any():
        idx = np.array(np.nonzero(hot), dtype=np.float32)
        extent_z = float(idx[2].max() - idx[2].min()) * voxel_mm
        spread = float(idx.std(axis=1).mean()) * voxel_mm
    else:
        extent_z, spread = 0.0, 0.0

    out = np.array(
        [sbr_max, sbr_mean, sbr_l, sbr_r, min(sbr_l, sbr_r), max(sbr_l, sbr_r), asym,
         ant_l, post_l, ant_r, post_r, pc_l, pc_r, min(pc_l, pc_r),
         abs(pc_l - pc_r) / pc_denom if pc_denom > EPS else 0.0,
         *vols, extent_z, spread], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), ml


def feature_matrix_v1_at_plane(cubes: np.ndarray, voxel_mm: float = 2.0,
                               robust_midline: bool = False):
    rows, mls = [], []
    for c in cubes:
        f, ml = striatal_features_v1_at_plane(c, voxel_mm, robust_midline)
        rows.append(f)
        mls.append(ml)
    return np.stack(rows), mls
