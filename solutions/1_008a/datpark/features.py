"""Semi-quantitative features from a canonicalised cube — the classical read.

Clinicians grade DaT scans on the shape and symmetry of striatal uptake, backed
by binding ratios. Cubes from datpark.preprocess are already expressed in
multiples of the non-specific reference level, so a "specific binding ratio"
here is just (uptake - 1).

These features are deliberately interpretable: they are diverse from a CNN,
naturally well calibrated, and they make the eventual solution explainable,
which matters for the BOAS open-algorithm indexing.

Orientation (all volumes are RAS, verified across all 1362 scans):
    axis 0 = i = Right(+)      axis 1 = j = Anterior(+)      axis 2 = k = Superior(+)
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "sbr_max", "sbr_mean", "sbr_l", "sbr_r", "sbr_min_side", "sbr_max_side",
    "asym_index", "sbr_ant_l", "sbr_post_l", "sbr_ant_r", "sbr_post_r",
    "pc_ratio_l", "pc_ratio_r", "pc_ratio_min", "pc_ratio_asym",
    "vol_gt2", "vol_gt3", "vol_gt4", "extent_z", "spread",
]


def _topk_mean(x: np.ndarray, k: int) -> float:
    if x.size == 0:
        return 0.0
    k = min(k, x.size)
    return float(np.partition(x.ravel(), -k)[-k:].mean())


def striatal_features(cube: np.ndarray, voxel_mm: float = 2.0) -> np.ndarray:
    """Compute the feature vector for one canonicalised cube."""
    cube = np.asarray(cube, dtype=np.float32)
    g = cube.shape[0]
    c = g // 2

    # central 60 mm box: comfortably contains both striata, excludes cortex rim
    half = max(4, int(round(30.0 / voxel_mm)))
    core = cube[c - half : c + half, c - half : c + half, c - half : c + half]
    hg = core.shape[0]

    right = core[hg // 2 :]          # i increasing = subject's right
    left = core[: hg // 2]
    k = max(8, int(round(0.5 / voxel_mm**3 * 1000)))  # ~0.5 mL of peak voxels

    sbr_max = float(core.max()) - 1.0
    sbr_mean = _topk_mean(core, k) - 1.0
    sbr_l = _topk_mean(left, k // 2) - 1.0
    sbr_r = _topk_mean(right, k // 2) - 1.0
    lo, hi = min(sbr_l, sbr_r), max(sbr_l, sbr_r)
    denom = sbr_l + sbr_r
    asym = abs(sbr_l - sbr_r) / denom if denom > 1e-6 else 0.0

    # anterior half ~ caudate, posterior half ~ putamen. Putaminal loss with
    # relative caudate sparing is the classic degenerative pattern, so the
    # posterior/anterior ratio is the shape cue that a bare SBR misses.
    def ant_post(side: np.ndarray) -> tuple[float, float]:
        mid = side.shape[1] // 2
        post = _topk_mean(side[:, :mid], k // 4) - 1.0
        ant = _topk_mean(side[:, mid:], k // 4) - 1.0
        return ant, post

    ant_l, post_l = ant_post(left)
    ant_r, post_r = ant_post(right)
    pc_l = post_l / ant_l if ant_l > 1e-6 else 0.0
    pc_r = post_r / ant_r if ant_r > 1e-6 else 0.0
    pc_denom = pc_l + pc_r

    # how much tissue still shows specific binding, and how compact it is
    vols = [float((core > t).mean()) for t in (2.0, 3.0, 4.0)]
    hot = core > 2.0
    if hot.any():
        idx = np.array(np.nonzero(hot), dtype=np.float32)
        extent_z = float(idx[2].max() - idx[2].min()) * voxel_mm
        spread = float(idx.std(axis=1).mean()) * voxel_mm
    else:
        extent_z, spread = 0.0, 0.0

    return np.array(
        [sbr_max, sbr_mean, sbr_l, sbr_r, lo, hi, asym,
         ant_l, post_l, ant_r, post_r, pc_l, pc_r, min(pc_l, pc_r),
         abs(pc_l - pc_r) / pc_denom if pc_denom > 1e-6 else 0.0,
         *vols, extent_z, spread],
        dtype=np.float32,
    )


def feature_matrix(cubes: np.ndarray, voxel_mm: float = 2.0) -> np.ndarray:
    return np.stack([striatal_features(c, voxel_mm) for c in cubes])
