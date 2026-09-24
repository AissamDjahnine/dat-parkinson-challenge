"""Robust anatomical midline from bilateral striatal landmarks.

Not used by this submission: the estimator is disabled, so the left/right dividing plane is the cube
midpoint. The module is imported for its geometry helpers (`core_of`, `k_for_volume`, `split_masks`).

THE MEASUREMENT PROBLEM IT ADDRESSES. The cube is centred on an intensity-weighted centroid of the
brightest brain voxels. In asymmetric disease the affected putamen is dimmer, so it contributes fewer and
lower-weighted voxels, and the centroid is pulled toward the healthier side. Dividing left from right at
the cube midpoint therefore places the plane slightly off the anatomical midline, by an amount that grows
with the asymmetry. On this cohort the signed centroid offset correlates with the signed uptake asymmetry
at +0.31 in abnormal scans (p < 1e-4), and the mean displacement is 1.3 mm.

THE ESTIMATOR. In each preliminary hemisphere, take a FIXED VOLUME (2 ml) of the hottest voxels and take
the UNWEIGHTED MEDIAN of their right-left coordinates; place the plane midway between the two landmarks.
A fixed count means a dim hemisphere contributes as many voxels as a bright one; an unweighted median
means intensity cannot move the plane. A fixed intensity threshold, or an intensity-weighted centroid of
the same voxels, would reintroduce the bias the estimator exists to remove. The median is also robust to
stray hot voxels in a way a mean is not.

WHY THE PRELIMINARY SPLIT DOES NOT POISON THE RESULT. Locating the landmarks needs hemispheres before the
midline exists, so it uses the cube midpoint. A sub-voxel error cannot move a striatum to the wrong side
when the two structures lie ~25 mm apart, and the 12-40 mm separation check rejects the estimate if it
ever did.

WHY IT IS DISABLED. Enabling it was worth -0.0003 against the same ensemble. The plane is quantised to one
2 mm voxel while the displacement averages 1.3 mm, and the regional features are top-k means whose hottest
voxels lie ~12 mm from the plane, so moving the plane barely changes which voxels they select.

Every validity guard returns the cube midpoint, which is exactly the geometry used when the models were
fitted, so no scan is handled differently from the fitting configuration. A deterministic function of a
single scan: no labels, no cohort statistics, no pooling across scans.

Orientation: volumes are canonicalised to RAS, so axis 0 = R(+), axis 1 = A(+), axis 2 = S(+).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- fixed geometric constants; not tuned against any held-out score ----
LANDMARK_ML = 2.0        # fixed volume of hottest voxels per preliminary hemisphere
MAX_SHIFT_MM = 8.0       # beyond this the estimate is not credible for a 60 mm core
SEP_MIN_MM = 12.0        # two striata closer than this means the split captured one structure
SEP_MAX_MM = 40.0        # further apart than this is not striatal anatomy in a 60 mm box
MIN_BINDING = 0.25       # median specific binding required in BOTH landmark sets
CORE_MM = 60.0


@dataclass(frozen=True)
class Midline:
    """`x` is in CORE voxel coordinates, so `features_v2` can compare voxel indices to it directly."""
    x: float
    shift_mm: float
    separation_mm: float
    reliable: bool
    reason: str = "ok"


def k_for_volume(vol_ml: float, voxel_mm: float) -> int:
    """Voxel count for a fixed volume. Same convention as features.py:48, so 0.5 ml -> 62 at 2 mm."""
    return max(8, int(round(vol_ml * 1000.0 / voxel_mm**3)))


def core_of(cube: np.ndarray, voxel_mm: float = 2.0) -> np.ndarray:
    """The central 60 mm box — identical slicing to features.py:42-43."""
    cube = np.asarray(cube, dtype=np.float32)
    g = cube.shape[0]
    c = g // 2
    half = max(4, int(round(CORE_MM / 2.0 / voxel_mm)))
    return cube[c - half:c + half, c - half:c + half, c - half:c + half]


def _landmark_median_x(sb: np.ndarray, x_index: np.ndarray, k: int) -> tuple[float, float]:
    """Unweighted median R-L coordinate of the hottest `k` voxels, and their median binding."""
    flat = sb.ravel()
    if flat.size == 0:
        return float("nan"), 0.0
    k = min(k, flat.size)
    sel = np.argpartition(flat, -k)[-k:]
    return float(np.median(x_index.ravel()[sel])), float(np.median(flat[sel]))


def estimate_midline(core: np.ndarray, voxel_mm: float = 2.0) -> Midline:
    """Robust R-L midline of one cube's core. Falls back to the cube midpoint on any failure."""
    core = np.asarray(core, dtype=np.float32)
    hg = core.shape[0]
    # ⚠ HALF-VOXEL BOOKKEEPING, worked out explicitly because it cost two wrong turns.
    # Voxel CENTRES are integers 0..hg-1, so the array's centre of symmetry is (hg-1)/2 = 14.5 for
    # hg=30. `features.py:46-47` splits `core[:15]` / `core[15:]` — 15 voxels each side, so its dividing
    # boundary is ALSO at 14.5. v1's plane is therefore already symmetric, and 14.5 is the correct
    # zero-shift reference. It is also directly usable as a threshold: `x < 14.5` selects 0..14, exactly
    # v1's left slice.
    # A first attempt "corrected" this to hg//2 = 15.0 after a symmetric phantom reported +1.0 mm. That
    # was the wrong fix: the phantom was at fault, placing its blobs symmetric about cube voxel 32 when
    # the core window's symmetry centre is cube 31.5. Both are recorded in test_midline.py.
    centre = (hg - 1) / 2.0
    fallback = lambda why: Midline(centre, 0.0, float("nan"), False, why)  # noqa: E731

    if core.size == 0 or not np.isfinite(core).all():
        return fallback("non-finite or empty core")

    # cubes are multiples of the per-scan non-specific reference, so specific binding is (uptake - 1)
    sb = np.maximum(core - 1.0, 0.0)
    if sb.max() <= 0.0:
        return fallback("no specific binding anywhere")

    x_index = np.broadcast_to(np.arange(hg, dtype=np.float32)[:, None, None], core.shape)
    mid = hg // 2
    k = k_for_volume(LANDMARK_ML, voxel_mm)

    # preliminary hemispheres at the CURRENT (biased) midpoint — see the module docstring
    xl, bl = _landmark_median_x(sb[:mid], x_index[:mid], k)
    xr, br = _landmark_median_x(sb[mid:], x_index[mid:], k)

    if not (np.isfinite(xl) and np.isfinite(xr)):
        return fallback("a landmark median was not finite")
    if bl < MIN_BINDING or br < MIN_BINDING:
        return fallback(f"landmark binding too low (l={bl:.3f}, r={br:.3f})")

    sep_mm = abs(xr - xl) * voxel_mm
    if not (SEP_MIN_MM <= sep_mm <= SEP_MAX_MM):
        return fallback(f"landmark separation {sep_mm:.1f} mm outside [{SEP_MIN_MM}, {SEP_MAX_MM}]")

    x = (xl + xr) / 2.0
    shift_mm = (x - centre) * voxel_mm
    if abs(shift_mm) > MAX_SHIFT_MM:
        return fallback(f"shift {shift_mm:+.1f} mm exceeds +/-{MAX_SHIFT_MM}")

    return Midline(float(x), float(shift_mm), float(sep_mm), True, "ok")


def split_masks(shape: tuple[int, ...], midline_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean left/right masks from a CONTINUOUS plane: left = x < midline, right = x >= midline.

    Assignment is by physical coordinate — no interpolation and no voxel shifting. That is deliberate for
    a feature-only experiment: resampling the volume would mix an interpolation change into the SBR branch
    and confound it with the midline change being measured.
    """
    hg = shape[0]
    x = np.broadcast_to(np.arange(hg, dtype=np.float32)[:, None, None], shape)
    left = x < midline_x
    return left, ~left
