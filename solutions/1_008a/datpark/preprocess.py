"""Canonicalise a DaT scan into a fixed grid, comparably across centres.

Every step here exists to close a leakage channel measured in EDA.md:

  * resample to isotropic MILLIMETRES via the header spacing
        -> kills the 3.6x field-of-view scale confound that tracks the scanner
  * crop a tight physical box on the STRIATUM
        -> removes the edge/streak artefact shortcut (2.7% of scans, 78%
           abnormal, p=0.004) by construction rather than by hoping
  * divide by a REFERENCE REGION (brain excluding the specific-binding tail)
        -> erases the 16,000x raw-count centre fingerprint, and is the same
           denominator the clinical striatal binding ratio uses

Everything is computed from a single scan in isolation: no dataset statistics,
no cross-sample information. That satisfies the competition's test-sample
independence rule and makes training deterministic with respect to the test set.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PreprocessConfig:
    voxel_mm: float = 2.0        # output isotropic voxel size
    crop_mm: float = 128.0       # physical side of the cropped cube
    brain_frac: float = 0.15     # brain mask threshold, as a fraction of p99.5
    brain_pct: float = 99.5      # percentile defining "bright" for the mask
    hot_pct: float = 99.0        # percentile (within brain) defining striatal voxels
    search_mm: float = 60.0      # max distance from brain centre to look for striatum
    ref_exclude_pct: float = 95.0  # drop the top 5% of brain voxels before averaging
    clip: float = 12.0           # cap on normalised intensity (ref-multiples)

    @property
    def grid(self) -> int:
        return int(round(self.crop_mm / self.voxel_mm))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grid"] = self.grid
        return d


DEFAULT = PreprocessConfig()


# --------------------------------------------------------------------------
def load_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (float32 volume, voxel sizes in mm). Import kept local so that
    callers who already have an array need not depend on nibabel."""
    import nibabel as nib

    img = nib.load(str(path))
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    # ⚠ CLAMP. `_crop_resample` computes half_vox = (crop_mm/2) / zooms and pads to that size,
    # so a corrupt pixdim turns into an allocation request. Measured: pixdim 0.05 mm asks for
    # 62.6 GiB, 0.01 mm for 7.63 TiB. On this 31 GB box numpy raises MemoryError and the scan
    # falls back cleanly, but the competition container has 220 GB — a 62 GiB allocation
    # SUCCEEDS there, and 8 concurrent workers exhaust the machine, killing the pool and
    # producing no submission.csv at all. Every one of the 1362 training scans lies in
    # 1.37-4.42 mm, so this clamp is a no-op on real data and a guard against a header bug.
    zooms = np.where(np.isfinite(zooms) & (zooms > 0), zooms, 1.0)
    zooms = np.clip(zooms, 0.5, 20.0)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3])
    return data, zooms


def brain_mask(data: np.ndarray, cfg: PreprocessConfig = DEFAULT) -> np.ndarray:
    """Largest connected bright component. SPECT background is near zero, so a
    relative threshold is enough; the connectivity step drops scatter specks."""
    hi = float(np.percentile(data, cfg.brain_pct))
    if hi <= 0:
        return np.ones(data.shape, bool)
    mask = data > cfg.brain_frac * hi
    if mask.sum() < 50:
        return np.ones(data.shape, bool)
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = lab == (int(np.argmax(sizes)) + 1)
    return mask


def striatum_centre(
    data: np.ndarray, mask: np.ndarray, zooms: np.ndarray, cfg: PreprocessConfig = DEFAULT
) -> np.ndarray:
    """Intensity-weighted centroid of the brightest brain voxels.

    In DaT imaging the striatum IS the brightest structure, so a high percentile
    localises it directly. The search is restricted to a sphere around the brain
    centre so that an out-of-brain streak artefact can never capture the crop --
    which matters, because those artefacts are label-correlated.
    """
    brain_com = np.asarray(ndimage.center_of_mass(mask), dtype=np.float64)

    grids = np.ogrid[tuple(slice(0, s) for s in data.shape)]
    d2 = sum((((g - c) * z) ** 2) for g, c, z in zip(grids, brain_com, zooms))
    near = d2 <= cfg.search_mm**2

    vals = data[mask]
    thr = float(np.percentile(vals, cfg.hot_pct)) if vals.size else 0.0
    hot = mask & near & (data > thr)
    if hot.sum() < 8:  # e.g. profoundly reduced uptake -> fall back to the brain centre
        return brain_com

    w = data[hot].astype(np.float64)
    idx = np.array(np.nonzero(hot), dtype=np.float64)
    return (idx * w).sum(axis=1) / w.sum()


def reference_level(
    data: np.ndarray, mask: np.ndarray, cfg: PreprocessConfig = DEFAULT
) -> float:
    """Non-specific binding level: median brain voxel after removing the
    specific-binding tail. This is the SBR denominator, computed without an
    atlas."""
    vals = data[mask]
    if vals.size < 32:
        vals = data[data > 0]
    if vals.size == 0:
        return 1.0
    cut = float(np.percentile(vals, cfg.ref_exclude_pct))
    body = vals[vals <= cut]
    ref = float(np.median(body)) if body.size else float(np.median(vals))
    return ref if ref > 1e-6 else 1.0


def _crop_resample(
    data: np.ndarray, centre: np.ndarray, zooms: np.ndarray, cfg: PreprocessConfig
) -> np.ndarray:
    """Crop `crop_mm` around `centre` (padding as needed) and resample to grid^3."""
    half_vox = (cfg.crop_mm / 2.0) / zooms
    lo = np.floor(centre - half_vox).astype(int)
    hi = np.ceil(centre + half_vox).astype(int)

    pad_lo = np.maximum(0, -lo)
    pad_hi = np.maximum(0, hi - np.asarray(data.shape))
    if pad_lo.any() or pad_hi.any():
        data = np.pad(data, list(zip(pad_lo, pad_hi)))
        lo = lo + pad_lo
        hi = hi + pad_lo
    crop = data[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]

    g = cfg.grid
    out = ndimage.zoom(crop, np.array([g, g, g]) / np.asarray(crop.shape), order=1)
    out = out[:g, :g, :g]
    if out.shape != (g, g, g):
        out = np.pad(out, [(0, g - s) for s in out.shape])
    return out.astype(np.float32)


def preprocess_array(
    data: np.ndarray, zooms: np.ndarray, cfg: PreprocessConfig = DEFAULT
) -> tuple[np.ndarray, dict]:
    """Canonicalise an already-loaded volume. Returns (grid^3 float32, meta).

    Output units are multiples of the non-specific binding level, so a striatal
    voxel typically lands around 2-8 and background around 1.
    """
    zooms = np.asarray(zooms, dtype=np.float64)
    mask = brain_mask(data, cfg)
    centre = striatum_centre(data, mask, zooms, cfg)
    ref = reference_level(data, mask, cfg)

    cube = _crop_resample(data, centre, zooms, cfg) / ref
    np.clip(cube, 0.0, cfg.clip, out=cube)

    meta = {
        "src_shape": [int(s) for s in data.shape],
        "src_zooms": [round(float(z), 4) for z in zooms],
        "grid": cfg.grid,
        "voxel_mm": cfg.voxel_mm,
        "crop_mm": cfg.crop_mm,
        "reference_level": round(float(ref), 4),
        "centre_vox": [round(float(c), 2) for c in centre],
        "brain_voxels": int(mask.sum()),
    }
    return cube, meta


def preprocess(
    path: str | Path, cfg: PreprocessConfig = DEFAULT
) -> tuple[np.ndarray, dict]:
    """Canonicalise one scan from disk."""
    data, zooms = load_volume(path)
    return preprocess_array(data, zooms, cfg)
