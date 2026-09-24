"""Preprocessing contracts — the rule-compliance ones matter most.

Competition rules 6/7 require each test sample to be processed independently and
training to be deterministic with respect to the test set. `preprocess_array`
satisfies that by computing every statistic from a single scan, and these tests
pin that property numerically rather than trusting the docstring.

All fixtures are synthetic. No competition data is read, so the suite runs on a
bare checkout and no scan data enters any log.
"""

from __future__ import annotations

import numpy as np

from datpark.preprocess import (
    DEFAULT,
    PreprocessConfig,
    brain_mask,
    preprocess_array,
    reference_level,
)


def _phantom(shape=(64, 64, 48), zooms=(3.0, 3.0, 3.5), scale=1000.0, seed=0):
    """A crude DaT-like volume: warm brain ball plus two bright striatal blobs."""
    rng = np.random.default_rng(seed)
    v = np.zeros(shape, dtype=np.float32)
    gz, gy, gx = np.mgrid[: shape[0], : shape[1], : shape[2]]
    c = np.array(shape) / 2.0
    mm = [((g - ci) * z) for g, ci, z in zip((gz, gy, gx), c, zooms)]
    r = np.sqrt(sum(m ** 2 for m in mm))
    v[r < 80.0] = 1.0                                     # brain
    for off in (-12.0, 12.0):                             # left/right striatum
        d = np.sqrt((mm[0] - off) ** 2 + (mm[1] - 8.0) ** 2 + mm[2] ** 2)
        v[d < 10.0] = 5.0
    v = v * scale + rng.normal(0, scale * 0.01, shape).astype(np.float32)
    return np.clip(v, 0, None), np.array(zooms)


def test_output_shape_is_config_grid():
    for voxel_mm, grid in ((2.0, 64), (1.3333, 96), (1.0, 128)):
        cfg = PreprocessConfig(voxel_mm=voxel_mm)
        assert cfg.grid == grid
        cube, _ = preprocess_array(*_phantom(), cfg=cfg)
        assert cube.shape == (grid, grid, grid)


def test_deterministic_same_input_same_output():
    """Rule 7: no randomness anywhere in the path."""
    data, zooms = _phantom()
    a, ma = preprocess_array(data, zooms)
    b, mb = preprocess_array(data, zooms)
    assert np.array_equal(a, b)
    assert ma == mb


def test_independent_of_other_scans():
    """Rule 6: a scan's cube must not depend on any other scan.

    Processing the same volume alone and interleaved with very differently scaled
    volumes must give identical output, since no dataset statistic exists.
    """
    data, zooms = _phantom(scale=1000.0)
    alone, _ = preprocess_array(data, zooms)
    for other_scale in (1.0, 50_000.0):
        other, oz = _phantom(scale=other_scale, seed=1)
        preprocess_array(other, oz)
        again, _ = preprocess_array(data, zooms)
        assert np.array_equal(alone, again), "cube changed after seeing another scan"


def test_intensity_scale_invariance():
    """The 16,000x raw-count centre fingerprint must be erased.

    Reference-region normalisation is a division, so multiplying the input by any
    constant must leave the cube unchanged up to numerical noise.
    """
    data, zooms = _phantom(scale=1000.0)
    base, mb = preprocess_array(data, zooms)
    for k in (0.01, 100.0):
        scaled, ms = preprocess_array(data * k, zooms)
        assert np.allclose(base, scaled, atol=2e-3), f"not invariant at k={k}"
        assert np.isclose(ms["reference_level"], mb["reference_level"] * k, rtol=1e-3)


def test_physical_scale_invariance():
    """FOV/voxel-size confound: same anatomy at different sampling -> same cube.

    Cropping a fixed millimetre box using header zooms is the whole point; if this
    fails, voxel-count resizing has crept back in.
    """
    fine, zf = _phantom(shape=(96, 96, 72), zooms=(2.0, 2.0, 2.33))
    coarse, zc = _phantom(shape=(48, 48, 36), zooms=(4.0, 4.0, 4.67))
    a, _ = preprocess_array(fine, zf)
    b, _ = preprocess_array(coarse, zc)
    # crude phantoms differ in sampling detail, so compare structure not voxels
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert corr > 0.85, f"physical-scale invariance weak: corr={corr:.3f}"


def test_values_are_clipped_reference_multiples():
    cube, meta = preprocess_array(*_phantom())
    assert cube.min() >= 0.0
    assert cube.max() <= DEFAULT.clip + 1e-6
    assert meta["reference_level"] > 0
    # background sits near 1 (a multiple of the non-specific level), striatum above
    assert 0.5 < float(np.median(cube[cube > 0.1])) < 4.0


def test_no_nan_or_inf():
    for scale in (1.0, 1e5):
        cube, _ = preprocess_array(*_phantom(scale=scale))
        assert np.isfinite(cube).all()


def test_anisotropic_zooms_handled():
    """94.3% of scans are isotropic; the rest must not silently distort."""
    cube, _ = preprocess_array(*_phantom(shape=(64, 64, 30), zooms=(3.0, 3.0, 6.0)))
    assert cube.shape == (DEFAULT.grid,) * 3
    assert np.isfinite(cube).all()


def test_brain_mask_is_nonempty_and_bounded():
    data, _ = _phantom()
    m = brain_mask(data)
    frac = float(m.mean())
    assert 0.0 < frac < 0.5, f"brain mask covers {frac:.3f} of the volume"


def test_reference_level_excludes_the_specific_binding_tail():
    """Averaging the whole brain would let the striatum set its own denominator."""
    data, _ = _phantom()
    m = brain_mask(data)
    ref = reference_level(data, m)
    assert ref < float(data[m].mean()) * 1.05
    assert ref > 0


def test_config_roundtrip():
    cfg = PreprocessConfig(voxel_mm=1.3333)
    d = cfg.to_dict()
    assert d["grid"] == 96 and d["voxel_mm"] == 1.3333
    assert PreprocessConfig(**{k: v for k, v in d.items() if k != "grid"}) == cfg
