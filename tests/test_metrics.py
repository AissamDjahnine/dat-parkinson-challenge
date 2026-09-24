"""Metric contracts. Log loss is the competition score, so it must be exact."""

from __future__ import annotations

import numpy as np

from datpark.metrics import clipped, expected_calibration_error, report, score


def test_log_loss_matches_closed_form():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.6, 0.3, 0.8])
    want = float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())
    assert abs(score(y, p)["log_loss"] - want) < 1e-9


def test_clipping_prevents_infinite_loss():
    """A single confident mistake must cost a lot, but stay finite."""
    y = np.array([0, 1])
    for p in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        s = score(y, p)
        assert np.isfinite(s["log_loss"])
    assert clipped(np.array([0.0, 1.0])).min() > 0
    assert clipped(np.array([0.0, 1.0])).max() < 1


def test_perfect_and_base_rate_predictions():
    y = np.array([0, 0, 1, 1])
    assert score(y, np.array([0.001, 0.001, 0.999, 0.999]))["log_loss"] < 0.01
    base = score(y, np.full(4, 0.5))
    assert abs(base["log_loss"] - np.log(2)) < 1e-6
    assert abs(base["log_loss_baseline"] - np.log(2)) < 1e-6


def test_baseline_is_the_prevalence_predictor():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    s = score(y, rng.random(500))
    prev = y.mean()
    want = float(-(prev * np.log(prev) + (1 - prev) * np.log1p(-prev)))
    assert abs(s["log_loss_baseline"] - want) < 1e-6


def test_auroc_is_nan_for_single_class():
    s = score(np.zeros(10, dtype=int), np.linspace(0.1, 0.9, 10))
    assert np.isnan(s["auroc"])


def test_auroc_direction():
    y = np.array([0, 0, 1, 1])
    assert score(y, np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0
    assert score(y, np.array([0.9, 0.8, 0.2, 0.1]))["auroc"] == 0.0


def test_ece_is_zero_for_perfect_calibration():
    """Predict the true rate everywhere -> nothing to correct."""
    n = 4000
    rng = np.random.default_rng(0)
    p = np.full(n, 0.3)
    y = (rng.random(n) < 0.3).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_ece_flags_gross_miscalibration():
    y = np.zeros(1000, dtype=int)
    assert expected_calibration_error(y, np.full(1000, 0.9)) > 0.8


def test_ece_bounds():
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = rng.random(300)
        y = (rng.random(300) < p).astype(int)
        e = expected_calibration_error(y, p)
        assert 0.0 <= e <= 1.0


def test_score_keys_present():
    s = score(np.array([0, 1]), np.array([0.4, 0.6]))
    for k in ("log_loss", "auroc", "brier", "ece", "n", "prevalence",
              "mean_pred", "log_loss_baseline"):
        assert k in s, f"missing metric {k}"
    assert s["n"] == 2


def test_report_runs_with_and_without_groups():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.5).astype(int)
    p = rng.random(200)
    assert report(y, p, title="no groups") is not None
    g = np.array([f"c{i % 4}" for i in range(200)])
    out = report(y, p, g, title="grouped")
    assert len(out) >= 4
