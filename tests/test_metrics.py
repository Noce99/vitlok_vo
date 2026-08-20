"""Error metrics must be zero on a perfect estimate and grow with real error."""

import numpy as np

from src.metrics import compute_all, compute_ate, compute_fpe, compute_rte


def _track(n=500, duration=100.0):
    t = np.linspace(0, duration, n)
    return np.column_stack([t, np.linspace(0, 100, n), np.zeros(n)])


def test_perfect_estimate_scores_zero():
    truth = _track()
    assert compute_ate(truth, truth.copy())["rmse"] < 1e-9
    assert compute_rte(truth, truth.copy(), delta=50)["rmse"] < 1e-9
    assert compute_fpe(truth, truth.copy())["final_distance_m"] < 1e-9


def test_constant_offset_shows_in_ate_but_not_rte():
    """A rigid offset moves every point equally, so relative motion is untouched."""
    truth = _track()
    shifted = truth.copy()
    shifted[:, 2] += 5.0

    assert compute_ate(truth, shifted)["rmse"] == np.float64(5.0)
    assert compute_rte(truth, shifted, delta=50)["rmse"] < 1e-9


def test_scale_error_shows_in_both():
    truth = _track()
    stretched = truth.copy()
    stretched[:, 1] *= 1.2

    assert compute_ate(truth, stretched)["rmse"] > 1.0
    assert compute_rte(truth, stretched, delta=50)["rmse"] > 0.1


def test_compute_all_keys():
    truth = _track()
    result = compute_all(truth, truth.copy())
    assert set(result) == {"ate", "rte", "fpe", "kitti"}
    assert set(result["ate"]) >= {"n", "rmse", "mean", "median", "std", "max"}


def test_rte_handles_short_tracks():
    """Fewer points than the segment length yields NaN, not a crash."""
    truth = _track(n=10)
    assert np.isnan(compute_rte(truth, truth.copy(), delta=100)["rmse"])
